from __future__ import annotations

import types

import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling


def _align_quality_scores_to_kv_length(
    quality_scores: torch.Tensor,
    *,
    batch_size: int,
    kv_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quality_scores = quality_scores.to(device=device, dtype=dtype)
    if quality_scores.dim() == 1:
        quality_scores = quality_scores.unsqueeze(0)
    if quality_scores.dim() != 2:
        raise ValueError(
            f"First-layer quality scores must be rank-1 or rank-2, got shape {tuple(quality_scores.shape)}."
        )

    quality_batch = quality_scores.size(0)
    if quality_batch != batch_size:
        if quality_batch == 1:
            quality_scores = quality_scores.expand(batch_size, -1)
        elif batch_size % quality_batch == 0:
            repeat_factor = batch_size // quality_batch
            quality_scores = quality_scores.repeat_interleave(repeat_factor, dim=0)
        else:
            raise ValueError(
                f"First-layer quality score batch mismatch: got {quality_batch}, expected {batch_size}."
            )

    quality_len = quality_scores.size(1)
    if quality_len == kv_seq_len:
        return quality_scores
    if quality_len > kv_seq_len:
        return quality_scores[:, :kv_seq_len]
    if quality_len == 1:
        return quality_scores.repeat(1, kv_seq_len)

    pad = torch.ones(
        (quality_scores.size(0), kv_seq_len - quality_len),
        device=quality_scores.device,
        dtype=quality_scores.dtype,
    )
    return torch.cat([quality_scores, pad], dim=1)


def _quality_aware_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        raise ValueError("position_embeddings is required for quality-aware first-layer flash attention.")
    cos, sin = position_embeddings
    query_states, key_states = qwen_omni_modeling.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = qwen_omni_modeling.repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen_omni_modeling.repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # Mirrors HF Qwen2_5OmniFlashAttention2.forward dtype handling.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        qwen_omni_modeling.logger.warning_once(
            "The input hidden states seems to be silently casted in float32, this might be related to "
            "upcasted embedding or layer norm layers in float32. Casting back for flash attention."
        )
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    quality_scores = getattr(self, "_quality_scores", None)
    if quality_scores is not None:
        quality_scores = _align_quality_scores_to_kv_length(
            quality_scores,
            batch_size=bsz,
            #kv_seq_len=key_states.shape[-2],
            #device=key_states.device,
            #dtype=key_states.dtype,

            kv_seq_len=value_states.shape[-2],
            device=value_states.device,
            dtype=value_states.dtype,
        )

        # Per-key quality scaling on keys, i.e. before softmax in attention logits.
        # key_states = key_states * quality_scores[:, None, :, None]
        # Per-key quality scaling on values (after softmax weighting).
        value_states = value_states * quality_scores[:, None, :, None]

    # Reashape to the expected shape for Flash Attention.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = qwen_omni_modeling._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None

    return attn_output, attn_weights, past_key_value


def _get_first_layer_self_attention(model):
    try:
        return model.thinker.model.layers[0].self_attn
    except Exception as exc:
        raise RuntimeError("Could not locate model.thinker.model.layers[0].self_attn for patch installation.") from exc


def install_quality_aware_first_attention_patch(model):
    first_attn = _get_first_layer_self_attention(model)
    if getattr(first_attn, "_quality_patch_installed", False):
        return
    if not hasattr(qwen_omni_modeling, "_flash_attention_forward"):
        raise RuntimeError(
            "Flash attention forward helper is unavailable in transformers runtime; cannot install flash patch."
        )

    configured_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if configured_impl is None:
        configured_impl = getattr(getattr(model.thinker, "config", None), "_attn_implementation", None)
    if configured_impl is not None and configured_impl != "flash_attention_2":
        raise RuntimeError(
            f"Expected attn_implementation='flash_attention_2' for first-layer quality patch, got {configured_impl!r}."
        )

    expected_flash_cls = getattr(qwen_omni_modeling, "Qwen2_5OmniFlashAttention2", None)
    if expected_flash_cls is not None and not isinstance(first_attn, expected_flash_cls):
        raise RuntimeError(
            "First thinker layer self-attention is not Qwen2_5OmniFlashAttention2; refusing to install flash patch."
        )
    if expected_flash_cls is None and "FlashAttention2" not in first_attn.__class__.__name__:
        raise RuntimeError(
            "First thinker layer self-attention does not look like FlashAttention2; refusing to install flash patch."
        )

    first_attn._quality_scores = None
    first_attn._quality_patch_installed = True
    first_attn._original_forward = first_attn.forward
    first_attn.forward = types.MethodType(_quality_aware_forward, first_attn)


def set_first_layer_quality_scores(model, quality_scores: torch.Tensor | None):
    _get_first_layer_self_attention(model)._quality_scores = quality_scores
