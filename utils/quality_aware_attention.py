from __future__ import annotations

import math
import types

import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling


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
        raise ValueError("position_embeddings is required for quality-aware attention.")
    cos, sin = position_embeddings
    query_states, key_states = qwen_omni_modeling.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = qwen_omni_modeling.repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen_omni_modeling.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    key_len = key_states.shape[-2]
    min_dtype = torch.finfo(attn_weights.dtype).min

    # Explicit causal mask is required because this patched path bypasses flash-attn's internal causal handling.
    past_kv_len = key_len - q_len
    query_positions = past_kv_len + torch.arange(q_len, device=attn_weights.device)
    key_positions = torch.arange(key_len, device=attn_weights.device)
    causal_positions = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    attn_weights = attn_weights.masked_fill(causal_positions.unsqueeze(0).unsqueeze(0), min_dtype)

    if attention_mask is not None:
        if attention_mask.dim() == 4:
            causal_mask = attention_mask[:, :, :, :key_len]
            attn_weights = attn_weights + causal_mask
        elif attention_mask.dim() == 2:
            padding_mask = attention_mask[:, :key_len].to(torch.bool)
            attn_weights = attn_weights.masked_fill(~padding_mask[:, None, None, :], min_dtype)
        else:
            raise ValueError(
                f"Unsupported attention_mask dims={attention_mask.dim()} for first-layer quality attention."
            )

    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    quality_scores = getattr(self, "_quality_scores", None)
    if quality_scores is not None:
        quality_scores = quality_scores.to(device=attn_weights.device, dtype=attn_weights.dtype)
        if quality_scores.dim() == 1:
            quality_scores = quality_scores.unsqueeze(0)
        if quality_scores.size(0) != bsz:
            if bsz % quality_scores.size(0) == 0:
                repeat_factor = bsz // quality_scores.size(0)
                quality_scores = quality_scores.repeat_interleave(repeat_factor, dim=0)
            else:
                raise ValueError(
                    f"First-layer quality score batch mismatch: got {quality_scores.size(0)}, expected {bsz}."
                )

        key_len = key_states.shape[-2]
        if quality_scores.size(1) < key_len:
            pad = torch.ones(
                (quality_scores.size(0), key_len - quality_scores.size(1)),
                device=quality_scores.device,
                dtype=quality_scores.dtype,
            )
            quality_scores = torch.cat([quality_scores, pad], dim=1)
        elif quality_scores.size(1) > key_len:
            quality_scores = quality_scores[:, :key_len]

        attn_weights = attn_weights * quality_scores[:, None, None, :]

    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def install_quality_aware_first_attention_patch(model):
    first_attn = model.thinker.model.layers[0].self_attn
    if getattr(first_attn, "_quality_patch_installed", False):
        return

    first_attn._quality_scores = None
    first_attn._quality_patch_installed = True
    first_attn._original_forward = first_attn.forward
    first_attn.forward = types.MethodType(_quality_aware_forward, first_attn)


def set_first_layer_quality_scores(model, quality_scores: torch.Tensor | None):
    model.thinker.model.layers[0].self_attn._quality_scores = quality_scores
