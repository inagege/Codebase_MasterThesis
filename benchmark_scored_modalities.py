import argparse
import csv
import math
import os
import sys
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling

from utils.benchmark_data_loading import (
    default_modalities_for_dataset,
    filter_samples_by_sample_id,
    get_prompt_for_classification,
    load_samples,
    normalize_dataset_name,
    normalize_meld_task,
    normalize_modalities,
    select_stratified_samples,
    validate_modalities,
)
from utils.parsing_util import extract_assistant_reply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="meld",
        help="Dataset to benchmark: meld, homeprice, imdb, voxceleb, nejm, marine.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated list from text,audio,video,image. If omitted, all modalities available in the selected dataset are used.",
    )
    parser.add_argument(
        "--noisy-modalities",
        type=str,
        default=None,
        help="Comma-separated modalities that should use noisy input variants.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split(s) to process. For MELD: train,val,test (comma-separated). Others: ignored and treated as all/dev.",
    )
    parser.add_argument(
        "--classification-task",
        type=str,
        default=None,
        help="MELD only: sentiment|emotion.",
    )
    parser.add_argument(
        "--start-at-sample",
        type=int,
        default=None,
        help="Optional lower bound (0-based). If set, skip all samples before this index.",
    )
    parser.add_argument(
        "--stratified-samples",
        type=int,
        default=None,
        help="Non-MELD only: deterministically select this many samples, stratified by label.",
    )
    parser.add_argument("--total-samples", type=int, default=None, help="Limit total files across all splits")
    parser.add_argument("--audio-subdir", type=str, default="audio_only", help="Subdir for WAV files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per generation batch. Reduce if you hit CUDA OOM.",
    )
    parser.add_argument("--out-path", type=str, default="out/prediction_noise.csv")
    parser.add_argument("--out-error-path", type=str, default="out/error_prediction_noise.csv")
    return parser.parse_args()


_TEXT_QUALITY_CACHE: dict[str, float] = {}
_AUDIO_QUALITY_CACHE: dict[str, float] = {}
_IMAGE_QUALITY_CACHE: dict[str, float] = {}
_VIDEO_QUALITY_CACHE: dict[str, float] = {}

_BRISQUE_SCORER = None
_PAM_SCORER = None
_PAM_REPO_ROOT = Path(__file__).resolve().parent / "PAM"

_PAM_RESAMPLE_RATE = 44100
_PAM_AUDIO_DURATION_SECONDS = 7
_PAM_SAMPLES_PER_CHUNK = _PAM_RESAMPLE_RATE * _PAM_AUDIO_DURATION_SECONDS

# Calibrates 1/(1+log1p(ppl)) upward so typical clean text lands near ~0.9.
TEXT_QUALITY_LOG_SCALE = 0.04


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _get_brisque_scorer():
    global _BRISQUE_SCORER
    if _BRISQUE_SCORER is not None:
        return _BRISQUE_SCORER
    try:
        from brisque import BRISQUE
    except Exception as exc:
        raise RuntimeError(
            "The `brisque` package is required but could not be imported. "
            "Install it in the active environment and rerun."
        ) from exc
    _BRISQUE_SCORER = BRISQUE()
    return _BRISQUE_SCORER


def _get_pam_scorer(device):
    global _PAM_SCORER
    if _PAM_SCORER is not None:
        return _PAM_SCORER

    if not _PAM_REPO_ROOT.exists():
        raise RuntimeError(f"PAM repository not found at {_PAM_REPO_ROOT}")
    pam_repo_str = str(_PAM_REPO_ROOT)
    if pam_repo_str not in sys.path:
        sys.path.insert(0, pam_repo_str)

    try:
        from PAM import PAM as PAMMetric
    except Exception as exc:
        raise RuntimeError(
            "Failed to import PAM from the local cloned repository. "
            f"Expected import path root: {_PAM_REPO_ROOT}"
        ) from exc

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    _PAM_SCORER = PAMMetric(use_cuda=use_cuda)
    return _PAM_SCORER


def _to_pil_rgb(image_input):
    from PIL import Image

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, torch.Tensor):
        image_np = image_input.detach().cpu().numpy()
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
            image_np = np.transpose(image_np, (1, 2, 0))
    elif isinstance(image_input, np.ndarray):
        image_np = image_input
    else:
        raise RuntimeError(f"Unsupported image input type for BRISQUE: {type(image_input)}")

    if image_np.ndim != 3:
        raise RuntimeError(f"Expected image tensor/array to have 3 dims, got shape {image_np.shape}")
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, repeats=3, axis=-1)

    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np).convert("RGB")


def _compute_brisque_from_image_rgb(image_rgb) -> float:
    scorer = _get_brisque_scorer()
    raw_score = float(scorer.score(np.asarray(image_rgb.convert("RGB"))))
    return _clamp01(1.0 - raw_score / 100.0)


def _compute_image_brisque_score_from_qwen_image(image_input, cache_key: str | None = None) -> float:
    if cache_key is not None and cache_key in _IMAGE_QUALITY_CACHE:
        return _IMAGE_QUALITY_CACHE[cache_key]

    image_rgb = _to_pil_rgb(image_input)
    score = _compute_brisque_from_image_rgb(image_rgb)

    if cache_key is not None:
        _IMAGE_QUALITY_CACHE[cache_key] = score
    print("Printing Image scores")
    print(score)
    return score


def _prepare_audio_chunks_for_pam(audio_path: str):
    import torchaudio
    import torchaudio.transforms as T

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise RuntimeError(f"Audio file has no samples: {audio_path}")

    if sample_rate != _PAM_RESAMPLE_RATE:
        resampler = T.Resample(sample_rate, _PAM_RESAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.reshape(-1)
    if waveform.shape[0] < 1:
        raise RuntimeError(f"Audio file has invalid waveform after reshape: {audio_path}")

    if _PAM_SAMPLES_PER_CHUNK >= waveform.shape[0]:
        repeat_factor = int(np.ceil(_PAM_SAMPLES_PER_CHUNK / waveform.shape[0]))
        waveform = waveform.repeat(repeat_factor)
        waveform = waveform[:_PAM_SAMPLES_PER_CHUNK]
    else:
        cutoff = int(np.floor(waveform.shape[0] / _PAM_SAMPLES_PER_CHUNK))
        initial_waveform = waveform[: cutoff * _PAM_SAMPLES_PER_CHUNK]
        remaining = waveform[cutoff * _PAM_SAMPLES_PER_CHUNK :]
        if remaining.shape[0] != 0:
            remaining = waveform[-_PAM_SAMPLES_PER_CHUNK :]
            waveform = torch.cat([initial_waveform, remaining])
        else:
            waveform = initial_waveform

    chunk_count = int(waveform.shape[0] / _PAM_SAMPLES_PER_CHUNK)
    if chunk_count < 1:
        raise RuntimeError(f"Could not derive PAM chunks for audio: {audio_path}")

    chunks = [
        waveform[_PAM_SAMPLES_PER_CHUNK * i : _PAM_SAMPLES_PER_CHUNK * (i + 1)].reshape(1, -1)
        for i in range(chunk_count)
    ]
    audio_chunks = torch.cat(chunks, dim=0)
    sample_index = [0, chunk_count]
    return audio_chunks, sample_index


def _compute_audio_pam_score(audio_path: str, device) -> float:
    path_key = str(audio_path)
    if path_key in _AUDIO_QUALITY_CACHE:
        return _AUDIO_QUALITY_CACHE[path_key]

    pam_scorer = _get_pam_scorer(device)
    audio_chunks, sample_index = _prepare_audio_chunks_for_pam(path_key)
    avg_scores, _ = pam_scorer.evaluate(audio_chunks, sample_index)
    if not avg_scores:
        raise RuntimeError(f"PAM did not return scores for {path_key}")
    score = _clamp01(float(avg_scores[0]))

    _AUDIO_QUALITY_CACHE[path_key] = score
    print("Printing Audio scores")
    print(score)
    return score


def _extract_video_tensor(video_input):
    video_tensor = video_input[0] if isinstance(video_input, tuple) else video_input
    if not torch.is_tensor(video_tensor):
        raise RuntimeError(f"Expected torch.Tensor video input, got {type(video_tensor)}")
    if video_tensor.ndim != 4:
        raise RuntimeError(f"Expected video tensor shape (T,C,H,W), got {tuple(video_tensor.shape)}")
    return video_tensor


def _compute_video_brisque_score_from_qwen_video(video_input, cache_key: str | None = None) -> float:
    if cache_key is not None and cache_key in _VIDEO_QUALITY_CACHE:
        return _VIDEO_QUALITY_CACHE[cache_key]

    video_tensor = _extract_video_tensor(video_input)
    frame_scores = []
    for frame_tensor in video_tensor:
        frame_rgb = _to_pil_rgb(frame_tensor)
        frame_scores.append(_compute_brisque_from_image_rgb(frame_rgb))
    if not frame_scores:
        raise RuntimeError("Video has zero sampled frames after Qwen preprocessing.")

    score = _clamp01(float(np.mean(frame_scores)))
    if cache_key is not None:
        _VIDEO_QUALITY_CACHE[cache_key] = score
    print("Printing Video scores")
    print(score)
    return score


def _compute_text_inverse_perplexities(texts: list[str], model, processor, device) -> list[float]:
    scores = [1.0] * len(texts)
    pending_indices = []
    pending_texts = []

    for idx, text in enumerate(texts):
        normalized = (text or "").strip()
        if not normalized:
            continue
        cached = _TEXT_QUALITY_CACHE.get(normalized)
        if cached is not None:
            scores[idx] = cached
            continue
        pending_indices.append(idx)
        pending_texts.append(normalized)

    if not pending_texts:
        return scores

    try:
        encoded = processor.tokenizer(
            pending_texts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.inference_mode():
            outputs = model.thinker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_audio_in_video=False,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].to(dtype=torch.float32)

        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        seq_loss = (token_loss * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1.0)
        ppl = torch.exp(seq_loss.clamp(max=50.0))
        text_quality = 1.0 / (1.0 + TEXT_QUALITY_LOG_SCALE * torch.log1p(ppl))
        inverse_perplexity = text_quality.clamp(0.0, 1.0).detach().cpu().tolist()
    except Exception as exc:
        print(f"[WARN] Failed text perplexity estimation: {exc}", flush=True)
        inverse_perplexity = [0.5] * len(pending_indices)

    for idx, value in zip(pending_indices, inverse_perplexity):
        normalized = (texts[idx] or "").strip()
        score = _clamp01(float(value))
        scores[idx] = score
        if normalized:
            _TEXT_QUALITY_CACHE[normalized] = score
    print("Printing Text scores")
    print(scores)
    
    return scores


def _compute_batch_modality_quality_scores(
    entries,
    enabled_modalities,
    model,
    processor,
    device,
    qwen_images,
    qwen_videos,
):
    text_scores = [1.0] * len(entries)
    if "text" in enabled_modalities:
        text_inputs = [entry["sample"].get("text", "") for entry in entries]
        text_scores = _compute_text_inverse_perplexities(text_inputs, model, processor, device)

    image_items = list(qwen_images or [])
    video_items = list(qwen_videos or [])
    image_idx = 0
    video_idx = 0

    modality_scores_per_entry = []
    for idx, entry in enumerate(entries):
        sample = entry["sample"]
        sample_scores: dict[str, float] = {}

        if "text" in enabled_modalities and sample.get("text", "").strip():
            sample_scores["text"] = _clamp01(text_scores[idx])
        if "audio" in enabled_modalities and sample.get("audio"):
            sample_scores["audio"] = _compute_audio_pam_score(sample["audio"], device=device)
        if "image" in enabled_modalities and sample.get("image"):
            if image_idx >= len(image_items):
                raise RuntimeError(
                    "Qwen image inputs are shorter than expected. "
                    "Cannot align BRISQUE scoring to model image inputs."
                )
            sample_scores["image"] = _compute_image_brisque_score_from_qwen_image(
                image_items[image_idx],
                cache_key=str(sample["image"]),
            )
            image_idx += 1
        if "video" in enabled_modalities and sample.get("video"):
            if video_idx >= len(video_items):
                raise RuntimeError(
                    "Qwen video inputs are shorter than expected. "
                    "Cannot align BRISQUE scoring to model video frames."
                )
            sample_scores["video"] = _compute_video_brisque_score_from_qwen_video(
                video_items[video_idx],
                cache_key=str(sample["video"]),
            )
            video_idx += 1

        modality_scores_per_entry.append(sample_scores)

    if image_idx != len(image_items):
        raise RuntimeError(
            f"Unused Qwen image inputs during quality scoring: used={image_idx}, available={len(image_items)}"
        )
    if video_idx != len(video_items):
        raise RuntimeError(
            f"Unused Qwen video inputs during quality scoring: used={video_idx}, available={len(video_items)}"
        )

    return modality_scores_per_entry


def _build_token_quality_scores(input_ids, attention_mask, modality_scores_per_entry, thinker_config):
    batch_size, seq_len = input_ids.shape
    if batch_size != len(modality_scores_per_entry):
        raise ValueError(
            f"Quality-score batch mismatch: token batch={batch_size}, quality batch={len(modality_scores_per_entry)}"
        )

    quality_scores = torch.ones((batch_size, seq_len), dtype=torch.float32, device=input_ids.device)
    audio_marker_token_ids = {
        thinker_config.audio_token_index,
        thinker_config.audio_start_token_id,
        thinker_config.audio_end_token_id,
    }

    for row, sample_scores in enumerate(modality_scores_per_entry):
        text_quality = float(sample_scores.get("text", 1.0))
        quality_scores[row].fill_(_clamp01(text_quality))

        if "audio" in sample_scores:
            audio_quality = float(sample_scores["audio"])
            for token_id in audio_marker_token_ids:
                quality_scores[row][input_ids[row] == token_id] = audio_quality
        if "image" in sample_scores:
            quality_scores[row][input_ids[row] == thinker_config.image_token_index] = float(sample_scores["image"])
        if "video" in sample_scores:
            quality_scores[row][input_ids[row] == thinker_config.video_token_index] = float(sample_scores["video"])

        if attention_mask is not None:
            quality_scores[row][attention_mask[row] == 0] = 1.0
    
    print("printing quality scores:")
    print(quality_scores)

    return quality_scores


def _install_quality_aware_first_attention_patch(model):
    first_attn = model.thinker.model.layers[0].self_attn
    if getattr(first_attn, "_quality_patch_installed", False):
        return

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

    first_attn._quality_scores = None
    first_attn._quality_patch_installed = True
    first_attn._original_forward = first_attn.forward
    first_attn.forward = types.MethodType(_quality_aware_forward, first_attn)


def _set_first_layer_quality_scores(model, quality_scores: torch.Tensor | None):
    model.thinker.model.layers[0].self_attn._quality_scores = quality_scores


def append_csv_row(path: str, fieldnames: list[str], row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def cast_floats_to_dtype(batch, dtype: torch.dtype):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            batch[key] = value.to(dtype)
    return batch


def log_error_row(args, err_fieldnames, dataset, sample, error_text, traceback_text=""):
    append_csv_row(
        args.out_error_path,
        err_fieldnames,
        {
            "dataset": dataset,
            "split": sample["split"],
            "sample_id": sample["sample_id"],
            "file": sample["file"],
            "error": error_text,
            "traceback": traceback_text,
        },
    )


def build_user_content(enabled_modalities, sample, args, err_fieldnames, dataset):
    user_content = []
    skip_sample = False

    if "video" in enabled_modalities:
        video_path = sample.get("video")
        if video_path is None or not Path(video_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing video file: {video_path}")
            skip_sample = True
        else:
            user_content.append({"type": "video", "video": str(video_path)})

    if "audio" in enabled_modalities:
        audio_path = sample.get("audio")
        if audio_path is None or not Path(audio_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing audio file: {audio_path}")
            skip_sample = True
        else:
            user_content.append({"type": "audio", "audio": str(audio_path)})

    if "image" in enabled_modalities:
        image_path = sample.get("image")
        if image_path is None or not Path(image_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing image file: {image_path}")
            skip_sample = True
        else:
            user_content.append({"type": "image", "image": str(image_path)})

    if "text" in enabled_modalities:
        text_value = sample.get("text", "").strip()
        if not text_value:
            log_error_row(args, err_fieldnames, dataset, sample, "Missing text input.")
            skip_sample = True
        else:
            user_content.append({"type": "text", "text": text_value})

    return user_content, skip_sample


def build_conversation_for_sample(sample, dataset, prompt, enabled_modalities, args, err_fieldnames):
    sample_prompt = prompt
    if dataset == "nejm":
        options = (sample.get("options") or "").strip()
        if options:
            sample_prompt = f"{prompt} Options: {options}"

    system_entry = {
        "role": "system",
        "content": [{"type": "text", "text": sample_prompt}],
    }

    user_content, skip_sample = build_user_content(
        enabled_modalities=enabled_modalities,
        sample=sample,
        args=args,
        err_fieldnames=err_fieldnames,
        dataset=dataset,
    )
    if skip_sample:
        return None

    return [system_entry, {"role": "user", "content": user_content}]


def run_batch_generation(model, processor, entries, enabled_modalities, device, dtype):
    conversations = [entry["conversation"] for entry in entries]
    text_prompt = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    return_video_metadata = "video" in enabled_modalities
    audios, images, videos = process_mm_info(
        conversations,
        use_audio_in_video=False,
        return_video_metadata=return_video_metadata,
    )

    videos_for_processor = videos
    if return_video_metadata and videos is not None:
        videos_for_processor = [video_item[0] if isinstance(video_item, tuple) else video_item for video_item in videos]

    proc_kwargs = {
        "text": text_prompt,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": False,
    }
    if "audio" in enabled_modalities:
        proc_kwargs["audio"] = audios
    if "video" in enabled_modalities:
        proc_kwargs["videos"] = videos_for_processor
    if "image" in enabled_modalities:
        proc_kwargs["images"] = images

    inputs = processor(**proc_kwargs).to(device)
    inputs = cast_floats_to_dtype(inputs, dtype)

    modality_quality_scores = _compute_batch_modality_quality_scores(
        entries=entries,
        enabled_modalities=enabled_modalities,
        model=model,
        processor=processor,
        device=device,
        qwen_images=images,
        qwen_videos=videos,
    )
    token_quality_scores = _build_token_quality_scores(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        modality_scores_per_entry=modality_quality_scores,
        thinker_config=model.thinker.config,
    )

    _set_first_layer_quality_scores(model, token_quality_scores)
    try:
        gen_output = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            output_scores=True,
            do_sample=False,
        )
    finally:
        _set_first_layer_quality_scores(model, None)

    text_ids = (
        gen_output.sequences
        if hasattr(gen_output, "sequences")
        else gen_output[0]
        if isinstance(gen_output, (list, tuple))
        else gen_output
    )
    if isinstance(text_ids, torch.Tensor):
        text_ids = text_ids.cpu()

    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if isinstance(decoded, str):
        decoded = [decoded]
    if len(decoded) != len(entries):
        raise RuntimeError(
            f"Decoded outputs ({len(decoded)}) do not match batch size ({len(entries)})."
        )

    return [extract_assistant_reply(out_text) for out_text in decoded]


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.start_at_sample is not None and args.start_at_sample < 0:
        raise ValueError("--start-at-sample must be >= 0 when provided.")
    if args.stratified_samples is not None and args.stratified_samples < 1:
        raise ValueError("--stratified-samples must be >= 1 when provided.")

    dataset = normalize_dataset_name(args.dataset)
    enabled_modalities = normalize_modalities(args.modalities)
    if enabled_modalities is None:
        enabled_modalities = default_modalities_for_dataset(dataset)

    noisy_modalities = normalize_modalities(args.noisy_modalities)
    validate_modalities(dataset, enabled_modalities, noisy_modalities)

    meld_task = None
    label_column = None
    if dataset == "meld":
        meld_task, label_column = normalize_meld_task(args.classification_task)
    elif dataset == "voxceleb":
        label_column = "nationality_wiki"

    prompt = get_prompt_for_classification(dataset, meld_task)

    print(f"[INFO] CWD: {os.getcwd()}", flush=True)
    print(f"[INFO] Dataset: {dataset}", flush=True)
    print(f"[INFO] Modalities enabled: {sorted(enabled_modalities)}", flush=True)
    print(f"[INFO] Noisy modalities: {args.noisy_modalities}", flush=True)
    print(f"[INFO] Split: {args.split}", flush=True)
    print(f"[INFO] audio_subdir={args.audio_subdir}", flush=True)
    print(f"[INFO] batch_size={args.batch_size}", flush=True)
    print(f"[INFO] start_at_sample={args.start_at_sample}", flush=True)
    print(f"[INFO] stratified_samples={args.stratified_samples}", flush=True)
    print(f"[INFO] total_samples={args.total_samples}", flush=True)
    if label_column is not None:
        print(f"[INFO] Label column: {label_column}", flush=True)
    print(f"[INFO] out_path={args.out_path}", flush=True)
    print(f"[INFO] out_error_path={args.out_error_path}", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_error_path) or ".", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    samples = load_samples(dataset, args, enabled_modalities, noisy_modalities, label_column)
    if args.stratified_samples is not None:
        if dataset == "meld":
            print(
                "[INFO] Ignoring --stratified-samples for MELD (explicit train/val/test splits are already defined).",
                flush=True,
            )
        else:
            if noisy_modalities is None:
                before_count = len(samples)
                if args.stratified_samples >= before_count:
                    print(
                        "[INFO] --stratified-samples is >= available samples; selecting all samples.",
                        flush=True,
                    )
                else:
                    samples = select_stratified_samples(samples, args.stratified_samples)
                    print(
                        f"[INFO] Applied deterministic stratified sampling: {before_count} -> {len(samples)} samples",
                        flush=True,
                    )
            else:
                base_samples = load_samples(dataset, args, enabled_modalities, None, label_column)
                base_before_count = len(base_samples)
                if args.stratified_samples >= base_before_count:
                    selected_base_samples = base_samples
                    print(
                        "[INFO] --stratified-samples is >= available unmodified samples; selecting all base samples.",
                        flush=True,
                    )
                else:
                    selected_base_samples = select_stratified_samples(base_samples, args.stratified_samples)
                    print(
                        "[INFO] Applied deterministic stratified sampling on unmodified data: "
                        f"{base_before_count} -> {len(selected_base_samples)} base samples",
                        flush=True,
                    )

                selected_base_ids = {sample.get("sample_id") for sample in selected_base_samples}
                noisy_before_count = len(samples)
                samples = filter_samples_by_sample_id(samples, selected_base_ids)
                print(
                    "[INFO] Expanded selected base samples across noisy variants: "
                    f"{noisy_before_count} -> {len(samples)} noisy samples",
                    flush=True,
                )
    if args.start_at_sample is not None:
        samples = samples[args.start_at_sample :]
    if args.total_samples is not None:
        samples = samples[: args.total_samples]

    print(f"[INFO] Total samples to process: {len(samples)}", flush=True)
    if not samples:
        raise RuntimeError("No samples found for the selected dataset/modality configuration.")

    print("[INFO] Loading model...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        enable_audio_output=False,
    )
    model.disable_talker()
    _install_quality_aware_first_attention_patch(model)
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[INFO] Model loaded.", flush=True)
    print("[INFO] Quality-aware first attention layer patch enabled.", flush=True)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"[INFO] device={device} dtype={dtype}", flush=True)

    fieldnames = [
        "dataset",
        "split",
        "sample_id",
        "file",
        "prediction",
        "label",
    ]
    err_fieldnames = [
        "dataset",
        "split",
        "sample_id",
        "file",
        "error",
        "traceback",
    ]

    num_written = 0
    for batch_start in range(0, len(samples), args.batch_size):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        entries = []

        for sample in batch_samples:
            conversation = build_conversation_for_sample(
                sample=sample,
                dataset=dataset,
                prompt=prompt,
                enabled_modalities=enabled_modalities,
                args=args,
                err_fieldnames=err_fieldnames,
            )
            if conversation is None:
                continue
            entries.append({"sample": sample, "conversation": conversation})

        if not entries:
            continue

        try:
            replies = run_batch_generation(
                model=model,
                processor=processor,
                entries=entries,
                enabled_modalities=enabled_modalities,
                device=device,
                dtype=dtype,
            )
        except Exception:
            print(
                f"[WARN] Batch {batch_start + 1}-{batch_start + len(batch_samples)} failed; retrying individually.",
                flush=True,
            )
            for entry in entries:
                sample = entry["sample"]
                try:
                    replies = run_batch_generation(
                        model=model,
                        processor=processor,
                        entries=[entry],
                        enabled_modalities=enabled_modalities,
                        device=device,
                        dtype=dtype,
                    )
                    reply = replies[0]
                    append_csv_row(
                        args.out_path,
                        fieldnames,
                        {
                            "dataset": dataset,
                            "split": sample["split"],
                            "sample_id": sample["sample_id"],
                            "file": sample["file"],
                            "prediction": reply,
                            "label": sample.get("label", "unknown"),
                        },
                    )
                    num_written += 1
                    if num_written % 20 == 0 or num_written == 1:
                        print(f"[INFO] Wrote prediction {num_written}/{len(samples)} -> {args.out_path}", flush=True)
                except Exception as exc:
                    log_error_row(
                        args=args,
                        err_fieldnames=err_fieldnames,
                        dataset=dataset,
                        sample=sample,
                        error_text=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
            continue

        for entry, reply in zip(entries, replies):
            sample = entry["sample"]
            append_csv_row(
                args.out_path,
                fieldnames,
                {
                    "dataset": dataset,
                    "split": sample["split"],
                    "sample_id": sample["sample_id"],
                    "file": sample["file"],
                    "prediction": reply,
                    "label": sample.get("label", "unknown"),
                },
            )
            num_written += 1
            if num_written % 20 == 0 or num_written == 1:
                print(f"[INFO] Wrote prediction {num_written}/{len(samples)} -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
