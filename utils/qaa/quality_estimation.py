from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_TEXT_QUALITY_CACHE: dict[str, float] = {}
_AUDIO_QUALITY_CACHE: dict[str, float] = {}
_IMAGE_QUALITY_CACHE: dict[str, float] = {}
_VIDEO_QUALITY_CACHE: dict[str, float] = {}

_BRISQUE_SCORER = None
_PAM_SCORER = None
_PAM_REPO_ROOT = Path(__file__).resolve().parent.parent.parent / "PAM"

_PAM_RESAMPLE_RATE = 44100
_PAM_AUDIO_DURATION_SECONDS = 7
_PAM_SAMPLES_PER_CHUNK = _PAM_RESAMPLE_RATE * _PAM_AUDIO_DURATION_SECONDS

TEXT_QUALITY_MAX_TOKENS = max(32, int(os.getenv("TEXT_QUALITY_MAX_TOKENS", "192")))
TEXT_QUALITY_MICROBATCH_SIZE = max(1, int(os.getenv("TEXT_QUALITY_MICROBATCH_SIZE", "1")))


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
    return float(0.0 - raw_score / 100.0)


def _compute_image_brisque_score_from_qwen_image(image_input, cache_key: str | None = None) -> float:
    if cache_key is not None and cache_key in _IMAGE_QUALITY_CACHE:
        return _IMAGE_QUALITY_CACHE[cache_key]

    image_rgb = _to_pil_rgb(image_input)
    score = _compute_brisque_from_image_rgb(image_rgb)

    if cache_key is not None:
        _IMAGE_QUALITY_CACHE[cache_key] = score
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
    score = float(avg_scores[0])

    _AUDIO_QUALITY_CACHE[path_key] = score
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

    score = float(np.mean(frame_scores))
    if cache_key is not None:
        _VIDEO_QUALITY_CACHE[cache_key] = score
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

    def _estimate_text_quality_batch(batch_texts: list[str], *, max_tokens: int) -> list[float]:
        encoded = processor.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
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

        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].to(dtype=torch.float32)

        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        seq_loss = (token_loss * target_mask).sum(dim=1) / target_mask.sum(dim=1)
        ppl = torch.exp(seq_loss)
        text_quality = torch.reciprocal(torch.log(ppl))
        result = text_quality.detach().cpu().tolist()

        del outputs, logits, targets, target_mask, token_loss, seq_loss, ppl, text_quality
        del input_ids, attention_mask, encoded
        return result

    fallback_token_limits: list[int] = []
    for token_limit in (
        TEXT_QUALITY_MAX_TOKENS,
        min(TEXT_QUALITY_MAX_TOKENS, 128),
        min(TEXT_QUALITY_MAX_TOKENS, 96),
        min(TEXT_QUALITY_MAX_TOKENS, 64),
    ):
        if token_limit >= 32 and token_limit not in fallback_token_limits:
            fallback_token_limits.append(token_limit)

    def _estimate_single_with_backoff(single_text: str) -> float:
        for token_limit in fallback_token_limits:
            try:
                return float(_estimate_text_quality_batch([single_text], max_tokens=token_limit)[0])
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        raise RuntimeError(
            "Failed text perplexity estimation after OOM backoff "
            f"(token_limits={fallback_token_limits})."
        )

    inverse_perplexity: list[float] = [0.5] * len(pending_indices)
    for start in range(0, len(pending_indices), TEXT_QUALITY_MICROBATCH_SIZE):
        end = min(start + TEXT_QUALITY_MICROBATCH_SIZE, len(pending_indices))
        batch_texts = pending_texts[start:end]
        try:
            batch_scores = _estimate_text_quality_batch(batch_texts, max_tokens=TEXT_QUALITY_MAX_TOKENS)
            inverse_perplexity[start:end] = batch_scores
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[WARN] OOM in text perplexity microbatch {start}:{end}; retrying per sample.",
                    flush=True,
                )
                for offset, single_text in enumerate(batch_texts):
                    try:
                        single_score = _estimate_single_with_backoff(single_text)
                        inverse_perplexity[start + offset] = single_score
                    except Exception as single_exc:
                        print(f"[WARN] Failed text perplexity estimation: {single_exc}", flush=True)
                        inverse_perplexity[start + offset] = 0.5
            else:
                print(f"[WARN] Failed text perplexity estimation: {exc}", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed text perplexity estimation: {exc}", flush=True)

    for idx, value in zip(pending_indices, inverse_perplexity):
        normalized = (texts[idx] or "").strip()
        score = float(value)
        scores[idx] = score
        if normalized:
            _TEXT_QUALITY_CACHE[normalized] = score

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
            sample_scores["text"] = float(text_scores[idx])
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


def _find_last_subsequence_start(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    target_len = len(subsequence)
    for start in range(len(sequence) - target_len, -1, -1):
        if sequence[start : start + target_len] == subsequence:
            return start
    return None


def _build_token_quality_scores(
    input_ids,
    attention_mask,
    modality_scores_per_entry,
    thinker_config,
    text_token_ids_per_entry=None,
):
    batch_size, seq_len = input_ids.shape
    if batch_size != len(modality_scores_per_entry):
        raise ValueError(
            f"Quality-score batch mismatch: token batch={batch_size}, quality batch={len(modality_scores_per_entry)}"
        )
    if text_token_ids_per_entry is not None and batch_size != len(text_token_ids_per_entry):
        raise ValueError(
            f"Text-token batch mismatch: token batch={batch_size}, text-token batch={len(text_token_ids_per_entry)}"
        )

    quality_scores = torch.ones((batch_size, seq_len), dtype=torch.float32, device=input_ids.device)
    audio_marker_token_ids = {
        thinker_config.audio_token_index,
        thinker_config.audio_start_token_id,
        thinker_config.audio_end_token_id,
    }

    for row, sample_scores in enumerate(modality_scores_per_entry):
        if attention_mask is not None:
            valid_positions = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
        else:
            valid_positions = torch.arange(seq_len, device=input_ids.device)

        if "text" in sample_scores and text_token_ids_per_entry is not None:
            text_quality = float(sample_scores["text"])
            text_token_ids = list(text_token_ids_per_entry[row] or [])
            if text_token_ids:
                valid_token_ids = input_ids[row, valid_positions].tolist()
                start_idx = _find_last_subsequence_start(valid_token_ids, text_token_ids)
                if start_idx is None and len(text_token_ids) > 1:
                    # Fallback for tokenizer boundary effects at segment start.
                    start_idx = _find_last_subsequence_start(valid_token_ids, text_token_ids[1:])
                    if start_idx is not None:
                        text_token_ids = text_token_ids[1:]
                if start_idx is not None and text_token_ids:
                    span = valid_positions[start_idx : start_idx + len(text_token_ids)]
                    quality_scores[row, span] = text_quality

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

    return quality_scores
