from __future__ import annotations

import hashlib
import re

import torch
from qwen_omni_utils import process_mm_info

from utils.parsing_util import extract_assistant_reply

_QWEN_MODALITY_CACHE: dict[str, dict[str, float]] = {
    "text": {},
    "audio": {},
    "image": {},
    "video": {},
}

_SCORE_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

_SYSTEM_PROMPT = (
    "You are an objective multimodal quality assessor. "
    "Return only one number between 0 and 1 where 1 means best quality."
)

_MODALITY_INSTRUCTION = {
    "text": (
        "Rate the quality of this text input from 0 to 1, where 1 is best quality. "
        "Output only the numeric score."
    ),
    "audio": (
        "Rate the perceptual quality of this audio from 0 to 1, where 1 is best quality. "
        "Output only the numeric score."
    ),
    "image": (
        "Rate the perceptual quality of this image from 0 to 1, where 1 is best quality. "
        "Output only the numeric score."
    ),
    "video": (
        "Rate the perceptual quality of this video from 0 to 1, where 1 is best quality. "
        "Output only the numeric score."
    ),
}


def _cast_floats_to_dtype(batch, dtype: torch.dtype):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            batch[key] = value.to(dtype)
    return batch


def _parse_quality_score(decoded_output: str) -> float | None:
    assistant_reply = extract_assistant_reply(decoded_output).strip()
    match = _SCORE_PATTERN.search(assistant_reply)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _cache_key_for_sample(modality: str, sample) -> str:
    if modality == "text":
        normalized_text = (sample.get("text") or "").strip()
        if not normalized_text:
            return ""
        return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()

    path_value = sample.get(modality)
    return str(path_value) if path_value else ""


def _build_quality_conversation(modality: str, sample):
    user_content = []

    if modality == "text":
        text_value = (sample.get("text") or "").strip()
        if not text_value:
            return None
        user_content.append({"type": "text", "text": f"Text input:\n{text_value}"})
    elif modality == "audio":
        audio_path = sample.get("audio")
        if not audio_path:
            return None
        user_content.append({"type": "audio", "audio": str(audio_path)})
    elif modality == "image":
        image_path = sample.get("image")
        if not image_path:
            return None
        user_content.append({"type": "image", "image": str(image_path)})
    elif modality == "video":
        video_path = sample.get("video")
        if not video_path:
            return None
        user_content.append({"type": "video", "video": str(video_path)})
    else:
        raise ValueError(f"Unsupported modality for Qwen quality scoring: {modality}")

    user_content.append({"type": "text", "text": _MODALITY_INSTRUCTION[modality]})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _score_modality_batch_with_qwen(
    modality: str,
    conversations,
    model,
    processor,
    device,
    dtype,
) -> list[float]:
    if not conversations:
        return []

    text_prompt = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    return_video_metadata = modality == "video"
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
    if modality == "audio":
        proc_kwargs["audio"] = audios
    elif modality == "image":
        proc_kwargs["images"] = images
    elif modality == "video":
        proc_kwargs["videos"] = videos_for_processor

    inputs = processor(**proc_kwargs).to(device)
    inputs = _cast_floats_to_dtype(inputs, dtype)

    gen_output = model.generate(
        **inputs,
        use_audio_in_video=False,
        return_audio=False,
        do_sample=False,
        max_new_tokens=8,
    )

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
    if len(decoded) != len(conversations):
        raise RuntimeError(
            f"Decoded quality outputs ({len(decoded)}) do not match batch size ({len(conversations)})."
        )

    scores: list[float] = []
    for idx, output_text in enumerate(decoded):
        parsed = _parse_quality_score(output_text)
        if parsed is None:
            reply = extract_assistant_reply(output_text).strip().replace("\n", " ")
            print(
                f"[WARN] Failed to parse Qwen quality score for {modality} sample {idx}; "
                f"defaulting to 0.5. Reply={reply!r}",
                flush=True,
            )
            parsed = 0.5
        scores.append(parsed)

    return scores


def compute_batch_modality_quality_scores_with_qwen(
    entries,
    enabled_modalities,
    model,
    processor,
    device,
    dtype,
):
    modality_scores_per_entry = [{} for _ in entries]

    for modality in ("text", "audio", "image", "video"):
        if modality not in enabled_modalities:
            continue

        cache = _QWEN_MODALITY_CACHE[modality]
        pending_indices = []
        pending_conversations = []
        pending_cache_keys = []

        for entry_idx, entry in enumerate(entries):
            sample = entry["sample"]
            conversation = _build_quality_conversation(modality, sample)
            if conversation is None:
                continue

            cache_key = _cache_key_for_sample(modality, sample)
            if cache_key and cache_key in cache:
                modality_scores_per_entry[entry_idx][modality] = float(cache[cache_key])
                continue

            pending_indices.append(entry_idx)
            pending_conversations.append(conversation)
            pending_cache_keys.append(cache_key)

        if not pending_conversations:
            continue

        batch_scores = _score_modality_batch_with_qwen(
            modality=modality,
            conversations=pending_conversations,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )

        if len(batch_scores) != len(pending_indices):
            raise RuntimeError(
                f"Qwen quality score count mismatch for modality {modality}: "
                f"scores={len(batch_scores)} pending={len(pending_indices)}"
            )

        for idx, score, cache_key in zip(pending_indices, batch_scores, pending_cache_keys):
            modality_scores_per_entry[idx][modality] = float(score)
            if cache_key:
                cache[cache_key] = float(score)

    return modality_scores_per_entry
