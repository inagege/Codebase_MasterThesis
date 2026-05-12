from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import types

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling

from utils.benchmark_data_loading import (
    default_modalities_for_dataset,
    get_prompt_for_classification,
    load_samples,
    normalize_dataset_name,
    normalize_meld_task,
    normalize_modalities,
    normalize_voxceleb_label_column,
    select_stratified_samples,
    validate_modalities,
)
from utils.calibration.quality_calibration import (
    SUPPORTED_MODALITIES,
    load_percentile_calibration,
)
from utils.qaa.quality_aware_attention import (
    QAA_NORMALIZATION_EXCLUDE_UNSCALED,
    QAA_NORMALIZATION_GLOBAL,
    compute_quality_adjusted_attention_weights,
)
from utils.qaa.quality_estimation import (
    _build_token_quality_scores,
    _compute_batch_modality_quality_scores,
)
from utils.qaa.quality_scoring_qwen import compute_batch_modality_quality_scores_with_qwen

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_QUALITY_CALIBRATION_PATHS = [
    "ecdf_manifest/quality_percentile_calibration_1m_noise_audio.json",
    "ecdf_manifest/quality_percentile_calibration_1m_noise_text.json",
    "ecdf_manifest/quality_percentile_calibration_1m_noise_image.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how first-layer attention ranking shifts when applying QAA quality scaling "
            "and normalization to baseline first-layer attention probabilities."
        )
    )
    parser.add_argument("--dataset", type=str, default="meld")
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated modalities from text,audio,video,image.",
    )
    parser.add_argument(
        "--noisy-modalities",
        type=str,
        default=None,
        help="Comma-separated modalities that should use noisy input variants.",
    )
    parser.add_argument(
        "--noise-severity",
        type=int,
        default=None,
        help="Optional noise severity level S used with --noisy-modalities.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--classification-task",
        type=str,
        default=None,
        help="For MELD: sentiment|emotion. For VoxCeleb: name|nationality.",
    )
    parser.add_argument("--audio-subdir", type=str, default="audio_only")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for thinker forward during attention-shift analysis.",
    )
    parser.add_argument(
        "--stratified-samples",
        type=int,
        default=16,
        help="Deterministically select this many samples (recommended for first run).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap after stratified selection.",
    )
    parser.add_argument(
        "--qwen-model-id",
        type=str,
        default=DEFAULT_QWEN_MODEL_ID,
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=("eager", "sdpa", "flash_attention_2"),
        help=(
            "Attention backend for this analysis run. "
            "Use eager/sdpa for attention extraction; flash_attention_2 does not expose attention tensors."
        ),
    )
    parser.add_argument(
        "--qaa-normalization-mode",
        type=str,
        choices=(QAA_NORMALIZATION_GLOBAL, QAA_NORMALIZATION_EXCLUDE_UNSCALED),
        default=QAA_NORMALIZATION_EXCLUDE_UNSCALED,
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="last",
        choices=("last", "all"),
        help="'last': only final valid query token per sample. 'all': all valid query tokens.",
    )
    parser.add_argument(
        "--query-token-scope",
        type=str,
        default="all",
        choices=("all", "modality"),
        help=(
            "Which query tokens to analyze. "
            "'all': any valid token from attention mask. "
            "'modality': only modality-specific tokens for enabled modalities."
        ),
    )
    parser.add_argument(
        "--key-token-scope",
        type=str,
        default="all",
        choices=("all", "modality"),
        help=(
            "Which key tokens are considered in ranking. "
            "'all': all valid tokens. "
            "'modality': only modality-specific tokens for enabled modalities."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k set used for overlap/jaccard metrics.",
    )
    parser.add_argument(
        "--force-quality-scores-one",
        action="store_true",
        help="Override modality quality scores to 1.0 before token mapping.",
    )
    parser.add_argument(
        "--force-modality-quality-scores",
        type=str,
        default=None,
        help="Optional modality overrides, for example: text=0.2,audio=0.9",
    )
    parser.add_argument(
        "--quality-calibration",
        action="store_true",
        help="Apply percentile calibration to raw modality quality scores.",
    )
    parser.add_argument(
        "--quality-calibration-path",
        type=str,
        action="append",
        default=None,
        help="Optional custom calibration path(s), comma-separated or repeated.",
    )
    parser.add_argument(
        "--qwen-quality",
        action="store_true",
        help="Use Qwen-based quality estimation instead of BRISQUE/PAM/perplexity path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/analysis/first_layer_qaa_attention",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Optional output filename prefix.",
    )
    return parser.parse_args()


def _flatten_calibration_paths(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return []
    flattened_paths = []
    seen = set()
    for raw_value in raw_values:
        for maybe_path in raw_value.split(","):
            path = maybe_path.strip()
            if not path or path in seen:
                continue
            seen.add(path)
            flattened_paths.append(path)
    return flattened_paths


def _parse_forced_modality_quality_scores(raw_value: str | None) -> dict[str, float]:
    if raw_value is None:
        return {}

    forced_scores = {}
    for part in raw_value.split(","):
        entry = part.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "Invalid --force-modality-quality-scores entry. "
                f"Expected modality=score, got '{entry}'."
            )
        modality_raw, score_raw = entry.split("=", 1)
        modality = modality_raw.strip().lower()
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{modality}' in --force-modality-quality-scores. "
                f"Supported: {sorted(SUPPORTED_MODALITIES)}"
            )
        if modality in forced_scores:
            raise ValueError(f"Duplicate modality '{modality}' in --force-modality-quality-scores.")
        score = float(score_raw.strip())
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Out-of-range score for modality '{modality}': {score}. Expected [0,1].")
        forced_scores[modality] = score
    return forced_scores


def _calibrate_scores_with_fallback(
    raw_scores_per_entry: list[dict[str, float]],
    quality_calibrators,
) -> list[dict[str, float]]:
    if not quality_calibrators:
        return [{modality: score for modality, score in sample_scores.items()} for sample_scores in raw_scores_per_entry]

    calibrated_scores_per_entry = []
    for sample_scores in raw_scores_per_entry:
        calibrated_sample_scores = {}
        for modality, raw_score in sample_scores.items():
            calibrator = quality_calibrators.get(modality)
            calibrated_sample_scores[modality] = (
                calibrator.calibrate(raw_score) if calibrator is not None else raw_score
            )
        calibrated_scores_per_entry.append(calibrated_sample_scores)
    return calibrated_scores_per_entry


def _cast_floats_to_dtype(batch, dtype: torch.dtype):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            batch[key] = value.to(dtype)
    return batch


def _build_user_content(enabled_modalities, sample):
    user_content = []
    if "video" in enabled_modalities:
        video_path = sample.get("video")
        if video_path is None or not Path(video_path).exists():
            return None
        user_content.append({"type": "video", "video": str(video_path)})
    if "audio" in enabled_modalities:
        audio_path = sample.get("audio")
        if audio_path is None or not Path(audio_path).exists():
            return None
        user_content.append({"type": "audio", "audio": str(audio_path)})
    if "image" in enabled_modalities:
        image_path = sample.get("image")
        if image_path is None or not Path(image_path).exists():
            return None
        user_content.append({"type": "image", "image": str(image_path)})
    if "text" in enabled_modalities:
        text_value = (sample.get("text") or "").strip()
        if not text_value:
            return None
        user_content.append({"type": "text", "text": text_value})
    return user_content


def _build_conversation_for_sample(sample, dataset, prompt, enabled_modalities):
    sample_prompt = prompt
    if dataset == "nejm":
        option_labels = sample.get("option_labels") or []
        options = (sample.get("options") or "").strip()
        if option_labels:
            option_block = "\n".join(f"- {label}" for label in option_labels)
        else:
            option_block = options
        if option_block:
            sample_prompt = (
                f"{prompt}\n"
                "Allowed labels for this specific case:\n"
                f"{option_block}\n"
                "Reply with exactly one label from the list above. Do not add explanation."
            )

    user_content = _build_user_content(enabled_modalities=enabled_modalities, sample=sample)
    if user_content is None:
        return None
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sample_prompt}],
        },
        {"role": "user", "content": user_content},
    ]


def _build_processor_inputs(processor, entries, enabled_modalities, device, dtype):
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
    inputs = _cast_floats_to_dtype(inputs, dtype)
    return inputs, images, videos


def _build_text_token_ids_per_entry(processor, entries, enabled_modalities):
    if "text" not in enabled_modalities:
        return None
    text_token_ids_per_entry = []
    for entry in entries:
        dataset_text = (entry["sample"].get("text") or "").strip()
        if not dataset_text:
            text_token_ids_per_entry.append([])
            continue
        encoded = processor.tokenizer(
            dataset_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        text_token_ids_per_entry.append([int(token_id) for token_id in encoded])
    return text_token_ids_per_entry


def _token_string(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        token = "<unk>"
    return str(token).replace("\n", "\\n")


def _make_query_indices(attention_mask_row: torch.Tensor | None, seq_len: int, mode: str) -> list[int]:
    if attention_mask_row is None:
        valid_positions = torch.arange(seq_len, dtype=torch.long)
    else:
        valid_positions = torch.nonzero(attention_mask_row, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return []
    if mode == "last":
        return [int(valid_positions[-1].item())]
    return [int(idx) for idx in valid_positions.tolist()]


def _make_query_indices_with_scope(
    attention_mask_row: torch.Tensor | None,
    seq_len: int,
    mode: str,
    query_scope_mask: torch.Tensor | None = None,
) -> list[int]:
    if attention_mask_row is None:
        valid_positions = torch.arange(seq_len, dtype=torch.long)
    else:
        valid_positions = torch.nonzero(attention_mask_row, as_tuple=False).flatten()

    if query_scope_mask is not None:
        valid_positions = valid_positions[query_scope_mask[valid_positions]]

    if valid_positions.numel() == 0:
        return []
    if mode == "last":
        return [int(valid_positions[-1].item())]
    return [int(idx) for idx in valid_positions.tolist()]


def _format_topk_triplets(indices: list[int], weights: torch.Tensor, input_ids: torch.Tensor, tokenizer) -> str:
    parts = []
    for idx in indices:
        token_id = int(input_ids[idx].item())
        token = _token_string(tokenizer, token_id)
        parts.append(f"{idx}:{token_id}:{token}:{float(weights[idx].item()):.6f}")
    return "|".join(parts)


def _write_rows_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_special_token_id(tokenizer, token_text: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is None:
        return None
    token_id = int(token_id)
    if token_id < 0:
        return None
    recovered = tokenizer.convert_ids_to_tokens(token_id)
    if recovered != token_text:
        return None
    return token_id


def _build_modality_token_ids(
    *,
    enabled_modalities: set[str],
    thinker_config,
    tokenizer,
) -> set[int]:
    token_ids: set[int] = set()

    if "audio" in enabled_modalities and hasattr(thinker_config, "audio_token_index"):
        token_ids.add(int(thinker_config.audio_token_index))
        for token_text in ("<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>"):
            token_id = _resolve_special_token_id(tokenizer, token_text)
            if token_id is not None:
                token_ids.add(token_id)

    if "video" in enabled_modalities and hasattr(thinker_config, "video_token_index"):
        token_ids.add(int(thinker_config.video_token_index))
        for token_text in ("<|VIDEO|>", "<|vision_bos|>", "<|vision_eos|>"):
            token_id = _resolve_special_token_id(tokenizer, token_text)
            if token_id is not None:
                token_ids.add(token_id)

    if "image" in enabled_modalities and hasattr(thinker_config, "image_token_index"):
        token_ids.add(int(thinker_config.image_token_index))
        for token_text in ("<|IMAGE|>", "<|vision_bos|>", "<|vision_eos|>"):
            token_id = _resolve_special_token_id(tokenizer, token_text)
            if token_id is not None:
                token_ids.add(token_id)

    return token_ids


def _format_exception_summary(exc: Exception, max_chars: int = 220) -> str:
    message = " ".join(str(exc).split()).strip()
    if not message:
        return exc.__class__.__name__
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3] + "..."


def _log_skipped_single_sample(entries, stage: str, exc: Exception):
    sample = entries[0]["sample"]
    print(
        "[WARN] Skipping sample after "
        f"{stage}: sample_id={sample.get('sample_id','')} file={sample.get('file','')} "
        f"error={exc.__class__.__name__}: {_format_exception_summary(exc)}",
        flush=True,
    )


def _skip_or_raise_batch_error(entries, stage: str, exc: Exception, batch_size: int):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(entries) == 1:
        _log_skipped_single_sample(entries, stage=stage, exc=exc)
        return
    raise RuntimeError(
        f"Batch failed during {stage}: {exc}. "
        f"Reduce --batch-size (current {batch_size}, recommended: 1 for audio/video analysis)."
    ) from exc


def _build_forced_scores_for_entries(
    *,
    entries,
    enabled_modalities: set[str],
    force_quality_scores_one: bool,
    forced_modality_scores: dict[str, float],
) -> tuple[list[dict[str, float]] | None, list[dict[str, float]] | None]:
    if force_quality_scores_one:
        raw = [{modality: 1.0 for modality in enabled_modalities} for _ in entries]
        effective = [{modality: 1.0 for modality in enabled_modalities} for _ in entries]
        return raw, effective

    if forced_modality_scores and all(modality in forced_modality_scores for modality in enabled_modalities):
        raw = [{modality: forced_modality_scores[modality] for modality in enabled_modalities} for _ in entries]
        effective = [{modality: forced_modality_scores[modality] for modality in enabled_modalities} for _ in entries]
        return raw, effective

    return None, None


def _build_causal_mask_for_eager(hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full((q_len, q_len), min_value, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, q_len, q_len)


def _install_first_layer_attention_capture(model):
    first_attn = model.thinker.model.layers[0].self_attn
    if getattr(first_attn, "_first_layer_capture_patch_installed", False):
        return first_attn

    first_attn._captured_attn_weights = None
    first_attn._original_forward_for_capture = first_attn.forward

    def _capture_forward(
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
        # When the thinker uses SDPA with output_attentions=False, causal masking can be implicit.
        # Eager attention needs an explicit additive mask, so reconstruct it if absent.
        effective_mask = attention_mask
        if effective_mask is None:
            effective_mask = _build_causal_mask_for_eager(hidden_states)

        attn_output, attn_weights, present_key_value = qwen_omni_modeling.Qwen2_5OmniAttention.forward(
            self,
            hidden_states=hidden_states,
            attention_mask=effective_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        self._captured_attn_weights = attn_weights.detach().to(dtype=torch.float32, device="cpu")
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, present_key_value

    first_attn.forward = types.MethodType(_capture_forward, first_attn)
    first_attn._first_layer_capture_patch_installed = True
    return first_attn


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.stratified_samples is not None and args.stratified_samples < 1:
        raise ValueError("--stratified-samples must be >= 1 when provided.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1 when provided.")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")
    if args.force_quality_scores_one and args.force_modality_quality_scores:
        raise ValueError("--force-quality-scores-one cannot be combined with --force-modality-quality-scores.")

    dataset = normalize_dataset_name(args.dataset)
    enabled_modalities = normalize_modalities(args.modalities)
    if enabled_modalities is None:
        enabled_modalities = default_modalities_for_dataset(dataset)
    noisy_modalities = normalize_modalities(args.noisy_modalities)
    validate_modalities(dataset, enabled_modalities, noisy_modalities)

    if dataset == "meld":
        meld_task, label_column = normalize_meld_task(args.classification_task)
    elif dataset == "voxceleb":
        meld_task = None
        label_column = normalize_voxceleb_label_column(args.classification_task)
    else:
        meld_task = None
        label_column = "label"

    prompt = get_prompt_for_classification(dataset, meld_task)
    samples = load_samples(dataset, args, enabled_modalities, noisy_modalities, label_column)
    if args.stratified_samples is not None and len(samples) > args.stratified_samples:
        samples = select_stratified_samples(samples, args.stratified_samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No samples found for the selected dataset/modality configuration.")

    forced_modality_scores = _parse_forced_modality_quality_scores(args.force_modality_quality_scores)
    calibration_paths = _flatten_calibration_paths(args.quality_calibration_path)
    if args.quality_calibration:
        if not calibration_paths:
            calibration_paths = list(DEFAULT_QUALITY_CALIBRATION_PATHS)
        quality_calibrators = load_percentile_calibration(calibration_paths)
        print(f"[INFO] Loaded quality calibrators for modalities: {sorted(quality_calibrators.keys())}", flush=True)
    else:
        quality_calibrators = None

    if args.attn_implementation == "flash_attention_2":
        print(
            "[WARN] flash_attention_2 does not expose explicit attention tensors for this script's capture path. "
            "Use eager/sdpa instead.",
            flush=True,
        )

    print(f"[INFO] Loading model {args.qwen_model_id} ...", flush=True)
    use_cuda = torch.cuda.is_available()
    model_dtype = torch.bfloat16 if use_cuda else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.qwen_model_id,
        torch_dtype=model_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        enable_audio_output=False,
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen_model_id)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"[INFO] Model ready on {device} with dtype={dtype}.", flush=True)
    first_attn = _install_first_layer_attention_capture(model)
    print("[INFO] First-layer attention capture patch installed.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"{dataset}_{args.qaa_normalization_mode}_{args.query_mode}"
    detail_csv_path = out_dir / f"{prefix}_detail.csv"
    summary_json_path = out_dir / f"{prefix}_summary.json"

    detail_rows: list[dict] = []
    skipped_samples = 0
    batch_count = 0

    total_queries = 0
    top1_changed_count = 0
    top1_from_unscaled_to_scaled = 0
    top1_from_scaled_to_unscaled = 0
    topk_jaccard_sum = 0.0
    total_variation_sum = 0.0
    queries_skipped_no_query_tokens = 0
    queries_skipped_no_key_tokens = 0

    modality_token_ids = _build_modality_token_ids(
        enabled_modalities=enabled_modalities,
        thinker_config=model.thinker.config,
        tokenizer=processor.tokenizer,
    )
    if args.query_token_scope == "modality" or args.key_token_scope == "modality":
        if not modality_token_ids:
            raise RuntimeError(
                "Requested modality-only query/key scope but no modality token ids were resolved "
                "for the current enabled modalities."
            )
        print(
            f"[INFO] Modality-token scope enabled with {len(modality_token_ids)} token ids: "
            f"{sorted(modality_token_ids)}",
            flush=True,
        )

    for batch_start in range(0, len(samples), args.batch_size):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        entries = []
        for sample in batch_samples:
            conversation = _build_conversation_for_sample(
                sample=sample,
                dataset=dataset,
                prompt=prompt,
                enabled_modalities=enabled_modalities,
            )
            if conversation is None:
                skipped_samples += 1
                continue
            entries.append({"sample": sample, "conversation": conversation})
        if not entries:
            continue

        batch_count += 1
        print(
            f"[INFO] Processing batch {batch_count} with {len(entries)} usable samples "
            f"(global offset {batch_start}).",
            flush=True,
        )

        try:
            inputs, images, videos = _build_processor_inputs(
                processor=processor,
                entries=entries,
                enabled_modalities=enabled_modalities,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            _skip_or_raise_batch_error(
                entries,
                stage="input preparation",
                exc=exc,
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue

        try:
            forced_raw_scores, forced_effective_scores = _build_forced_scores_for_entries(
                entries=entries,
                enabled_modalities=enabled_modalities,
                force_quality_scores_one=args.force_quality_scores_one,
                forced_modality_scores=forced_modality_scores,
            )
            if forced_raw_scores is not None and forced_effective_scores is not None:
                raw_modality_quality_scores = forced_raw_scores
                modality_quality_scores = forced_effective_scores
            else:
                if args.qwen_quality:
                    raw_modality_quality_scores = compute_batch_modality_quality_scores_with_qwen(
                        entries=entries,
                        enabled_modalities=enabled_modalities,
                        model=model,
                        processor=processor,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    raw_modality_quality_scores = _compute_batch_modality_quality_scores(
                        entries=entries,
                        enabled_modalities=enabled_modalities,
                        model=model,
                        processor=processor,
                        device=device,
                        qwen_images=images,
                        qwen_videos=videos,
                    )

                if args.force_quality_scores_one:
                    modality_quality_scores = [
                        {modality: 1.0 for modality in sample_scores}
                        for sample_scores in raw_modality_quality_scores
                    ]
                else:
                    modality_quality_scores = _calibrate_scores_with_fallback(
                        raw_modality_quality_scores,
                        quality_calibrators,
                    )
                if forced_modality_scores:
                    modality_quality_scores = [
                        {
                            modality: forced_modality_scores.get(modality, score)
                            for modality, score in sample_scores.items()
                        }
                        for sample_scores in modality_quality_scores
                    ]

            text_token_ids_per_entry = _build_text_token_ids_per_entry(
                processor=processor,
                entries=entries,
                enabled_modalities=enabled_modalities,
            )
            token_quality_scores, scaled_token_mask = _build_token_quality_scores(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                modality_scores_per_entry=modality_quality_scores,
                thinker_config=model.thinker.config,
                text_token_ids_per_entry=text_token_ids_per_entry,
                return_scaled_token_mask=True,
            )
        except Exception as exc:
            _skip_or_raise_batch_error(
                entries,
                stage="quality scoring",
                exc=exc,
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue

        try:
            with torch.inference_mode():
                first_attn._captured_attn_weights = None
                _ = model.thinker(
                    **inputs,
                    use_audio_in_video=False,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
        except torch.OutOfMemoryError as exc:
            _skip_or_raise_batch_error(
                entries,
                stage="CUDA OOM",
                exc=exc,
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue
        except Exception as exc:
            _skip_or_raise_batch_error(
                entries,
                stage="model forward",
                exc=exc,
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue
        first_layer_attn = first_attn._captured_attn_weights
        if first_layer_attn is None:
            _skip_or_raise_batch_error(
                entries,
                stage="attention capture",
                exc=RuntimeError(
                    "First-layer attention capture failed (no captured attention tensor). "
                    "Use --attn-implementation sdpa or eager."
                ),
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue
        try:
            adjusted_attn = compute_quality_adjusted_attention_weights(
                first_layer_attn,
                token_quality_scores.detach().to(dtype=torch.float32, device="cpu"),
                quality_scaled_token_mask=scaled_token_mask.detach().to("cpu")
                if args.qaa_normalization_mode == QAA_NORMALIZATION_EXCLUDE_UNSCALED
                else None,
                quality_normalization_mode=args.qaa_normalization_mode,
            )
        except Exception as exc:
            _skip_or_raise_batch_error(
                entries,
                stage="attention adjustment",
                exc=exc,
                batch_size=args.batch_size,
            )
            skipped_samples += 1
            continue

        attention_mask = inputs.get("attention_mask")
        input_ids = inputs["input_ids"]

        first_layer_attn_cpu = first_layer_attn
        adjusted_attn_cpu = adjusted_attn
        token_quality_cpu = token_quality_scores.detach().cpu()
        scaled_mask_cpu = scaled_token_mask.detach().cpu()
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

        for row_idx, entry in enumerate(entries):
            sample = entry["sample"]
            seq_len = input_ids_cpu.size(1)
            row_input_ids = input_ids_cpu[row_idx]
            modality_mask_row = None
            if args.query_token_scope == "modality" or args.key_token_scope == "modality":
                modality_mask_row = torch.zeros(seq_len, dtype=torch.bool)
                for token_id in modality_token_ids:
                    modality_mask_row |= row_input_ids == token_id

            query_scope_mask = modality_mask_row if args.query_token_scope == "modality" else None
            query_indices = _make_query_indices_with_scope(
                attention_mask_row=attention_mask_cpu[row_idx] if attention_mask_cpu is not None else None,
                seq_len=seq_len,
                mode=args.query_mode,
                query_scope_mask=query_scope_mask,
            )
            if not query_indices:
                queries_skipped_no_query_tokens += 1
                continue

            if attention_mask_cpu is None:
                valid_key_positions = torch.arange(seq_len, dtype=torch.long)
            else:
                valid_key_positions = torch.nonzero(attention_mask_cpu[row_idx], as_tuple=False).flatten()
            if args.key_token_scope == "modality":
                if modality_mask_row is None:
                    raise RuntimeError("Internal error: modality key scope requested without modality mask.")
                valid_key_positions = valid_key_positions[modality_mask_row[valid_key_positions]]
            if valid_key_positions.numel() == 0:
                queries_skipped_no_key_tokens += len(query_indices)
                continue
            valid_key_mask = torch.zeros(seq_len, dtype=torch.bool)
            valid_key_mask[valid_key_positions] = True

            for query_idx in query_indices:
                baseline_mean = first_layer_attn_cpu[row_idx, :, query_idx, :].mean(dim=0)
                adjusted_mean = adjusted_attn_cpu[row_idx, :, query_idx, :].mean(dim=0)

                baseline_ranking_weights = baseline_mean.clone()
                adjusted_ranking_weights = adjusted_mean.clone()
                baseline_ranking_weights[~valid_key_mask] = float("-inf")
                adjusted_ranking_weights[~valid_key_mask] = float("-inf")

                k = min(args.top_k, int(valid_key_positions.numel()))
                topk_before = torch.topk(baseline_ranking_weights, k).indices.tolist()
                topk_after = torch.topk(adjusted_ranking_weights, k).indices.tolist()
                top1_before = int(topk_before[0])
                top1_after = int(topk_after[0])

                set_before = set(topk_before)
                set_after = set(topk_after)
                overlap = len(set_before & set_after)
                union = len(set_before | set_after)
                jaccard = float(overlap / union) if union > 0 else 1.0

                baseline_valid = baseline_mean[valid_key_positions]
                adjusted_valid = adjusted_mean[valid_key_positions]
                tvd = float(0.5 * torch.abs(adjusted_valid - baseline_valid).sum().item())

                top1_before_scaled = bool(scaled_mask_cpu[row_idx, top1_before].item())
                top1_after_scaled = bool(scaled_mask_cpu[row_idx, top1_after].item())
                top1_changed = top1_before != top1_after

                total_queries += 1
                topk_jaccard_sum += jaccard
                total_variation_sum += tvd
                if top1_changed:
                    top1_changed_count += 1
                    if (not top1_before_scaled) and top1_after_scaled:
                        top1_from_unscaled_to_scaled += 1
                    elif top1_before_scaled and (not top1_after_scaled):
                        top1_from_scaled_to_unscaled += 1

                query_token_id = int(input_ids_cpu[row_idx, query_idx].item())
                row = {
                    "dataset": dataset,
                    "split": sample.get("split", ""),
                    "sample_id": sample.get("sample_id", ""),
                    "file": sample.get("file", ""),
                    "query_idx": query_idx,
                    "query_token_id": query_token_id,
                    "query_token": _token_string(processor.tokenizer, query_token_id),
                    "top1_before_idx": top1_before,
                    "top1_before_token_id": int(input_ids_cpu[row_idx, top1_before].item()),
                    "top1_before_token": _token_string(
                        processor.tokenizer, int(input_ids_cpu[row_idx, top1_before].item())
                    ),
                    "top1_before_weight": float(baseline_mean[top1_before].item()),
                    "top1_before_quality": float(token_quality_cpu[row_idx, top1_before].item()),
                    "top1_before_scaled": int(top1_before_scaled),
                    "top1_after_idx": top1_after,
                    "top1_after_token_id": int(input_ids_cpu[row_idx, top1_after].item()),
                    "top1_after_token": _token_string(
                        processor.tokenizer, int(input_ids_cpu[row_idx, top1_after].item())
                    ),
                    "top1_after_weight": float(adjusted_mean[top1_after].item()),
                    "top1_after_quality": float(token_quality_cpu[row_idx, top1_after].item()),
                    "top1_after_scaled": int(top1_after_scaled),
                    "top1_changed": int(top1_changed),
                    "topk_overlap": overlap,
                    "topk_jaccard": jaccard,
                    "total_variation_distance": tvd,
                    "topk_before": _format_topk_triplets(
                        topk_before,
                        baseline_mean,
                        input_ids_cpu[row_idx],
                        processor.tokenizer,
                    ),
                    "topk_after": _format_topk_triplets(
                        topk_after,
                        adjusted_mean,
                        input_ids_cpu[row_idx],
                        processor.tokenizer,
                    ),
                }
                detail_rows.append(row)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_rows_csv(detail_csv_path, detail_rows)

    if total_queries == 0:
        raise RuntimeError(
            "No queries analyzed. This usually means all selected samples were skipped due to missing modalities/files."
        )

    summary = {
        "dataset": dataset,
        "enabled_modalities": sorted(enabled_modalities),
        "noisy_modalities": sorted(noisy_modalities) if noisy_modalities is not None else None,
        "qaa_normalization_mode": args.qaa_normalization_mode,
        "query_mode": args.query_mode,
        "query_token_scope": args.query_token_scope,
        "key_token_scope": args.key_token_scope,
        "top_k": args.top_k,
        "samples_selected": len(samples),
        "samples_skipped": skipped_samples,
        "queries_analyzed": total_queries,
        "queries_skipped_no_query_tokens": queries_skipped_no_query_tokens,
        "queries_skipped_no_key_tokens": queries_skipped_no_key_tokens,
        "top1_changed_count": top1_changed_count,
        "top1_changed_rate": top1_changed_count / total_queries,
        "top1_from_unscaled_to_scaled": top1_from_unscaled_to_scaled,
        "top1_from_scaled_to_unscaled": top1_from_scaled_to_unscaled,
        "avg_topk_jaccard": topk_jaccard_sum / total_queries,
        "avg_total_variation_distance": total_variation_sum / total_queries,
        "detail_csv_path": str(detail_csv_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[INFO] Analysis complete.", flush=True)
    print(f"[INFO] Detail CSV: {detail_csv_path}", flush=True)
    print(f"[INFO] Summary JSON: {summary_json_path}", flush=True)
    print(
        "[INFO] Top-1 changed "
        f"{top1_changed_count}/{total_queries} queries "
        f"({summary['top1_changed_rate']:.4f}).",
        flush=True,
    )
    print(
        "[INFO] Avg top-k Jaccard="
        f"{summary['avg_topk_jaccard']:.4f}, "
        f"Avg TVD={summary['avg_total_variation_distance']:.6f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
