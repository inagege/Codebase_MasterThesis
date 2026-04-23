import argparse
import csv
import hashlib
import os
import random
import re
import traceback
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

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
from utils.calibration.quality_calibration import (
    SUPPORTED_MODALITIES,
    load_percentile_calibration,
)
from utils.qaa.quality_aware_attention import (
    install_quality_aware_first_attention_patch,
    set_first_layer_quality_scores,
)
from utils.qaa.quality_estimation import (
    _build_token_quality_scores,
    _compute_batch_modality_quality_scores,
)
from utils.qaa.quality_scoring_qwen import compute_batch_modality_quality_scores_with_qwen

MODALITY_FILENAME_TOKENS = {
    "audio": "a",
    "image": "i",
    "text": "t",
    "video": "v",
}

DEFAULT_QUALITY_CALIBRATION_PATHS = [
    "ecdf_manifest/quality_percentile_calibration_1m_noise_audio.json",
    "ecdf_manifest/quality_percentile_calibration_1m_noise_text.json",
    "ecdf_manifest/quality_percentile_calibration_1m_noise_image.json",
]
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


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
        "--noise-severity",
        type=int,
        default=None,
        help="Optional noise severity level S. When set with --noisy-modalities, only variants with folder token S=<level> are loaded.",
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
    parser.add_argument("--audio-subdir", type=str, default="audio_only", help="Subdir for WAV files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per generation batch. Reduce if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--force-quality-scores-one",
        action="store_true",
        help=(
            "Run the normal scored-modalities path, but override all computed modality quality scores to 1.0 "
            "before token-level quality weighting."
        ),
    )
    parser.add_argument(
        "--quality-placebo-random",
        action="store_true",
        help=(
            "Enable placebo/sham baseline for quality-aware attention by replacing modality quality scores "
            "with deterministic per-sample/per-modality pseudo-random values in (0,1)."
        ),
    )
    parser.add_argument(
        "--quality-placebo-random-seed",
        type=int,
        default=0,
        help="Seed used for deterministic pseudo-random scores when --quality-placebo-random is enabled.",
    )
    parser.add_argument(
        "--quality-calibration",
        action="store_true",
        help=(
            "Enable frozen percentile calibration of modality quality scores. "
            "When set, default calibration manifests are loaded automatically."
        ),
    )
    parser.add_argument(
        "--qwen-quality",
        action="store_true",
        help=(
            "Use Qwen2.5-Omni to estimate modality quality scores in [0,1]. "
            "Cannot be combined with --quality-calibration."
        ),
    )
    parser.add_argument(
        "--qwen-model-id",
        type=str,
        default=DEFAULT_QWEN_MODEL_ID,
        help=(
            "Hugging Face model ID for Qwen2.5-Omni (used for generation and, with --qwen-quality, "
            "for modality quality scoring)."
        ),
    )
    parser.add_argument(
        "--quality-calibration-path",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional override path(s) to percentile calibration JSON files. "
            "Can be provided multiple times or as comma-separated list. "
            "When omitted and --quality-calibration is set, defaults are loaded from ecdf_manifest/."
        ),
    )
    parser.add_argument(
        "--quality-score-out-path",
        type=str,
        default=None,
        help=(
            "Optional override CSV path to store per-sample raw/calibrated modality quality scores. "
            "If omitted, it is generated automatically from dataset/modality/noise/task."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Optional override output CSV path for predictions. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--out-error-path",
        type=str,
        default=None,
        help="Optional override output CSV path for error rows. Auto-generated when omitted.",
    )
    return parser.parse_args()

def append_csv_row(path: str, fieldnames: list[str], row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _quality_score_fieldnames() -> list[str]:
    fieldnames = ["dataset", "split", "sample_id", "file"]
    for modality in SUPPORTED_MODALITIES:
        fieldnames.append(f"{modality}_raw_quality")
        fieldnames.append(f"{modality}_calibrated_quality")
    return fieldnames


def append_quality_score_row(
    path: str,
    dataset: str,
    sample,
    raw_scores: dict[str, float],
    calibrated_scores: dict[str, float],
):
    row = {
        "dataset": dataset,
        "split": sample["split"],
        "sample_id": sample["sample_id"],
        "file": sample["file"],
    }
    for modality in SUPPORTED_MODALITIES:
        row[f"{modality}_raw_quality"] = raw_scores.get(modality, "")
        row[f"{modality}_calibrated_quality"] = calibrated_scores.get(modality, "")
    append_csv_row(path, _quality_score_fieldnames(), row)


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


def _modalities_to_path_token(modalities: set[str] | None) -> str:
    if not modalities:
        return ""
    return "".join(MODALITY_FILENAME_TOKENS[modality] for modality in sorted(modalities))


def _model_id_to_path_token(model_id: str) -> str:
    normalized_model_id = model_id.strip().lower()

    # Preferred concise token for Qwen2.5-Omni checkpoints, e.g. Qwen_7B / Qwen_3B.
    size_match = re.search(r"qwen2(?:\.5)?-omni-(\d+(?:\.\d+)?)b", normalized_model_id)
    if size_match:
        size_token = size_match.group(1).replace(".", "p")
        return f"Qwen_{size_token}B"

    # Fallback for other model IDs: keep path-safe token and avoid dots.
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_id.strip()) or "unknown_model"


def _placebo_path_suffix(
    quality_placebo_random: bool,
    quality_placebo_random_seed: int,
) -> str:
    if not quality_placebo_random:
        return ""
    return f"_placebo_rand_seed_{quality_placebo_random_seed}"


def _build_auto_output_paths(
    dataset: str,
    enabled_modalities: set[str],
    noisy_modalities: set[str] | None,
    quality_calibration_enabled: bool,
    qwen_quality_enabled: bool,
    quality_placebo_random: bool,
    quality_placebo_random_seed: int,
    meld_task: str | None,
    qwen_model_id: str,
) -> tuple[str, str, str]:
    modality_token = _modalities_to_path_token(enabled_modalities)
    noisy_token = _modalities_to_path_token(noisy_modalities)
    model_token = _model_id_to_path_token(qwen_model_id)
    placebo_suffix = _placebo_path_suffix(
        quality_placebo_random=quality_placebo_random,
        quality_placebo_random_seed=quality_placebo_random_seed,
    )

    if quality_placebo_random:
        base_dir = Path("out") / "placebo" / "random" / model_token / dataset
    elif qwen_quality_enabled:
        base_dir = Path("out") / "qwen_scored" / model_token / dataset
    elif quality_calibration_enabled:
        base_dir = Path("out") / "calibration" / model_token / dataset
    else:
        base_dir = Path("out") / "scored" / model_token / dataset
    if dataset == "meld" and meld_task:
        base_dir = base_dir / meld_task

    quality_score_path = base_dir / f"quality_scores_{modality_token}_noise_{noisy_token}{placebo_suffix}.csv"
    prediction_path = base_dir / f"prediction_{modality_token}_noise_{noisy_token}{placebo_suffix}.csv"
    error_path = base_dir / f"error_{modality_token}_noise_{noisy_token}{placebo_suffix}.csv"
    return str(quality_score_path), str(prediction_path), str(error_path)


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


def _deterministic_unit_interval_score(
    *,
    random_seed: int,
    split: str,
    sample_id,
    file_name: str,
    modality: str,
) -> float:
    seed_payload = f"{random_seed}|{split}|{sample_id}|{file_name}|{modality}".encode("utf-8")
    digest = hashlib.sha256(seed_payload).digest()
    derived_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(derived_seed)
    sampled_int = rng.randint(1, 999_999)
    return sampled_int / 1_000_000.0


def _build_placebo_modality_scores(
    *,
    entries,
    modality_scores_per_entry: list[dict[str, float]],
    quality_placebo_random: bool,
    quality_placebo_random_seed: int,
) -> list[dict[str, float]]:
    if not quality_placebo_random:
        return [{modality: score for modality, score in sample_scores.items()} for sample_scores in modality_scores_per_entry]

    if len(entries) != len(modality_scores_per_entry):
        raise ValueError(
            "Placebo score batch mismatch: "
            f"entries={len(entries)}, scores={len(modality_scores_per_entry)}"
        )
    placebo_scores = []
    for entry, sample_scores in zip(entries, modality_scores_per_entry):
        sample = entry["sample"]
        placebo_scores.append(
            {
                modality: _deterministic_unit_interval_score(
                    random_seed=quality_placebo_random_seed,
                    split=str(sample.get("split", "")),
                    sample_id=sample.get("sample_id", ""),
                    file_name=str(sample.get("file", "")),
                    modality=modality,
                )
                for modality in sample_scores
            }
        )
    return placebo_scores


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


def run_batch_generation(
    model,
    processor,
    entries,
    enabled_modalities,
    device,
    dtype,
    force_quality_scores_one=False,
    quality_calibrators=None,
    qwen_quality=False,
    quality_placebo_random=False,
    quality_placebo_random_seed=0,
):
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

    if quality_placebo_random:
        placeholder_scores = [
            {modality: 1.0 for modality in enabled_modalities}
            for _ in entries
        ]
        placebo_scores = _build_placebo_modality_scores(
            entries=entries,
            modality_scores_per_entry=placeholder_scores,
            quality_placebo_random=True,
            quality_placebo_random_seed=quality_placebo_random_seed,
        )
        # In placebo mode, the placebo itself is both the recorded "raw" and effective score.
        raw_modality_quality_scores = [
            {modality: score for modality, score in sample_scores.items()}
            for sample_scores in placebo_scores
        ]
        modality_quality_scores = [
            {modality: score for modality, score in sample_scores.items()}
            for sample_scores in placebo_scores
        ]
    else:
        if qwen_quality:
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
        if force_quality_scores_one:
            modality_quality_scores = [
                {modality: 1.0 for modality in sample_scores}
                for sample_scores in raw_modality_quality_scores
            ]
        else:
            modality_quality_scores = _calibrate_scores_with_fallback(
                raw_modality_quality_scores,
                quality_calibrators,
            )

    text_token_ids_per_entry = None
    if "text" in enabled_modalities:
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

    token_quality_scores = _build_token_quality_scores(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        modality_scores_per_entry=modality_quality_scores,
        thinker_config=model.thinker.config,
        text_token_ids_per_entry=text_token_ids_per_entry,
    )

    set_first_layer_quality_scores(model, token_quality_scores)
    try:
        gen_output = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            output_scores=True,
            do_sample=False,
        )
    finally:
        set_first_layer_quality_scores(model, None)

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

    return (
        [extract_assistant_reply(out_text) for out_text in decoded],
        raw_modality_quality_scores,
        modality_quality_scores,
    )


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.start_at_sample is not None and args.start_at_sample < 0:
        raise ValueError("--start-at-sample must be >= 0 when provided.")
    if args.stratified_samples is not None and args.stratified_samples < 1:
        raise ValueError("--stratified-samples must be >= 1 when provided.")
    if args.noise_severity is not None and args.noise_severity < 0:
        raise ValueError("--noise-severity must be >= 0 when provided.")
    if args.noise_severity is not None and args.noisy_modalities is None:
        print(
            "[WARN] --noise-severity was provided without --noisy-modalities; it has no effect.",
            flush=True,
        )
    if args.force_quality_scores_one and args.quality_placebo_random:
        raise ValueError("--force-quality-scores-one cannot be combined with --quality-placebo-random.")
    if args.qwen_quality and args.quality_calibration:
        raise ValueError("--qwen-quality cannot be combined with --quality-calibration.")
    if args.quality_placebo_random and args.qwen_quality:
        print(
            "[WARN] --qwen-quality is enabled, but placebo mode bypasses Qwen quality estimation.",
            flush=True,
        )
    if args.quality_placebo_random and args.quality_calibration:
        print(
            "[WARN] --quality-calibration is enabled, but placebo mode bypasses calibration and uses placebo scores.",
            flush=True,
        )
    if not args.qwen_model_id or not args.qwen_model_id.strip():
        raise ValueError("--qwen-model-id must be a non-empty string.")
    args.qwen_model_id = args.qwen_model_id.strip()

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

    calibration_paths = []
    if args.quality_calibration:
        calibration_paths = _flatten_calibration_paths(args.quality_calibration_path)
        if not calibration_paths:
            calibration_paths = list(DEFAULT_QUALITY_CALIBRATION_PATHS)
    elif args.quality_calibration_path:
        print(
            "[WARN] Ignoring --quality-calibration-path because --quality-calibration is not enabled.",
            flush=True,
        )

    missing_calibration_files = [
        calibration_path for calibration_path in calibration_paths if not Path(calibration_path).exists()
    ]
    if missing_calibration_files:
        raise FileNotFoundError(
            "Calibration manifest file(s) not found: "
            f"{missing_calibration_files}"
        )

    (
        auto_quality_score_out_path,
        auto_out_path,
        auto_out_error_path,
    ) = _build_auto_output_paths(
        dataset=dataset,
        enabled_modalities=enabled_modalities,
        noisy_modalities=noisy_modalities,
        quality_calibration_enabled=args.quality_calibration,
        qwen_quality_enabled=args.qwen_quality,
        quality_placebo_random=args.quality_placebo_random,
        quality_placebo_random_seed=args.quality_placebo_random_seed,
        meld_task=meld_task,
        qwen_model_id=args.qwen_model_id,
    )
    if args.quality_score_out_path is None:
        args.quality_score_out_path = auto_quality_score_out_path
    if args.out_path is None:
        args.out_path = auto_out_path
    if args.out_error_path is None:
        args.out_error_path = auto_out_error_path

    print(f"[INFO] CWD: {os.getcwd()}", flush=True)
    print(f"[INFO] Dataset: {dataset}", flush=True)
    print(f"[INFO] Modalities enabled: {sorted(enabled_modalities)}", flush=True)
    print(f"[INFO] Noisy modalities: {args.noisy_modalities}", flush=True)
    print(f"[INFO] noise_severity={args.noise_severity}", flush=True)
    print(f"[INFO] Split: {args.split}", flush=True)
    print(f"[INFO] audio_subdir={args.audio_subdir}", flush=True)
    print(f"[INFO] batch_size={args.batch_size}", flush=True)
    print(f"[INFO] start_at_sample={args.start_at_sample}", flush=True)
    print(f"[INFO] stratified_samples={args.stratified_samples}", flush=True)
    print(f"[INFO] force_quality_scores_one={args.force_quality_scores_one}", flush=True)
    print(f"[INFO] quality_placebo_random={args.quality_placebo_random}", flush=True)
    print(f"[INFO] quality_placebo_random_seed={args.quality_placebo_random_seed}", flush=True)
    print(f"[INFO] quality_calibration_enabled={args.quality_calibration}", flush=True)
    print(f"[INFO] qwen_quality={args.qwen_quality}", flush=True)
    print(f"[INFO] qwen_model_id={args.qwen_model_id}", flush=True)
    print(f"[INFO] quality_calibration_paths={calibration_paths}", flush=True)
    print(f"[INFO] quality_score_out_path={args.quality_score_out_path}", flush=True)
    if label_column is not None:
        print(f"[INFO] Label column: {label_column}", flush=True)
    print(f"[INFO] out_path={args.out_path}", flush=True)
    print(f"[INFO] out_error_path={args.out_error_path}", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_error_path) or ".", exist_ok=True)
    if args.quality_score_out_path:
        os.makedirs(os.path.dirname(args.quality_score_out_path) or ".", exist_ok=True)

    quality_calibrators = None
    if calibration_paths:
        quality_calibrators = {}
        for calibration_path in calibration_paths:
            path_calibrators = load_percentile_calibration(calibration_path)
            duplicate_modalities = sorted(set(quality_calibrators) & set(path_calibrators))
            if duplicate_modalities:
                raise ValueError(
                    "Duplicate modality calibrators found across quality calibration files "
                    f"for modalities: {duplicate_modalities}. "
                    "Please keep one calibration source per modality."
                )
            quality_calibrators.update(path_calibrators)
            loaded_modalities = sorted(path_calibrators)
            print(
                "[INFO] Loaded percentile quality calibration file "
                f"{calibration_path} for modalities: {loaded_modalities}",
                flush=True,
            )

        if "video" in enabled_modalities and "video" not in quality_calibrators and "image" in quality_calibrators:
            quality_calibrators["video"] = quality_calibrators["image"]
            print(
                "[INFO] Reusing image percentile calibration for video modality.",
                flush=True,
            )

        print(
            "[INFO] Combined percentile quality calibration modalities: "
            f"{sorted(quality_calibrators)}",
            flush=True,
        )
        missing_enabled_modalities = sorted(enabled_modalities - set(quality_calibrators))
        if missing_enabled_modalities:
            print(
                "[WARN] Calibration file does not contain enabled modalities; raw scores will be used for: "
                f"{missing_enabled_modalities}",
                flush=True,
            )

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

    print(f"[INFO] Total samples to process: {len(samples)}", flush=True)
    if not samples:
        raise RuntimeError("No samples found for the selected dataset/modality configuration.")

    print("[INFO] Loading model...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.qwen_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        enable_audio_output=False,
    )
    model.disable_talker()
    install_quality_aware_first_attention_patch(model)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen_model_id)
    print("[INFO] Model loaded.", flush=True)
    print("[INFO] Quality-aware first attention layer patch enabled.", flush=True)
    if args.force_quality_scores_one:
        print("[INFO] Overriding computed modality quality scores to 1.0 for this run.", flush=True)
    elif args.quality_placebo_random:
        print(
            "[INFO] Placebo mode active: overriding modality quality scores with deterministic pseudo-random values "
            f"(seed={args.quality_placebo_random_seed}).",
            flush=True,
        )

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
            replies, raw_score_batch, calibrated_score_batch = run_batch_generation(
                model=model,
                processor=processor,
                entries=entries,
                enabled_modalities=enabled_modalities,
                device=device,
                dtype=dtype,
                force_quality_scores_one=args.force_quality_scores_one,
                quality_calibrators=quality_calibrators,
                qwen_quality=args.qwen_quality,
                quality_placebo_random=args.quality_placebo_random,
                quality_placebo_random_seed=args.quality_placebo_random_seed,
            )
        except Exception:
            print(
                f"[WARN] Batch {batch_start + 1}-{batch_start + len(batch_samples)} failed; retrying individually.",
                flush=True,
            )
            for entry in entries:
                sample = entry["sample"]
                try:
                    replies, raw_scores_single, calibrated_scores_single = run_batch_generation(
                        model=model,
                        processor=processor,
                        entries=[entry],
                        enabled_modalities=enabled_modalities,
                        device=device,
                        dtype=dtype,
                        force_quality_scores_one=args.force_quality_scores_one,
                        quality_calibrators=quality_calibrators,
                        qwen_quality=args.qwen_quality,
                        quality_placebo_random=args.quality_placebo_random,
                        quality_placebo_random_seed=args.quality_placebo_random_seed,
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
                    if args.quality_score_out_path:
                        append_quality_score_row(
                            path=args.quality_score_out_path,
                            dataset=dataset,
                            sample=sample,
                            raw_scores=raw_scores_single[0],
                            calibrated_scores=calibrated_scores_single[0],
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

        for entry, reply, raw_scores, calibrated_scores in zip(
            entries,
            replies,
            raw_score_batch,
            calibrated_score_batch,
        ):
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
            if args.quality_score_out_path:
                append_quality_score_row(
                    path=args.quality_score_out_path,
                    dataset=dataset,
                    sample=sample,
                    raw_scores=raw_scores,
                    calibrated_scores=calibrated_scores,
                )
            num_written += 1
            if num_written % 20 == 0 or num_written == 1:
                print(f"[INFO] Wrote prediction {num_written}/{len(samples)} -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
