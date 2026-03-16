from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.quality_estimation import (
    _compute_audio_pam_score,
    _compute_image_brisque_score_from_qwen_image,
    _compute_text_inverse_perplexities,
    _compute_video_brisque_score_from_qwen_video,
)

SUPPORTED_MODALITIES = ("text", "audio", "image", "video")
KADID_FILENAME_PATTERN = re.compile(r"I\d+_(\d+)_(\d+)\.png$", re.IGNORECASE)
ODAQ_LD_PATTERN = re.compile(r"_LD(\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw modality quality scores from imported calibration manifests under "
            "data/calibration_data/manifests."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/calibration_data/manifests",
        help="Directory containing modality manifest CSV files created by import_calibration_datasets.py.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="text,audio,image,video",
        help="Comma-separated modalities to score.",
    )
    parser.add_argument(
        "--max-files-per-modality",
        type=int,
        default=0,
        help="Maximum files to score per modality across all manifests. Use 0 to score all files.",
    )
    parser.add_argument(
        "--batch-size-text",
        type=int,
        default=16,
        help="Batch size for text quality scoring.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output CSV path for raw quality scores.",
    )
    return parser.parse_args()


def _parse_modalities(modalities_arg: str) -> list[str]:
    modalities = [token.strip().lower() for token in modalities_arg.split(",") if token.strip()]
    bad = sorted(set(modalities) - set(SUPPORTED_MODALITIES))
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Supported: {list(SUPPORTED_MODALITIES)}")
    if not modalities:
        raise ValueError("No modalities selected.")
    return modalities


def _manifest_paths_for_modality(manifest_dir: Path, modality: str) -> list[Path]:
    return sorted(manifest_dir.glob(f"{modality}_*.csv"))


def _load_manifest_rows(manifest_paths: list[Path], max_files: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest_path in manifest_paths:
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                if max_files > 0 and len(rows) >= max_files:
                    return rows
    return rows


def _dataset_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        dataset_name = row.get("dataset", "unknown")
        counts[dataset_name] = counts.get(dataset_name, 0) + 1
    return counts


def _quality_coverage_message(modality: str, rows: list[dict[str, str]]) -> str | None:
    if modality == "image":
        distortion_types: set[int] = set()
        severity_levels: set[int] = set()
        for row in rows:
            if row.get("dataset", "").lower() != "kadid10k":
                continue
            file_name = Path(row.get("relative_path", "")).name
            if file_name.startswith("._"):
                file_name = file_name[2:]
            match = KADID_FILENAME_PATTERN.search(file_name)
            if match is None:
                continue
            distortion_types.add(int(match.group(1)))
            severity_levels.add(int(match.group(2)))
        if distortion_types and severity_levels:
            return (
                "KADID quality coverage: "
                f"distortion_types={len(distortion_types)} "
                f"severity_levels={sorted(severity_levels)}"
            )
        return None

    if modality == "audio":
        ld_levels: set[int] = set()
        for row in rows:
            if row.get("dataset", "").lower() != "odaq":
                continue
            match = ODAQ_LD_PATTERN.search(row.get("relative_path", ""))
            if match is None:
                continue
            ld_levels.add(int(match.group(1)))
        if ld_levels:
            return f"ODAQ quality coverage: ld_levels={sorted(ld_levels)}"
        return None

    return None


def _write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "modality",
        "file_index",
        "absolute_path",
        "text_raw_quality",
        "audio_raw_quality",
        "image_raw_quality",
        "video_raw_quality",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_text_rows(
    rows: list[dict[str, str]],
    model,
    processor,
    device,
    batch_size: int,
    max_total_chunks: int | None = None,
) -> list[dict]:
    scored_rows = []
    batch_texts: list[str] = []
    batch_rows: list[dict[str, str]] = []
    processed_chunks = 0

    def _chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 120) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        chunks = []
        buffer: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line)
            if buffer and current_len + 1 + line_len > max_chars and current_len >= min_chars:
                chunks.append(" ".join(buffer))
                buffer = [line]
                current_len = line_len
            else:
                buffer.append(line)
                current_len += (1 if current_len > 0 else 0) + line_len
        if buffer and current_len >= min_chars:
            chunks.append(" ".join(buffer))
        elif buffer and not chunks:
            chunks.append(" ".join(buffer))
        return chunks

    def _flush_batch():
        nonlocal batch_texts, batch_rows
        if not batch_texts:
            return
        scores = _compute_text_inverse_perplexities(batch_texts, model, processor, device)
        for src_row, score in zip(batch_rows, scores):
            scored_rows.append(
                {
                    "dataset": src_row["dataset"],
                    "modality": "text",
                    "file_index": src_row["file_index"],
                    "absolute_path": src_row["absolute_path"],
                    "text_raw_quality": score,
                    "audio_raw_quality": "",
                    "image_raw_quality": "",
                    "video_raw_quality": "",
                }
            )
        batch_texts = []
        batch_rows = []

    for row in rows:
        if max_total_chunks is not None and processed_chunks >= max_total_chunks:
            break
        path = Path(row["absolute_path"])
        try:
            text_value = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not text_value:
            continue
        text_chunks = _chunk_text(text_value)
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if not chunk_text:
                continue
            if max_total_chunks is not None and processed_chunks >= max_total_chunks:
                break
            chunk_row = dict(row)
            chunk_row["file_index"] = f"{row.get('file_index', '0')}:{chunk_idx}"
            batch_texts.append(chunk_text)
            batch_rows.append(chunk_row)
            processed_chunks += 1
            if len(batch_texts) >= batch_size:
                _flush_batch()
    _flush_batch()
    return scored_rows


def _score_audio_rows(rows: list[dict[str, str]], device) -> list[dict]:
    scored_rows = []
    first_errors = []
    for row in rows:
        path = row["absolute_path"]
        try:
            score = _compute_audio_pam_score(path, device=device)
        except Exception as exc:
            if len(first_errors) < 3:
                first_errors.append(f"{path}: {exc}")
            continue
        scored_rows.append(
            {
                "dataset": row["dataset"],
                "modality": "audio",
                "file_index": row["file_index"],
                "absolute_path": path,
                "text_raw_quality": "",
                "audio_raw_quality": score,
                "image_raw_quality": "",
                "video_raw_quality": "",
            }
        )
    if first_errors:
        print("[WARN] Audio scoring errors (first few):", flush=True)
        for err in first_errors:
            print(f"[WARN]   {err}", flush=True)
    return scored_rows


def _score_image_rows(rows: list[dict[str, str]]) -> list[dict]:
    scored_rows = []
    for row in rows:
        path = Path(row["absolute_path"])
        try:
            with Image.open(path) as image:
                image_rgb = image.convert("RGB")
                score = _compute_image_brisque_score_from_qwen_image(image_rgb)
        except Exception:
            continue
        scored_rows.append(
            {
                "dataset": row["dataset"],
                "modality": "image",
                "file_index": row["file_index"],
                "absolute_path": str(path),
                "text_raw_quality": "",
                "audio_raw_quality": "",
                "image_raw_quality": score,
                "video_raw_quality": "",
            }
        )
    return scored_rows


def _score_video_rows(rows: list[dict[str, str]]) -> list[dict]:
    import torchvision

    scored_rows = []
    for row in rows:
        path = row["absolute_path"]
        try:
            video_tensor, _, _ = torchvision.io.read_video(path, pts_unit="sec")
            if video_tensor.numel() == 0:
                continue
            # (T,H,W,C) -> (T,C,H,W)
            video_tensor = video_tensor.permute(0, 3, 1, 2).to(dtype=torch.float32)
            frame_count = video_tensor.shape[0]
            if frame_count > 16:
                idx = torch.linspace(0, frame_count - 1, steps=16).round().long()
                video_tensor = video_tensor[idx]
            score = _compute_video_brisque_score_from_qwen_video(video_tensor)
        except Exception:
            continue
        scored_rows.append(
            {
                "dataset": row["dataset"],
                "modality": "video",
                "file_index": row["file_index"],
                "absolute_path": path,
                "text_raw_quality": "",
                "audio_raw_quality": "",
                "image_raw_quality": "",
                "video_raw_quality": score,
            }
        )
    return scored_rows


def main():
    args = parse_args()
    modalities = _parse_modalities(args.modalities)
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    rows_by_modality: dict[str, list[dict[str, str]]] = {}
    manifests_by_modality: dict[str, list[Path]] = {}
    row_limit_label = "all" if args.max_files_per_modality <= 0 else str(args.max_files_per_modality)
    for modality in modalities:
        manifest_paths = _manifest_paths_for_modality(manifest_dir, modality)
        manifests_by_modality[modality] = manifest_paths
        if not manifest_paths:
            rows_by_modality[modality] = []
            continue
        rows_by_modality[modality] = _load_manifest_rows(manifest_paths, args.max_files_per_modality)
        dataset_counts = _dataset_counts(rows_by_modality[modality])
        dataset_counts_message = ", ".join(
            f"{dataset}={count}" for dataset, count in sorted(dataset_counts.items())
        )
        print(
            f"[INFO] Prepared modality={modality} rows={len(rows_by_modality[modality])} "
            f"(max_files_per_modality={row_limit_label}) datasets=[{dataset_counts_message}]",
            flush=True,
        )
        coverage_message = _quality_coverage_message(modality, rows_by_modality[modality])
        if coverage_message:
            print(f"[INFO] {coverage_message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    processor = None
    if "text" in modalities and rows_by_modality.get("text"):
        print("[INFO] Loading Qwen model for text quality scoring...", flush=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
            enable_audio_output=False,
        )
        model.disable_talker()
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        device = next(model.parameters()).device
        print(f"[INFO] Text quality model ready on {device}", flush=True)

    out_path = Path(args.out_path)
    if out_path.exists():
        out_path.unlink()

    for modality in modalities:
        manifest_paths = manifests_by_modality.get(modality, [])
        if not manifest_paths:
            print(f"[WARN] No manifests found for modality={modality} in {manifest_dir}", flush=True)
            continue
        rows = rows_by_modality.get(modality, [])
        if not rows:
            print(f"[WARN] Empty manifests for modality={modality}", flush=True)
            continue

        print(
            f"[INFO] Scoring modality={modality} from {len(manifest_paths)} manifest(s), rows={len(rows)}",
            flush=True,
        )
        if modality == "text":
            if model is None or processor is None:
                print("[WARN] Skipping text scoring because no text model is loaded.", flush=True)
                continue
            text_chunk_limit = args.max_files_per_modality if args.max_files_per_modality > 0 else None
            scored_rows = _score_text_rows(
                rows,
                model,
                processor,
                device,
                args.batch_size_text,
                max_total_chunks=text_chunk_limit,
            )
        elif modality == "audio":
            scored_rows = _score_audio_rows(rows, device)
        elif modality == "image":
            scored_rows = _score_image_rows(rows)
        else:
            scored_rows = _score_video_rows(rows)

        if rows and not scored_rows:
            raise RuntimeError(
                f"Scoring produced zero usable rows for modality={modality}. "
                "Check manifest paths and modality dependencies."
            )
        _write_rows(out_path, scored_rows)
        print(f"[INFO] modality={modality} scored_rows={len(scored_rows)}", flush=True)

    print(f"[INFO] Wrote raw calibration scores to {out_path}", flush=True)


if __name__ == "__main__":
    main()
