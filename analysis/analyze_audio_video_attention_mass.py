from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


DEFAULT_AUDIO_TOKENS = ("<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>")
DEFAULT_VIDEO_TOKENS = ("<|VIDEO|>", "<|vision_bos|>", "<|vision_eos|>")

# Triplets are serialized as: idx:token_id:token:weight|idx:token_id:token:weight|...
# Token strings may contain "|" and ":" (e.g., <|AUDIO|>), so parsing must use
# a boundary-aware regex rather than str.split("|").
TOPK_ENTRY_RE = re.compile(
    r"(?P<idx>\d+):(?P<token_id>-?\d+):(?P<token>.*?):(?P<weight>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?=(?:\|\d+:-?\d+:)|$)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze summed attention mass on audio vs video tokens in first-layer QAA detail CSVs, "
            "comparing baseline attention to quality-adjusted attention."
        )
    )
    parser.add_argument(
        "--detail-csv",
        type=str,
        action="append",
        required=True,
        help="Path to *_detail.csv from analyze_first_layer_qaa_attention_shift.py (repeatable).",
    )
    parser.add_argument(
        "--audio-token",
        type=str,
        action="append",
        default=None,
        help="Audio token string to include (repeatable). Defaults include <|AUDIO|>, <|audio_bos|>, <|audio_eos|>.",
    )
    parser.add_argument(
        "--video-token",
        type=str,
        action="append",
        default=None,
        help="Video token string to include (repeatable). Defaults include <|VIDEO|>, <|vision_bos|>, <|vision_eos|>.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/analysis/first_layer_qaa_attention",
        help="Output directory for generated summaries.",
    )
    parser.add_argument(
        "--write-query-csv",
        action="store_true",
        help="Write per-query audio/video mass CSV (can be large).",
    )
    return parser.parse_args()


def _derived_prefix(detail_csv_path: Path) -> str:
    stem = detail_csv_path.stem
    if stem.endswith("_detail"):
        return stem[: -len("_detail")]
    return stem


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _token_weight_iter(serialized_topk: str):
    for match in TOPK_ENTRY_RE.finditer(serialized_topk):
        yield match.group("token"), float(match.group("weight"))


def _compute_modality_masses(serialized_topk: str, audio_tokens: set[str], video_tokens: set[str]):
    audio_mass = 0.0
    video_mass = 0.0
    total_mass = 0.0
    token_count = 0
    for token, weight in _token_weight_iter(serialized_topk):
        token_count += 1
        total_mass += weight
        if token in audio_tokens:
            audio_mass += weight
        elif token in video_tokens:
            video_mass += weight
    return audio_mass, video_mass, total_mass, token_count


def _new_sample_bucket():
    return {
        "query_count": 0,
        "baseline_audio_sum": 0.0,
        "baseline_video_sum": 0.0,
        "adjusted_audio_sum": 0.0,
        "adjusted_video_sum": 0.0,
        "delta_audio_sum": 0.0,
        "delta_video_sum": 0.0,
        "baseline_audio_share_av_sum": 0.0,
        "adjusted_audio_share_av_sum": 0.0,
        "baseline_audio_share_av_count": 0,
        "adjusted_audio_share_av_count": 0,
        "baseline_audio_gt_video_count": 0,
        "adjusted_audio_gt_video_count": 0,
    }


def analyze_detail_csv(
    detail_csv_path: Path,
    out_dir: Path,
    audio_tokens: set[str],
    video_tokens: set[str],
    write_query_csv: bool,
):
    prefix = _derived_prefix(detail_csv_path)
    summary_path = out_dir / f"{prefix}_av_mass_summary.json"
    sample_csv_path = out_dir / f"{prefix}_sample_av_mass.csv"
    query_csv_path = out_dir / f"{prefix}_query_av_mass.csv"

    aggregate = {
        "query_count": 0,
        "baseline_audio_sum": 0.0,
        "baseline_video_sum": 0.0,
        "adjusted_audio_sum": 0.0,
        "adjusted_video_sum": 0.0,
        "delta_audio_sum": 0.0,
        "delta_video_sum": 0.0,
        "baseline_total_mass_sum": 0.0,
        "adjusted_total_mass_sum": 0.0,
        "baseline_token_count_sum": 0,
        "adjusted_token_count_sum": 0,
        "baseline_audio_share_av_sum": 0.0,
        "adjusted_audio_share_av_sum": 0.0,
        "baseline_audio_share_av_count": 0,
        "adjusted_audio_share_av_count": 0,
        "baseline_audio_gt_video_count": 0,
        "adjusted_audio_gt_video_count": 0,
    }

    per_sample = {}

    query_writer = None
    query_handle = None
    if write_query_csv:
        query_handle = query_csv_path.open("w", newline="", encoding="utf-8")
        query_writer = csv.DictWriter(
            query_handle,
            fieldnames=[
                "dataset",
                "split",
                "sample_id",
                "file",
                "query_idx",
                "baseline_audio_mass",
                "baseline_video_mass",
                "adjusted_audio_mass",
                "adjusted_video_mass",
                "delta_audio_mass",
                "delta_video_mass",
                "baseline_audio_share_in_av_mass",
                "adjusted_audio_share_in_av_mass",
                "baseline_audio_vs_video_ratio",
                "adjusted_audio_vs_video_ratio",
                "baseline_total_mass_from_topk",
                "adjusted_total_mass_from_topk",
                "baseline_parsed_token_count",
                "adjusted_parsed_token_count",
            ],
        )
        query_writer.writeheader()

    with detail_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            baseline_audio, baseline_video, baseline_total, baseline_token_count = _compute_modality_masses(
                row["topk_before"], audio_tokens, video_tokens
            )
            adjusted_audio, adjusted_video, adjusted_total, adjusted_token_count = _compute_modality_masses(
                row["topk_after"], audio_tokens, video_tokens
            )

            baseline_av_mass = baseline_audio + baseline_video
            adjusted_av_mass = adjusted_audio + adjusted_video
            baseline_audio_share_av = _safe_ratio(baseline_audio, baseline_av_mass)
            adjusted_audio_share_av = _safe_ratio(adjusted_audio, adjusted_av_mass)
            delta_audio = adjusted_audio - baseline_audio
            delta_video = adjusted_video - baseline_video

            aggregate["query_count"] += 1
            aggregate["baseline_audio_sum"] += baseline_audio
            aggregate["baseline_video_sum"] += baseline_video
            aggregate["adjusted_audio_sum"] += adjusted_audio
            aggregate["adjusted_video_sum"] += adjusted_video
            aggregate["delta_audio_sum"] += delta_audio
            aggregate["delta_video_sum"] += delta_video
            aggregate["baseline_total_mass_sum"] += baseline_total
            aggregate["adjusted_total_mass_sum"] += adjusted_total
            aggregate["baseline_token_count_sum"] += baseline_token_count
            aggregate["adjusted_token_count_sum"] += adjusted_token_count

            if not math.isnan(baseline_audio_share_av):
                aggregate["baseline_audio_share_av_sum"] += baseline_audio_share_av
                aggregate["baseline_audio_share_av_count"] += 1
            if not math.isnan(adjusted_audio_share_av):
                aggregate["adjusted_audio_share_av_sum"] += adjusted_audio_share_av
                aggregate["adjusted_audio_share_av_count"] += 1
            if baseline_audio > baseline_video:
                aggregate["baseline_audio_gt_video_count"] += 1
            if adjusted_audio > adjusted_video:
                aggregate["adjusted_audio_gt_video_count"] += 1

            sample_key = (row["sample_id"], row["file"], row.get("split", ""))
            sample_bucket = per_sample.get(sample_key)
            if sample_bucket is None:
                sample_bucket = _new_sample_bucket()
                per_sample[sample_key] = sample_bucket
            sample_bucket["query_count"] += 1
            sample_bucket["baseline_audio_sum"] += baseline_audio
            sample_bucket["baseline_video_sum"] += baseline_video
            sample_bucket["adjusted_audio_sum"] += adjusted_audio
            sample_bucket["adjusted_video_sum"] += adjusted_video
            sample_bucket["delta_audio_sum"] += delta_audio
            sample_bucket["delta_video_sum"] += delta_video
            if not math.isnan(baseline_audio_share_av):
                sample_bucket["baseline_audio_share_av_sum"] += baseline_audio_share_av
                sample_bucket["baseline_audio_share_av_count"] += 1
            if not math.isnan(adjusted_audio_share_av):
                sample_bucket["adjusted_audio_share_av_sum"] += adjusted_audio_share_av
                sample_bucket["adjusted_audio_share_av_count"] += 1
            if baseline_audio > baseline_video:
                sample_bucket["baseline_audio_gt_video_count"] += 1
            if adjusted_audio > adjusted_video:
                sample_bucket["adjusted_audio_gt_video_count"] += 1

            if query_writer is not None:
                query_writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "split": row.get("split", ""),
                        "sample_id": row["sample_id"],
                        "file": row["file"],
                        "query_idx": row["query_idx"],
                        "baseline_audio_mass": baseline_audio,
                        "baseline_video_mass": baseline_video,
                        "adjusted_audio_mass": adjusted_audio,
                        "adjusted_video_mass": adjusted_video,
                        "delta_audio_mass": delta_audio,
                        "delta_video_mass": delta_video,
                        "baseline_audio_share_in_av_mass": baseline_audio_share_av,
                        "adjusted_audio_share_in_av_mass": adjusted_audio_share_av,
                        "baseline_audio_vs_video_ratio": _safe_ratio(baseline_audio, baseline_video),
                        "adjusted_audio_vs_video_ratio": _safe_ratio(adjusted_audio, adjusted_video),
                        "baseline_total_mass_from_topk": baseline_total,
                        "adjusted_total_mass_from_topk": adjusted_total,
                        "baseline_parsed_token_count": baseline_token_count,
                        "adjusted_parsed_token_count": adjusted_token_count,
                    }
                )

    if query_handle is not None:
        query_handle.close()

    query_count = aggregate["query_count"]
    if query_count < 1:
        raise RuntimeError(f"No rows found in detail CSV: {detail_csv_path}")

    baseline_audio_share_av_avg = (
        aggregate["baseline_audio_share_av_sum"] / aggregate["baseline_audio_share_av_count"]
        if aggregate["baseline_audio_share_av_count"] > 0
        else float("nan")
    )
    adjusted_audio_share_av_avg = (
        aggregate["adjusted_audio_share_av_sum"] / aggregate["adjusted_audio_share_av_count"]
        if aggregate["adjusted_audio_share_av_count"] > 0
        else float("nan")
    )

    summary = {
        "detail_csv_path": str(detail_csv_path),
        "audio_tokens": sorted(audio_tokens),
        "video_tokens": sorted(video_tokens),
        "query_count": query_count,
        "avg_baseline_audio_mass": aggregate["baseline_audio_sum"] / query_count,
        "avg_baseline_video_mass": aggregate["baseline_video_sum"] / query_count,
        "avg_adjusted_audio_mass": aggregate["adjusted_audio_sum"] / query_count,
        "avg_adjusted_video_mass": aggregate["adjusted_video_sum"] / query_count,
        "avg_delta_audio_mass_adjusted_minus_baseline": aggregate["delta_audio_sum"] / query_count,
        "avg_delta_video_mass_adjusted_minus_baseline": aggregate["delta_video_sum"] / query_count,
        "avg_baseline_audio_share_in_av_mass": baseline_audio_share_av_avg,
        "avg_adjusted_audio_share_in_av_mass": adjusted_audio_share_av_avg,
        "delta_audio_share_in_av_mass_adjusted_minus_baseline": adjusted_audio_share_av_avg - baseline_audio_share_av_avg,
        "baseline_audio_gt_video_rate": aggregate["baseline_audio_gt_video_count"] / query_count,
        "adjusted_audio_gt_video_rate": aggregate["adjusted_audio_gt_video_count"] / query_count,
        "avg_baseline_total_mass_from_topk": aggregate["baseline_total_mass_sum"] / query_count,
        "avg_adjusted_total_mass_from_topk": aggregate["adjusted_total_mass_sum"] / query_count,
        "avg_baseline_parsed_token_count": aggregate["baseline_token_count_sum"] / query_count,
        "avg_adjusted_parsed_token_count": aggregate["adjusted_token_count_sum"] / query_count,
        "sample_csv_path": str(sample_csv_path),
        "query_csv_path": str(query_csv_path) if write_query_csv else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with sample_csv_path.open("w", newline="", encoding="utf-8") as sample_handle:
        sample_writer = csv.DictWriter(
            sample_handle,
            fieldnames=[
                "sample_id",
                "file",
                "split",
                "query_count",
                "avg_baseline_audio_mass",
                "avg_baseline_video_mass",
                "avg_adjusted_audio_mass",
                "avg_adjusted_video_mass",
                "avg_delta_audio_mass_adjusted_minus_baseline",
                "avg_delta_video_mass_adjusted_minus_baseline",
                "avg_baseline_audio_share_in_av_mass",
                "avg_adjusted_audio_share_in_av_mass",
                "delta_audio_share_in_av_mass_adjusted_minus_baseline",
                "baseline_audio_gt_video_rate",
                "adjusted_audio_gt_video_rate",
            ],
        )
        sample_writer.writeheader()
        for (sample_id, file_name, split), bucket in sorted(per_sample.items()):
            sample_query_count = bucket["query_count"]
            baseline_share_avg = (
                bucket["baseline_audio_share_av_sum"] / bucket["baseline_audio_share_av_count"]
                if bucket["baseline_audio_share_av_count"] > 0
                else float("nan")
            )
            adjusted_share_avg = (
                bucket["adjusted_audio_share_av_sum"] / bucket["adjusted_audio_share_av_count"]
                if bucket["adjusted_audio_share_av_count"] > 0
                else float("nan")
            )
            sample_writer.writerow(
                {
                    "sample_id": sample_id,
                    "file": file_name,
                    "split": split,
                    "query_count": sample_query_count,
                    "avg_baseline_audio_mass": bucket["baseline_audio_sum"] / sample_query_count,
                    "avg_baseline_video_mass": bucket["baseline_video_sum"] / sample_query_count,
                    "avg_adjusted_audio_mass": bucket["adjusted_audio_sum"] / sample_query_count,
                    "avg_adjusted_video_mass": bucket["adjusted_video_sum"] / sample_query_count,
                    "avg_delta_audio_mass_adjusted_minus_baseline": bucket["delta_audio_sum"] / sample_query_count,
                    "avg_delta_video_mass_adjusted_minus_baseline": bucket["delta_video_sum"] / sample_query_count,
                    "avg_baseline_audio_share_in_av_mass": baseline_share_avg,
                    "avg_adjusted_audio_share_in_av_mass": adjusted_share_avg,
                    "delta_audio_share_in_av_mass_adjusted_minus_baseline": adjusted_share_avg - baseline_share_avg,
                    "baseline_audio_gt_video_rate": bucket["baseline_audio_gt_video_count"] / sample_query_count,
                    "adjusted_audio_gt_video_rate": bucket["adjusted_audio_gt_video_count"] / sample_query_count,
                }
            )

    print(f"[INFO] Processed detail CSV: {detail_csv_path}", flush=True)
    print(f"[INFO] Summary JSON: {summary_path}", flush=True)
    print(f"[INFO] Sample CSV: {sample_csv_path}", flush=True)
    if write_query_csv:
        print(f"[INFO] Query CSV: {query_csv_path}", flush=True)
    print(
        "[INFO] Avg baseline audio/video mass: "
        f"{summary['avg_baseline_audio_mass']:.6f}/{summary['avg_baseline_video_mass']:.6f}",
        flush=True,
    )
    print(
        "[INFO] Avg adjusted audio/video mass: "
        f"{summary['avg_adjusted_audio_mass']:.6f}/{summary['avg_adjusted_video_mass']:.6f}",
        flush=True,
    )
    print(
        "[INFO] Avg delta audio/video mass (adjusted-baseline): "
        f"{summary['avg_delta_audio_mass_adjusted_minus_baseline']:.6f}/"
        f"{summary['avg_delta_video_mass_adjusted_minus_baseline']:.6f}",
        flush=True,
    )


def main():
    args = parse_args()
    detail_paths = [Path(p) for p in args.detail_csv]
    for path in detail_paths:
        if not path.exists():
            raise FileNotFoundError(f"Detail CSV not found: {path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_tokens = set(args.audio_token) if args.audio_token else set(DEFAULT_AUDIO_TOKENS)
    video_tokens = set(args.video_token) if args.video_token else set(DEFAULT_VIDEO_TOKENS)
    if not audio_tokens:
        raise ValueError("Audio token set is empty.")
    if not video_tokens:
        raise ValueError("Video token set is empty.")

    for detail_path in detail_paths:
        analyze_detail_csv(
            detail_csv_path=detail_path,
            out_dir=out_dir,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
            write_query_csv=args.write_query_csv,
        )


if __name__ == "__main__":
    main()
