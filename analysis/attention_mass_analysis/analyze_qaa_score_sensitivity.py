from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PREDICTION_GLOBS = [
    "out/force_scored/*/predictions_*_noise_*_audio*_video*.csv",
]
DEFAULT_DETAIL_GLOBS = [
    "out/analysis/first_layer_qaa_attention/*_detail.csv",
]
DEFAULT_BASELINE_PATTERN = "out/Qwen_7B/meld/{task}/prediction_{modalities}_noise_{noisy}.csv"

FORCED_PREDICTION_RE = re.compile(
    r"^predictions_(?P<modalities>[a-z]+)_noise_(?P<noisy>[a-z]+)_"
    r"audio(?P<audio>[0-9]+(?:\.[0-9]+)?)_video(?P<video>[0-9]+(?:\.[0-9]+)?)\.csv$"
)
DETAIL_FILE_RE = re.compile(
    r"^(?P<task>[a-z]+)_(?P<modalities>[a-z]+)_noise_(?P<noisy>[a-z]+)_(?P<score_tag>.+)_detail\.csv$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidated QAA sensitivity analysis over prediction CSVs and first-layer attention detail CSVs. "
            "Computes score-invariance diagnostics, pairwise disagreement, and scaled-vs-unscaled dominance metrics."
        )
    )
    parser.add_argument(
        "--prediction-glob",
        type=str,
        action="append",
        default=None,
        help=(
            "Glob(s) for prediction CSVs to compare. "
            "Defaults to out/force_scored/*/predictions_*_noise_*_audio*_video*.csv."
        ),
    )
    parser.add_argument(
        "--detail-glob",
        type=str,
        action="append",
        default=None,
        help=(
            "Glob(s) for first-layer detail CSVs. "
            "Defaults to out/analysis/first_layer_qaa_attention/*_detail.csv."
        ),
    )
    parser.add_argument(
        "--baseline-pattern",
        type=str,
        default=DEFAULT_BASELINE_PATTERN,
        help=(
            "Path pattern for baseline prediction CSV lookup with placeholders "
            "{task}, {modalities}, {noisy}. Set empty string to disable baseline comparisons."
        ),
    )
    parser.add_argument(
        "--normalize-label-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize labels/predictions to lowercase before metric computation (default: true).",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=100,
        help="Minimum shared rows required for pairwise disagreement comparisons.",
    )
    parser.add_argument(
        "--tvd-threshold",
        type=float,
        action="append",
        default=None,
        help=(
            "Repeatable TVD threshold(s) for attention-level overall-change rates, "
            "for example: --tvd-threshold 0 --tvd-threshold 0.001 --tvd-threshold 0.01."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/analysis/qaa_score_sensitivity",
        help="Output directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="qaa_score_sensitivity",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _expand_globs(globs: list[str]) -> list[Path]:
    paths = []
    for pattern in globs:
        paths.extend(Path("..").glob(pattern))
    deduped = sorted({p.resolve() for p in paths if p.is_file()})
    return [Path(p) for p in deduped]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _format_threshold_label(threshold: float) -> str:
    return f"{threshold:.6g}"


def _parse_score_token(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token) is None:
        return None
    if "." in token:
        return float(token)
    if token in {"0", "1"}:
        return float(token)
    if token.startswith("0"):
        return int(token) / (10 ** len(token))
    token_int = int(token)
    if token_int <= 100:
        return token_int / 100.0
    return token_int / 1000.0


def _normalize_label(value: str, normalize_case: bool) -> str:
    normalized = (value or "").strip()
    return normalized.lower() if normalize_case else normalized


def _row_key(row: dict[str, str], row_index: int) -> tuple:
    split = row.get("split", "")
    sample_id = row.get("sample_id", "")
    file_name = row.get("file", "")
    dialog_id = row.get("dialog_id", "")
    utterance_id = row.get("utterance_id", "")
    if sample_id:
        return split, sample_id, file_name
    if dialog_id or utterance_id:
        return split, dialog_id, utterance_id, file_name
    return ("row", row_index)


def _read_prediction_rows(path: Path, *, normalize_case: bool) -> dict[tuple, tuple[str, str]]:
    rows: dict[tuple, tuple[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            key = _row_key(row, idx)
            pred = _normalize_label(row.get("prediction", ""), normalize_case)
            label = _normalize_label(row.get("label", ""), normalize_case)
            rows[key] = (pred, label)
    return rows


def _metrics_from_rows(rows: dict[tuple, tuple[str, str]]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
        }

    labels = [label for _, label in rows.values()]
    preds = [pred for pred, _ in rows.values()]
    n = len(labels)
    accuracy = _safe_div(sum(1 for pred, label in zip(preds, labels) if pred == label), n)

    class_labels = sorted(set(labels) | set(preds))
    class_f1_scores = []
    for class_label in class_labels:
        tp = sum(1 for pred, label in zip(preds, labels) if pred == class_label and label == class_label)
        fp = sum(1 for pred, label in zip(preds, labels) if pred == class_label and label != class_label)
        fn = sum(1 for pred, label in zip(preds, labels) if pred != class_label and label == class_label)
        denominator = 2 * tp + fp + fn
        class_f1_scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    macro_f1 = sum(class_f1_scores) / len(class_f1_scores) if class_f1_scores else float("nan")

    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def _aligned_subset(rows: dict[tuple, tuple[str, str]], keys: set[tuple]) -> dict[tuple, tuple[str, str]]:
    return {key: rows[key] for key in keys if key in rows}


def _pairwise_disagreement(
    rows_a: dict[tuple, tuple[str, str]],
    rows_b: dict[tuple, tuple[str, str]],
) -> tuple[int, float]:
    shared = set(rows_a) & set(rows_b)
    if not shared:
        return 0, float("nan")
    disagreements = sum(1 for key in shared if rows_a[key][0] != rows_b[key][0])
    return len(shared), disagreements / len(shared)


def _pearson(xs: list[float], ys: list[float]) -> float:
    paired = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(paired) < 2:
        return float("nan")
    x_values = [x for x, _ in paired]
    y_values = [y for _, y in paired]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in paired)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _group_key_for_prediction(path: Path) -> tuple[str, str, str]:
    match = FORCED_PREDICTION_RE.match(path.name)
    if match is None:
        return path.parent.name, "unknown", "unknown"
    task = path.parent.name
    return task, match.group("modalities"), match.group("noisy")


def _parse_prediction_metadata(path: Path) -> dict:
    metadata = {
        "task": path.parent.name,
        "modalities": None,
        "noisy_modalities": None,
        "audio_score": None,
        "video_score": None,
    }
    match = FORCED_PREDICTION_RE.match(path.name)
    if match is None:
        return metadata
    metadata["modalities"] = match.group("modalities")
    metadata["noisy_modalities"] = match.group("noisy")
    metadata["audio_score"] = _parse_score_token(match.group("audio"))
    metadata["video_score"] = _parse_score_token(match.group("video"))
    return metadata


def _analyze_prediction_group(
    paths: list[Path],
    *,
    normalize_case: bool,
    min_overlap: int,
    baseline_pattern: str | None,
) -> dict:
    loaded_rows = {}
    run_rows_all = []
    per_run = []

    for path in sorted(paths):
        rows = _read_prediction_rows(path, normalize_case=normalize_case)
        loaded_rows[path] = rows
        run_rows_all.append(set(rows))
        run_metrics = _metrics_from_rows(rows)
        meta = _parse_prediction_metadata(path)
        per_run.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "task": meta["task"],
                "modalities": meta["modalities"],
                "noisy_modalities": meta["noisy_modalities"],
                "audio_score": meta["audio_score"],
                "video_score": meta["video_score"],
                "metrics_all_rows": run_metrics,
            }
        )

    common_keys = set.intersection(*run_rows_all) if run_rows_all else set()
    for run in per_run:
        rows = loaded_rows[Path(run["file_path"])]
        aligned_rows = _aligned_subset(rows, common_keys)
        run["metrics_common_rows"] = _metrics_from_rows(aligned_rows)

    pairwise = []
    for idx_a in range(len(per_run)):
        for idx_b in range(idx_a + 1, len(per_run)):
            path_a = Path(per_run[idx_a]["file_path"])
            path_b = Path(per_run[idx_b]["file_path"])
            n_shared, disagreement_rate = _pairwise_disagreement(loaded_rows[path_a], loaded_rows[path_b])
            if n_shared < min_overlap:
                continue
            pairwise.append(
                {
                    "run_a": per_run[idx_a]["file_name"],
                    "run_b": per_run[idx_b]["file_name"],
                    "n_shared": n_shared,
                    "disagreement_rate": disagreement_rate,
                }
            )

    all_common_accuracies = [run["metrics_common_rows"]["accuracy"] for run in per_run]
    all_common_f1s = [run["metrics_common_rows"]["macro_f1"] for run in per_run]
    valid_acc = [x for x in all_common_accuracies if math.isfinite(x)]
    valid_f1 = [x for x in all_common_f1s if math.isfinite(x)]
    metric_summary = {
        "run_count": len(per_run),
        "common_row_count": len(common_keys),
        "accuracy_min_common": min(valid_acc) if valid_acc else float("nan"),
        "accuracy_max_common": max(valid_acc) if valid_acc else float("nan"),
        "accuracy_delta_common": (max(valid_acc) - min(valid_acc)) if valid_acc else float("nan"),
        "macro_f1_min_common": min(valid_f1) if valid_f1 else float("nan"),
        "macro_f1_max_common": max(valid_f1) if valid_f1 else float("nan"),
        "macro_f1_delta_common": (max(valid_f1) - min(valid_f1)) if valid_f1 else float("nan"),
        "avg_pairwise_disagreement": (
            sum(item["disagreement_rate"] for item in pairwise) / len(pairwise) if pairwise else float("nan")
        ),
        "max_pairwise_disagreement": max((item["disagreement_rate"] for item in pairwise), default=float("nan")),
    }

    audio_scores = [run.get("audio_score", float("nan")) for run in per_run]
    video_scores = [run.get("video_score", float("nan")) for run in per_run]
    score_diffs = [
        (audio - video) if (audio is not None and video is not None) else float("nan")
        for audio, video in zip(audio_scores, video_scores)
    ]
    score_sensitivity = {
        "pearson_audio_vs_accuracy_common": _pearson(audio_scores, all_common_accuracies),
        "pearson_audio_vs_macro_f1_common": _pearson(audio_scores, all_common_f1s),
        "pearson_audio_minus_video_vs_accuracy_common": _pearson(score_diffs, all_common_accuracies),
        "pearson_audio_minus_video_vs_macro_f1_common": _pearson(score_diffs, all_common_f1s),
    }

    baseline_summary = None
    if baseline_pattern:
        first_run = per_run[0]
        if first_run.get("modalities") and first_run.get("noisy_modalities"):
            baseline_path = Path(
                baseline_pattern.format(
                    task=first_run["task"],
                    modalities=first_run["modalities"],
                    noisy=first_run["noisy_modalities"],
                )
            )
            if baseline_path.exists():
                baseline_rows = _read_prediction_rows(baseline_path, normalize_case=normalize_case)
                baseline_metrics = _metrics_from_rows(baseline_rows)
                baseline_vs_runs = []
                for run in per_run:
                    path = Path(run["file_path"])
                    n_shared, disagreement_rate = _pairwise_disagreement(baseline_rows, loaded_rows[path])
                    if n_shared < min_overlap:
                        continue
                    overlap_keys = set(baseline_rows) & set(loaded_rows[path])
                    run_overlap_rows = _aligned_subset(loaded_rows[path], overlap_keys)
                    baseline_overlap_rows = _aligned_subset(baseline_rows, overlap_keys)
                    baseline_vs_runs.append(
                        {
                            "run_name": run["file_name"],
                            "n_shared": n_shared,
                            "disagreement_rate_vs_baseline": disagreement_rate,
                            "run_accuracy_on_overlap": _metrics_from_rows(run_overlap_rows)["accuracy"],
                            "run_macro_f1_on_overlap": _metrics_from_rows(run_overlap_rows)["macro_f1"],
                            "baseline_accuracy_on_overlap": _metrics_from_rows(baseline_overlap_rows)["accuracy"],
                            "baseline_macro_f1_on_overlap": _metrics_from_rows(baseline_overlap_rows)["macro_f1"],
                        }
                    )
                baseline_summary = {
                    "path": str(baseline_path),
                    "metrics_all_rows": baseline_metrics,
                    "comparisons": baseline_vs_runs,
                }

    return {
        "group_id": f"{per_run[0]['task']}|{per_run[0].get('modalities')}|noisy={per_run[0].get('noisy_modalities')}",
        "task": per_run[0]["task"],
        "modalities": per_run[0].get("modalities"),
        "noisy_modalities": per_run[0].get("noisy_modalities"),
        "runs": per_run,
        "metric_summary": metric_summary,
        "score_sensitivity": score_sensitivity,
        "pairwise_disagreement": pairwise,
        "baseline": baseline_summary,
    }


def _parse_detail_file_metadata(path: Path) -> dict:
    metadata = {
        "task": None,
        "modalities": None,
        "noisy_modalities": None,
        "score_tag": None,
    }
    match = DETAIL_FILE_RE.match(path.name)
    if match is None:
        return metadata
    metadata["task"] = match.group("task")
    metadata["modalities"] = match.group("modalities")
    metadata["noisy_modalities"] = match.group("noisy")
    metadata["score_tag"] = match.group("score_tag")
    return metadata


def _analyze_detail_csv(path: Path, *, tvd_thresholds: list[float]) -> dict:
    metadata = _parse_detail_file_metadata(path)
    top1_before_counter: Counter[str] = Counter()
    top1_after_counter: Counter[str] = Counter()

    n = 0
    before_scaled_count = 0
    after_scaled_count = 0
    before_quality_one_count = 0
    after_quality_one_count = 0
    changed_count = 0
    changed_before_scaled_count = 0
    changed_before_unscaled_count = 0
    tvd_sum = 0.0
    tvd_max = 0.0
    finite_tvd_count = 0
    tvd_threshold_counts = {threshold: 0 for threshold in tvd_thresholds}

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        header_fields = header.split(",")
        field_idx = {name: idx for idx, name in enumerate(header_fields)}

        required_fields = [
            "top1_before_token",
            "top1_after_token",
            "top1_before_scaled",
            "top1_after_scaled",
            "top1_before_quality",
            "top1_after_quality",
            "top1_changed",
            "total_variation_distance",
        ]
        missing = [field for field in required_fields if field not in field_idx]
        if missing:
            raise RuntimeError(f"Detail CSV is missing required columns {missing}: {path}")

        max_required_idx = max(field_idx[field] for field in required_fields)
        split_limit = max_required_idx + 1

        for line in handle:
            parts = line.rstrip("\n").split(",", split_limit)
            if len(parts) <= max_required_idx:
                continue

            n += 1

            before_token = parts[field_idx["top1_before_token"]]
            after_token = parts[field_idx["top1_after_token"]]
            top1_before_counter[before_token] += 1
            top1_after_counter[after_token] += 1

            before_scaled = parts[field_idx["top1_before_scaled"]] == "1"
            after_scaled = parts[field_idx["top1_after_scaled"]] == "1"
            before_scaled_count += int(before_scaled)
            after_scaled_count += int(after_scaled)

            before_quality = _safe_float(parts[field_idx["top1_before_quality"]])
            after_quality = _safe_float(parts[field_idx["top1_after_quality"]])
            before_quality_one_count += int(math.isfinite(before_quality) and abs(before_quality - 1.0) < 1e-9)
            after_quality_one_count += int(math.isfinite(after_quality) and abs(after_quality - 1.0) < 1e-9)

            changed = parts[field_idx["top1_changed"]] == "1"
            changed_count += int(changed)
            if changed and before_scaled:
                changed_before_scaled_count += 1
            if changed and not before_scaled:
                changed_before_unscaled_count += 1

            tvd = _safe_float(parts[field_idx["total_variation_distance"]])
            if math.isfinite(tvd):
                tvd_sum += tvd
                finite_tvd_count += 1
                if tvd > tvd_max:
                    tvd_max = tvd
                for threshold in tvd_thresholds:
                    if tvd > threshold:
                        tvd_threshold_counts[threshold] += 1

    if n == 0:
        raise RuntimeError(f"Detail CSV has no data rows: {path}")

    dominant_before_token, dominant_before_count = top1_before_counter.most_common(1)[0]
    dominant_after_token, dominant_after_count = top1_after_counter.most_common(1)[0]
    before_unscaled_count = n - before_scaled_count
    tvd_threshold_rates = {
        _format_threshold_label(threshold): _safe_div(count, finite_tvd_count)
        for threshold, count in tvd_threshold_counts.items()
    }

    return {
        "file_path": str(path),
        "file_name": path.name,
        "task": metadata["task"],
        "modalities": metadata["modalities"],
        "noisy_modalities": metadata["noisy_modalities"],
        "score_tag": metadata["score_tag"],
        "query_count": n,
        "top1_changed_rate": changed_count / n,
        "top1_before_scaled_rate": before_scaled_count / n,
        "top1_after_scaled_rate": after_scaled_count / n,
        "top1_before_quality_one_rate": before_quality_one_count / n,
        "top1_after_quality_one_rate": after_quality_one_count / n,
        "changed_given_top1_before_scaled": _safe_div(changed_before_scaled_count, before_scaled_count),
        "changed_given_top1_before_unscaled": _safe_div(changed_before_unscaled_count, before_unscaled_count),
        "avg_total_variation_distance": tvd_sum / n,
        "finite_tvd_count": finite_tvd_count,
        "max_total_variation_distance": tvd_max,
        "tvd_gt_threshold_rate": tvd_threshold_rates,
        "dominant_top1_before_token": dominant_before_token,
        "dominant_top1_before_token_rate": dominant_before_count / n,
        "dominant_top1_after_token": dominant_after_token,
        "dominant_top1_after_token_rate": dominant_after_count / n,
        "top5_top1_before_tokens": top1_before_counter.most_common(5),
        "top5_top1_after_tokens": top1_after_counter.most_common(5),
    }


def _write_markdown_report(path: Path, report: dict):
    lines = []
    lines.append("# QAA Score Sensitivity Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- normalize_label_case: `{report['normalize_label_case']}`")
    lines.append(f"- tvd_thresholds: `{report['tvd_thresholds']}`")
    lines.append(f"- prediction_groups: `{len(report['prediction_groups'])}`")
    lines.append(f"- detail_summaries: `{len(report['detail_summaries'])}`")
    lines.append("")

    lines.append("## Prediction Groups")
    lines.append("")
    if not report["prediction_groups"]:
        lines.append("- No prediction groups found.")
        lines.append("")
    for group in report["prediction_groups"]:
        lines.append(f"### {group['group_id']}")
        lines.append("")
        summary = group["metric_summary"]
        lines.append(f"- runs: `{summary['run_count']}`")
        lines.append(f"- common_row_count: `{summary['common_row_count']}`")
        lines.append(
            "- common accuracy range: "
            f"`{summary['accuracy_min_common']:.6f} .. {summary['accuracy_max_common']:.6f}` "
            f"(delta `{summary['accuracy_delta_common']:.6f}`)"
        )
        lines.append(
            "- common macro-F1 range: "
            f"`{summary['macro_f1_min_common']:.6f} .. {summary['macro_f1_max_common']:.6f}` "
            f"(delta `{summary['macro_f1_delta_common']:.6f}`)"
        )
        lines.append(
            "- pairwise disagreement: "
            f"avg `{summary['avg_pairwise_disagreement']:.6f}`, max `{summary['max_pairwise_disagreement']:.6f}`"
        )
        sens = group["score_sensitivity"]
        lines.append(
            "- Pearson(audio_score, common accuracy): "
            f"`{sens['pearson_audio_vs_accuracy_common']:.6f}`"
        )
        lines.append(
            "- Pearson(audio_score, common macro-F1): "
            f"`{sens['pearson_audio_vs_macro_f1_common']:.6f}`"
        )
        lines.append("")
        lines.append("| run | audio_score | video_score | n_common | acc_common | macroF1_common |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for run in group["runs"]:
            metrics_common = run["metrics_common_rows"]
            lines.append(
                f"| `{run['file_name']}` | `{run['audio_score']}` | `{run['video_score']}` | "
                f"`{metrics_common['n']}` | `{metrics_common['accuracy']:.6f}` | `{metrics_common['macro_f1']:.6f}` |"
            )
        lines.append("")

    lines.append("## First-Layer Detail Summaries")
    lines.append("")
    if not report["detail_summaries"]:
        lines.append("- No detail summaries found.")
        lines.append("")
    for detail in report["detail_summaries"]:
        lines.append(f"### {detail['file_name']}")
        lines.append("")
        lines.append(f"- query_count: `{detail['query_count']}`")
        lines.append(f"- top1_changed_rate: `{detail['top1_changed_rate']:.6f}`")
        lines.append(
            "- top1_before_scaled_rate / after_scaled_rate: "
            f"`{detail['top1_before_scaled_rate']:.6f}` / `{detail['top1_after_scaled_rate']:.6f}`"
        )
        lines.append(
            "- top1_before_quality_one_rate / after_quality_one_rate: "
            f"`{detail['top1_before_quality_one_rate']:.6f}` / `{detail['top1_after_quality_one_rate']:.6f}`"
        )
        lines.append(
            "- changed|before_scaled / changed|before_unscaled: "
            f"`{detail['changed_given_top1_before_scaled']:.6f}` / `{detail['changed_given_top1_before_unscaled']:.6f}`"
        )
        lines.append(
            "- avg/max TVD: "
            f"`{detail['avg_total_variation_distance']:.6f}` / `{detail['max_total_variation_distance']:.6f}`"
        )
        lines.append("- TVD overall-change rates (tvd > threshold):")
        for threshold_label, rate in detail["tvd_gt_threshold_rate"].items():
            lines.append(f"  - `>{threshold_label}`: `{rate:.6f}`")
        lines.append(
            "- dominant top1 before token: "
            f"`{detail['dominant_top1_before_token']}` ({detail['dominant_top1_before_token_rate']:.6f})"
        )
        lines.append(
            "- dominant top1 after token: "
            f"`{detail['dominant_top1_after_token']}` ({detail['dominant_top1_after_token_rate']:.6f})"
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    prediction_globs = args.prediction_glob if args.prediction_glob is not None else list(DEFAULT_PREDICTION_GLOBS)
    detail_globs = args.detail_glob if args.detail_glob is not None else list(DEFAULT_DETAIL_GLOBS)
    tvd_thresholds = args.tvd_threshold if args.tvd_threshold is not None else [0.0, 1e-4, 1e-3, 1e-2, 5e-2]
    tvd_thresholds = sorted({float(threshold) for threshold in tvd_thresholds})
    baseline_pattern = args.baseline_pattern.strip() if args.baseline_pattern is not None else ""
    if not baseline_pattern:
        baseline_pattern = None

    prediction_paths = _expand_globs(prediction_globs)
    detail_paths = _expand_globs(detail_globs)

    prediction_groups = {}
    for path in prediction_paths:
        key = _group_key_for_prediction(path)
        prediction_groups.setdefault(key, []).append(path)

    analyzed_prediction_groups = []
    for key in sorted(prediction_groups):
        group_paths = prediction_groups[key]
        if len(group_paths) < 2:
            continue
        analyzed_prediction_groups.append(
            _analyze_prediction_group(
                group_paths,
                normalize_case=args.normalize_label_case,
                min_overlap=args.min_overlap,
                baseline_pattern=baseline_pattern,
            )
        )

    detail_summaries = []
    for path in detail_paths:
        detail_summaries.append(_analyze_detail_csv(path, tvd_thresholds=tvd_thresholds))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalize_label_case": args.normalize_label_case,
        "tvd_thresholds": tvd_thresholds,
        "prediction_globs": prediction_globs,
        "detail_globs": detail_globs,
        "baseline_pattern": baseline_pattern,
        "prediction_files_discovered": [str(path) for path in prediction_paths],
        "detail_files_discovered": [str(path) for path in detail_paths],
        "prediction_groups": analyzed_prediction_groups,
        "detail_summaries": detail_summaries,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, report)

    print(f"[INFO] Prediction CSVs discovered: {len(prediction_paths)}", flush=True)
    print(f"[INFO] Detail CSVs discovered: {len(detail_paths)}", flush=True)
    print(f"[INFO] Prediction groups analyzed: {len(analyzed_prediction_groups)}", flush=True)
    print(f"[INFO] JSON report: {json_path}", flush=True)
    print(f"[INFO] Markdown report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
