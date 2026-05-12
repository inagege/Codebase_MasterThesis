import argparse
import math
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

BASELINE_FILE_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z])\.csv$", re.IGNORECASE)
FORCED_FILE_RE = re.compile(
    r"^predictions_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z])_audio(?P<audio>\d+)_video(?P<video>\d+)\.csv$",
    re.IGNORECASE,
)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

PANEL_SPECS = [
    ("emotion", "a"),
    ("sentiment", "a"),
    ("emotion", "v"),
    ("sentiment", "v"),
]

PANEL_TITLE = {
    ("emotion", "a"): "Emotion | Noisy Modality: Audio",
    ("sentiment", "a"): "Sentiment | Noisy Modality: Audio",
    ("emotion", "v"): "Emotion | Noisy Modality: Video",
    ("sentiment", "v"): "Sentiment | Noisy Modality: Video",
}

TARGET_MODALITY_LABEL = {
    "a": "Audio",
    "v": "Video",
}

METHOD_ORDER_HINTS = {
    "a": ["bandlimit", "bitcrushing", "clipping", "compress", "jitter", "mp3", "reverb", "snr_white"],
    "v": ["fps_drop", "gaussian_noise", "motion_blur", "moving_occlusion", "occlusion", "pixelate", "scale_down", "zoom_blur"],
}

JOIN_COLUMN_PRIORITY = ["split_norm", "sample_id", "file", "dataset"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2x2 heatmaps for Qwen_7B forced-quality runs (emotion/sentiment x audio/video noise), "
            "showing delta vs baseline."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="out",
        help="Root containing model folders (default: out).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen_7B",
        help="Model folder under input-root (default: Qwen_7B).",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Severity to include (default: 3).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["macro_f1", "accuracy"],
        default="macro_f1",
        help="Metric used for heatmap cell values (default: macro_f1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps/qwen7b_forced_quality_delta_noise_severity1.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="analysis/out/csv/qwen7b_forced_quality_delta_noise_severity1.csv",
        help="Optional CSV output with per-cell values.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Qwen 7B Performance with Forced Quality - Noise Severity 1",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser.parse_args()


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if NUMERIC_RE.match(text):
        try:
            as_float = float(text)
            if math.isfinite(as_float) and as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            pass
    return text.casefold()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def normalize_split_for_join(value: object) -> str:
    split = normalize_text(value)
    if split == "test_all":
        return "all"
    if split.startswith("test_"):
        return split[len("test_") :]
    return split


def parse_split_metadata(split: str) -> dict[str, object]:
    split_text = str(split or "").strip()
    split_lower = split_text.casefold()
    if split_lower in {"all", "dev", "test_all"} or "unmodified" in split_lower:
        return {"is_unmodified": True, "target": "", "method": "unmodified", "severity": None}

    match = SPLIT_NOISE_RE.search(split_text)
    if not match:
        return {"is_unmodified": False, "target": "", "method": "unknown", "severity": None}

    return {
        "is_unmodified": False,
        "target": match.group("target").lower(),
        "method": match.group("method").lower(),
        "severity": int(match.group("severity")),
    }


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"split", "prediction", "label"}.issubset(frame.columns):
        return pd.DataFrame(columns=["split", "prediction_norm", "label_norm", "is_correct"])

    frame = frame.dropna(subset=["split", "prediction", "label"]).copy()
    frame["prediction_norm"] = frame["prediction"].map(normalize_value)
    frame["label_norm"] = frame["label"].map(normalize_value)
    frame = frame[(frame["label_norm"] != "") & (frame["label_norm"] != "unknown")].copy()
    frame["is_correct"] = frame["prediction_norm"].eq(frame["label_norm"]).astype(int)
    return frame


def macro_f1_score(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    if not labels:
        return float("nan")

    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    f1_values: list[float] = []
    for label in labels:
        tp = int(((y_pred == label) & (y_true == label)).sum())
        fp = int(((y_pred == label) & (y_true != label)).sum())
        fn = int(((y_pred != label) & (y_true == label)).sum())
        denom = 2 * tp + fp + fn
        f1_values.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1_values))


def collect_label_set(paths: list[Path]) -> list[str]:
    labels: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["label"])
        for raw_label in frame["label"].tolist():
            label = normalize_value(raw_label)
            if label and label != "unknown":
                labels.add(label)
    return sorted(labels)


def determine_join_columns(df_baseline: pd.DataFrame, df_forced: pd.DataFrame) -> list[str]:
    columns = [col for col in JOIN_COLUMN_PRIORITY if col in df_baseline.columns and col in df_forced.columns]
    if "split_norm" in columns and "sample_id" in columns:
        return ["split_norm", "sample_id"]
    if "split_norm" in columns and "file" in columns:
        return ["split_norm", "file"]
    if "split_norm" in columns:
        return ["split_norm"]
    if "sample_id" in columns:
        return ["sample_id"]
    if "file" in columns:
        return ["file"]
    if "dataset" in columns:
        return ["dataset"]
    return columns[:1]


def deduplicate_by_join_key(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    if not join_columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=join_columns, keep="first")


def prepare_frame_for_pairing(path: Path, target: str, severity: int) -> pd.DataFrame:
    frame = read_prediction_file(path)
    if frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    work["split_norm"] = work["split"].map(normalize_split_for_join)
    split_meta = work["split_norm"].map(parse_split_metadata)
    work["is_unmodified_meta"] = split_meta.map(lambda meta: bool(meta["is_unmodified"]))
    work["target_meta"] = split_meta.map(lambda meta: str(meta["target"]))
    work["method"] = split_meta.map(lambda meta: str(meta["method"]))
    work["severity_meta"] = split_meta.map(lambda meta: meta["severity"])
    work["severity_meta"] = pd.to_numeric(work["severity_meta"], errors="coerce")
    work = work[
        (~work["is_unmodified_meta"])
        & (work["target_meta"] == target)
        & (work["severity_meta"] == int(severity))
    ].copy()
    if work.empty:
        return pd.DataFrame()

    for col in ["sample_id", "file", "dataset"]:
        if col in work.columns:
            work[col] = work[col].map(normalize_text)
    return work


def compute_overlap_metrics_by_method(
    baseline_path: Path,
    forced_path: Path,
    target: str,
    severity: int,
    labels: list[str],
) -> pd.DataFrame:
    baseline_frame = prepare_frame_for_pairing(path=baseline_path, target=target, severity=severity)
    forced_frame = prepare_frame_for_pairing(path=forced_path, target=target, severity=severity)
    if baseline_frame.empty or forced_frame.empty:
        return pd.DataFrame(columns=["method", "paired_n", "accuracy_baseline", "accuracy_forced", "macro_f1_baseline", "macro_f1_forced"])

    join_columns = determine_join_columns(df_baseline=baseline_frame, df_forced=forced_frame)
    if not join_columns:
        return pd.DataFrame(columns=["method", "paired_n", "accuracy_baseline", "accuracy_forced", "macro_f1_baseline", "macro_f1_forced"])

    baseline_subset = baseline_frame[join_columns + ["method", "prediction_norm", "label_norm", "is_correct"]].copy()
    forced_subset = forced_frame[join_columns + ["method", "prediction_norm", "label_norm", "is_correct"]].copy()

    baseline_subset = deduplicate_by_join_key(baseline_subset, join_columns)
    forced_subset = deduplicate_by_join_key(forced_subset, join_columns)

    merged = baseline_subset.merge(
        forced_subset,
        on=join_columns,
        how="inner",
        suffixes=("_baseline", "_forced"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["method", "paired_n", "accuracy_baseline", "accuracy_forced", "macro_f1_baseline", "macro_f1_forced"])

    merged = merged[merged["method_baseline"] == merged["method_forced"]].copy()
    if merged.empty:
        return pd.DataFrame(columns=["method", "paired_n", "accuracy_baseline", "accuracy_forced", "macro_f1_baseline", "macro_f1_forced"])

    records: list[dict[str, object]] = []
    for method, method_frame in merged.groupby("method_baseline", sort=True):
        y_true = method_frame["label_norm_baseline"]
        records.append(
            {
                "method": str(method),
                "paired_n": int(len(method_frame)),
                "accuracy_baseline": float(method_frame["is_correct_baseline"].mean() * 100.0),
                "accuracy_forced": float(method_frame["is_correct_forced"].mean() * 100.0),
                "macro_f1_baseline": float(macro_f1_score(y_true, method_frame["prediction_norm_baseline"], labels) * 100.0),
                "macro_f1_forced": float(macro_f1_score(y_true, method_frame["prediction_norm_forced"], labels) * 100.0),
            }
        )

    if not records:
        return pd.DataFrame(columns=["method", "paired_n", "accuracy_baseline", "accuracy_forced", "macro_f1_baseline", "macro_f1_forced"])
    return pd.DataFrame.from_records(records)


def parse_quality_value(two_digit_token: str) -> float:
    token = str(two_digit_token).strip()
    if not token:
        return float("nan")
    if not token.isdigit():
        return float("nan")
    scale = 10 ** max(len(token) - 1, 1)
    return int(token) / float(scale)


def format_quality_label(audio_quality: float, video_quality: float) -> str:
    audio_text = f"{audio_quality:.3f}".rstrip("0").rstrip(".")
    video_text = f"{video_quality:.3f}".rstrip("0").rstrip(".")
    return f"(A={audio_text}, V={video_text})"


def ordered_methods(methods: list[str], target: str) -> list[str]:
    methods_sorted = sorted({str(method) for method in methods})
    hint = METHOD_ORDER_HINTS.get(target, [])
    in_hint = [method for method in hint if method in methods_sorted]
    extra = [method for method in methods_sorted if method not in in_hint]
    return in_hint + extra


def format_method_label(method: str) -> str:
    label = str(method).replace("_", " ")
    if label == "gaussian noise":
        return "uniform noise"
    if label == "occlusion":
        return "static occlusion"
    return label


def collect_panel_records(model_root: Path, dataset: str, target: str, severity: int) -> pd.DataFrame:
    baseline_path = model_root / dataset / f"prediction_av_noise_{target}.csv"
    forced_dir = model_root / "forced_scores" / dataset
    if not baseline_path.exists() or not forced_dir.exists():
        return pd.DataFrame()

    forced_paths: list[Path] = []
    for path in sorted(forced_dir.glob(f"predictions_av_noise_{target}_audio*_video*.csv")):
        match = FORCED_FILE_RE.match(path.name)
        if not match:
            continue
        if (match.group("modalities") or "").lower() != "av":
            continue
        if (match.group("noise") or "").lower() != target:
            continue
        forced_paths.append(path)

    if not forced_paths:
        return pd.DataFrame()

    labels = collect_label_set([baseline_path, *forced_paths])

    records: list[dict[str, object]] = []
    for forced_path in forced_paths:
        match = FORCED_FILE_RE.match(forced_path.name)
        if not match:
            continue

        audio_quality = parse_quality_value(match.group("audio"))
        video_quality = parse_quality_value(match.group("video"))
        quality_label = format_quality_label(audio_quality=audio_quality, video_quality=video_quality)

        paired_metrics = compute_overlap_metrics_by_method(
            baseline_path=baseline_path,
            forced_path=forced_path,
            target=target,
            severity=severity,
            labels=labels,
        )
        if paired_metrics.empty:
            continue

        for row in paired_metrics.itertuples(index=False):
            records.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "method": row.method,
                    "audio_quality": audio_quality,
                    "video_quality": video_quality,
                    "quality_label": quality_label,
                    "paired_n": int(row.paired_n),
                    "accuracy_forced": float(row.accuracy_forced),
                    "accuracy_baseline": float(row.accuracy_baseline),
                    "accuracy_delta": float(row.accuracy_forced - row.accuracy_baseline),
                    "macro_f1_forced": float(row.macro_f1_forced),
                    "macro_f1_baseline": float(row.macro_f1_baseline),
                    "macro_f1_delta": float(row.macro_f1_forced - row.macro_f1_baseline),
                    "forced_file": forced_path.name,
                }
            )

    return pd.DataFrame.from_records(records)


def panel_pivot(records: pd.DataFrame, metric: str, target: str, methods_order: list[str] | None = None) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    methods = methods_order if methods_order is not None else ordered_methods(records["method"].dropna().astype(str).tolist(), target=target)
    quality_cols = (
        records[["quality_label", "audio_quality", "video_quality"]]
        .drop_duplicates()
        .sort_values(["audio_quality", "video_quality", "quality_label"])
    )
    ordered_quality_labels = quality_cols["quality_label"].tolist()

    pivot = records.pivot_table(index="method", columns="quality_label", values=f"{metric}_delta", aggfunc="mean")
    pivot = pivot.reindex(index=methods, columns=ordered_quality_labels)
    return pivot


def plot_heatmaps(records: pd.DataFrame, metric: str, title: str, output_path: Path, dpi: int) -> None:
    base_cmap = sns.color_palette("coolwarm", as_cmap=True) if sns is not None else plt.get_cmap("coolwarm")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "coolwarm_emphasized_center",
        [
            (0.00, base_cmap(0.00)),
            (0.35, base_cmap(0.18)),
            (0.47, base_cmap(0.35)),
            (0.50, base_cmap(0.50)),
            (0.53, base_cmap(0.65)),
            (0.65, base_cmap(0.82)),
            (1.00, base_cmap(1.00)),
        ],
    ).copy()
    cmap.set_bad("#f2f2f2")
    norm = mcolors.TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.6, 9.8),
        sharey="row",
        constrained_layout=False,
    )
    methods_by_target = {
        target: ordered_methods(
            records.loc[records["target"] == target, "method"].dropna().astype(str).tolist(),
            target=target,
        )
        for target in sorted({str(value) for value in records["target"].dropna().tolist()})
    }

    im = None
    for panel_idx, (dataset, target) in enumerate(PANEL_SPECS):
        row_idx = panel_idx // 2
        col_idx = panel_idx % 2
        ax = axes[row_idx, col_idx]

        panel_rows = records[(records["dataset"] == dataset) & (records["target"] == target)].copy()
        pivot = panel_pivot(panel_rows, metric=metric, target=target, methods_order=methods_by_target.get(target))
        ax.set_title("")

        if pivot.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11)
            continue

        matrix = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

        x_labels = list(pivot.columns)
        y_labels = [format_method_label(method) for method in pivot.index]

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=24, ha="right", fontsize=11)
        ax.set_yticks(np.arange(len(y_labels)))
        if col_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=11)
        else:
            ax.tick_params(axis="y", labelleft=False)

        if col_idx == 0:
            ax.set_ylabel(TARGET_MODALITY_LABEL.get(target, target.upper()), fontsize=15)
            ax.yaxis.set_label_coords(-0.315, 0.5)

        ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)

        text_threshold = 45.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                text_color = "white" if abs(value) >= text_threshold else "#1f1f1f"
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=12, color=text_color)

    if im is None:
        raise RuntimeError("No plottable data found for any panel.")

    fig.suptitle("")
    # Reserve explicit space at the right and place a slim colorbar outside the subplot grid.
    cax = fig.add_axes([0.922, 0.135, 0.008, 0.73])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pertubation-Baseline (pp)", fontsize=14, labelpad=2)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=12)

    fig.supxlabel("Forced Quality Scores", fontsize=15, y=0.035)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.886, bottom=0.155, top=0.900, wspace=0.05, hspace=0.55)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_root = Path(args.input_root) / args.model

    frames: list[pd.DataFrame] = []
    for dataset, target in PANEL_SPECS:
        panel_df = collect_panel_records(model_root=model_root, dataset=dataset, target=target, severity=args.severity)
        if not panel_df.empty:
            frames.append(panel_df)

    if not frames:
        raise RuntimeError(
            f"No records found under {model_root} for forced-quality files. "
            "Check input paths, filenames, and requested severity."
        )

    records = pd.concat(frames, ignore_index=True)
    records = records.sort_values(["dataset", "target", "method", "audio_quality", "video_quality"]).reset_index(drop=True)

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(csv_output, index=False)

    output_path = Path(args.output)
    plot_heatmaps(records=records, metric=args.metric, title=args.title, output_path=output_path, dpi=args.dpi)

    print(f"[OK] Wrote plot: {output_path}")
    print(f"[OK] Wrote CSV:  {csv_output}")


if __name__ == "__main__":
    main()
