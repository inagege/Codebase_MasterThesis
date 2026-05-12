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
    r"^predictions_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z])_(?P<k1>[a-z]+)(?P<v1>\d+)_(?P<k2>[a-z]+)(?P<v2>\d+)\.csv$",
    re.IGNORECASE,
)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
JOIN_COLUMN_PRIORITY = ["split_norm", "sample_id", "file", "dataset"]
MODALITY_ORDER = ["a", "i", "t", "v"]

MODALITY_LABEL = {
    "a": "Audio",
    "i": "Image",
    "t": "Text",
    "v": "Video",
}
MODALITY_SHORT = {
    "a": "A",
    "i": "I",
    "t": "T",
    "v": "V",
}
MODALITY_BY_KEY = {
    "audio": "a",
    "video": "v",
    "image": "i",
    "text": "t",
}
DATASET_LABEL = {
    "imdb": "IMDB",
    "homeprice": "Austin Housing",
    "nejm": "NEJM",
    "marine": "Marine Animals",
    "voxceleb": "VoxCeleb2",
    "sentiment": "MELD Sentiment",
    "emotion": "MELD Emotion",
}
METHOD_ORDER_HINTS = {
    "a": ["bandlimit", "bitcrushing", "clipping", "compress", "jitter", "mp3", "reverb", "snr_white"],
    "i": ["gaussian_noise", "jpeg", "motion_blur", "occlusion", "pixelate", "scale_down", "zoom_blur"],
    "t": ["char_delete", "char_replace", "keyboard", "ocr", "synonym_replace", "top4_paper"],
    "v": ["fps_drop", "gaussian_noise", "motion_blur", "moving_occlusion", "occlusion", "pixelate", "scale_down", "zoom_blur"],
}
FORCED_SUBDIR_CANDIDATES = ["forced_scores", "force_scored"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create forced-quality delta heatmaps per dataset. "
            "Each figure compares selected models side-by-side."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="out",
        help="Root containing model folders (default: out).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model folders under input-root to compare (default: Qwen_3B Qwen_7B).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to plot (default: auto-discover from forced-score folders).",
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
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps/forced_quality_by_dataset",
        help="Output directory for per-dataset PNGs.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="analysis/out/csv/forced_quality_delta_by_dataset_models_s1.csv",
        help="Combined CSV output path.",
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
        return pd.DataFrame()

    join_columns = determine_join_columns(df_baseline=baseline_frame, df_forced=forced_frame)
    if not join_columns:
        return pd.DataFrame()

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
        return pd.DataFrame()
    merged = merged[merged["method_baseline"] == merged["method_forced"]].copy()
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for method, frame in merged.groupby("method_baseline", sort=True):
        y_true = frame["label_norm_baseline"]
        rows.append(
            {
                "method": str(method),
                "paired_n": int(len(frame)),
                "accuracy_baseline": float(frame["is_correct_baseline"].mean() * 100.0),
                "accuracy_forced": float(frame["is_correct_forced"].mean() * 100.0),
                "macro_f1_baseline": float(macro_f1_score(y_true, frame["prediction_norm_baseline"], labels) * 100.0),
                "macro_f1_forced": float(macro_f1_score(y_true, frame["prediction_norm_forced"], labels) * 100.0),
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_quality_value(token: str) -> float:
    text = str(token).strip()
    if not text or not text.isdigit():
        return float("nan")
    scale = 10 ** max(len(text) - 1, 1)
    return int(text) / float(scale)


def format_quality_scalar(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def parse_forced_file(path: Path) -> dict[str, object] | None:
    match = FORCED_FILE_RE.match(path.name)
    if not match:
        return None

    modalities = (match.group("modalities") or "").lower()
    target = (match.group("noise") or "").lower()
    key_1 = (match.group("k1") or "").lower()
    key_2 = (match.group("k2") or "").lower()
    value_1 = parse_quality_value(match.group("v1"))
    value_2 = parse_quality_value(match.group("v2"))

    qualities_by_modality: dict[str, float] = {}
    modality_1 = MODALITY_BY_KEY.get(key_1)
    modality_2 = MODALITY_BY_KEY.get(key_2)
    if modality_1:
        qualities_by_modality[modality_1] = value_1
    if modality_2:
        qualities_by_modality[modality_2] = value_2

    return {
        "modalities": modalities,
        "target": target,
        "qualities_by_modality": qualities_by_modality,
    }


def should_skip_quality_point(dataset: str, qualities_by_modality: dict[str, float]) -> bool:
    # User-requested exclusion for MELD Emotion only.
    if str(dataset).casefold() != "emotion":
        return False
    audio_q = qualities_by_modality.get("a")
    video_q = qualities_by_modality.get("v")
    if audio_q is None or video_q is None:
        return False
    return abs(float(audio_q) - 0.05) < 1e-12 and abs(float(video_q) - 0.05) < 1e-12


def find_forced_dataset_dir(model_root: Path, dataset: str) -> Path | None:
    for subdir in FORCED_SUBDIR_CANDIDATES:
        candidate = model_root / subdir / dataset
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def quality_columns_for_modalities(modalities: str, qualities_by_modality: dict[str, float]) -> tuple[float, float, str] | None:
    if len(modalities) < 2:
        return None
    first = modalities[0]
    second = modalities[1]
    if first not in qualities_by_modality or second not in qualities_by_modality:
        return None
    q_first = float(qualities_by_modality[first])
    q_second = float(qualities_by_modality[second])
    label = (
        f"({MODALITY_SHORT.get(first, first.upper())}={format_quality_scalar(q_first)}, "
        f"{MODALITY_SHORT.get(second, second.upper())}={format_quality_scalar(q_second)})"
    )
    return q_first, q_second, label


def baseline_configs_for_dataset(model_root: Path, dataset: str) -> list[tuple[str, str, Path]]:
    dataset_dir = model_root / dataset
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []

    configs: list[tuple[str, str, Path]] = []
    for path in sorted(dataset_dir.glob("prediction_*_noise_*.csv")):
        match = BASELINE_FILE_RE.match(path.name)
        if not match:
            continue
        modalities = (match.group("modalities") or "").lower()
        target = (match.group("noise") or "").lower()
        if len(modalities) <= 1:
            continue
        if len(target) != 1 or target not in modalities:
            continue
        configs.append((modalities, target, path))
    return configs


def collect_records_for_model_dataset(model_root: Path, model_name: str, dataset: str, severity: int) -> pd.DataFrame:
    forced_dir = find_forced_dataset_dir(model_root=model_root, dataset=dataset)
    if forced_dir is None:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for modalities, target, baseline_path in baseline_configs_for_dataset(model_root=model_root, dataset=dataset):
        forced_paths: list[Path] = []
        for path in sorted(forced_dir.glob(f"predictions_{modalities}_noise_{target}_*.csv")):
            parsed = parse_forced_file(path)
            if parsed is None:
                continue
            if parsed["modalities"] != modalities or parsed["target"] != target:
                continue
            forced_paths.append(path)
        if not forced_paths:
            continue

        labels = collect_label_set([baseline_path, *forced_paths])
        for forced_path in forced_paths:
            parsed = parse_forced_file(forced_path)
            if parsed is None:
                continue
            if should_skip_quality_point(dataset=dataset, qualities_by_modality=parsed["qualities_by_modality"]):  # type: ignore[index]
                continue
            quality_info = quality_columns_for_modalities(modalities, parsed["qualities_by_modality"])  # type: ignore[index]
            if quality_info is None:
                continue
            quality_1, quality_2, quality_label = quality_info
            metrics = compute_overlap_metrics_by_method(
                baseline_path=baseline_path,
                forced_path=forced_path,
                target=target,
                severity=severity,
                labels=labels,
            )
            if metrics.empty:
                continue

            for item in metrics.itertuples(index=False):
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "modalities": modalities,
                        "target": target,
                        "method": str(item.method),
                        "quality_1": float(quality_1),
                        "quality_2": float(quality_2),
                        "quality_label": quality_label,
                        "paired_n": int(item.paired_n),
                        "accuracy_baseline": float(item.accuracy_baseline),
                        "accuracy_forced": float(item.accuracy_forced),
                        "accuracy_delta": float(item.accuracy_forced - item.accuracy_baseline),
                        "macro_f1_baseline": float(item.macro_f1_baseline),
                        "macro_f1_forced": float(item.macro_f1_forced),
                        "macro_f1_delta": float(item.macro_f1_forced - item.macro_f1_baseline),
                    }
                )

    return pd.DataFrame.from_records(rows)


def ordered_methods(methods: list[str], target: str) -> list[str]:
    all_methods = sorted({str(method) for method in methods})
    hint = METHOD_ORDER_HINTS.get(target, [])
    in_hint = [method for method in hint if method in all_methods]
    extra = [method for method in all_methods if method not in in_hint]
    return in_hint + extra


def format_method_label(method: str) -> str:
    label = str(method).replace("_", " ")
    if label == "gaussian noise":
        return "uniform noise"
    if label == "occlusion":
        return "static occlusion"
    return label


def severity_name(severity: int) -> int:
    severity_map = {3: 1, 5: 2}
    return int(severity_map.get(int(severity), int(severity)))


def discover_datasets(input_root: Path, models: list[str]) -> list[str]:
    datasets: set[str] = set()
    for model in models:
        model_root = input_root / model
        for subdir in FORCED_SUBDIR_CANDIDATES:
            forced_root = model_root / subdir
            if not forced_root.exists() or not forced_root.is_dir():
                continue
            for path in forced_root.iterdir():
                if path.is_dir():
                    datasets.add(path.name.casefold())
    return sorted(datasets, key=lambda value: DATASET_LABEL.get(value, value).casefold())


def plot_dataset_figure(
    dataset_records: pd.DataFrame,
    dataset: str,
    models: list[str],
    metric: str,
    severity: int,
    output_path: Path,
    dpi: int,
) -> None:
    if dataset_records.empty:
        return

    model_order = [model for model in models if model in set(dataset_records["model"].astype(str).tolist())]
    if not model_order:
        model_order = [model for model in models if str(model).strip()]
    if not model_order:
        return

    target_order = [modality for modality in MODALITY_ORDER if modality in set(dataset_records["target"].astype(str).tolist())]
    if not target_order:
        return

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
        nrows=len(target_order),
        ncols=len(model_order),
        figsize=(8.0 * len(model_order), 4.6 * len(target_order) + 1.1),
        sharey="row",
        constrained_layout=False,
    )
    if len(target_order) == 1 and len(model_order) == 1:
        axes = np.array([[axes]])
    elif len(target_order) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(model_order) == 1:
        axes = np.expand_dims(axes, axis=1)

    methods_by_target = {
        target: ordered_methods(
            dataset_records.loc[dataset_records["target"] == target, "method"].dropna().astype(str).tolist(),
            target=target,
        )
        for target in target_order
    }

    quality_by_target: dict[str, pd.DataFrame] = {}
    for target in target_order:
        quality_frame = (
            dataset_records.loc[dataset_records["target"] == target, ["quality_label", "quality_1", "quality_2"]]
            .drop_duplicates()
            .sort_values(["quality_1", "quality_2", "quality_label"])
        )
        quality_by_target[target] = quality_frame

    im = None
    metric_col = f"{metric}_delta"
    for row_idx, target in enumerate(target_order):
        methods = methods_by_target[target]
        quality_labels = quality_by_target[target]["quality_label"].tolist()
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            panel_rows = dataset_records[
                (dataset_records["target"] == target) & (dataset_records["model"] == model_name)
            ].copy()

            pivot = (
                panel_rows.pivot_table(index="method", columns="quality_label", values=metric_col, aggfunc="mean")
                .reindex(index=methods, columns=quality_labels)
            )
            if pivot.empty or len(methods) == 0 or len(quality_labels) == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            matrix = pivot.to_numpy(dtype=float)
            masked = np.ma.masked_invalid(matrix)
            im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

            y_labels = [format_method_label(method) for method in pivot.index]
            x_labels = list(pivot.columns)
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=24, ha="right", fontsize=12)
            ax.set_yticks(np.arange(len(y_labels)))
            if col_idx == 0:
                ax.set_yticklabels(y_labels, fontsize=12)
                ax.set_ylabel(MODALITY_LABEL.get(target, target.upper()), fontsize=16)
                ax.yaxis.set_label_coords(-0.30, 0.5)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if row_idx == 0:
                ax.set_title("")

            ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)

            threshold = 45.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if not np.isfinite(value):
                        continue
                    color = "white" if abs(value) >= threshold else "#1f1f1f"
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=12, color=color)

    if im is None:
        return

    dataset_title = DATASET_LABEL.get(dataset, dataset.upper())
    severity_title = severity_name(severity)
    cax = fig.add_axes([0.865, 0.14, 0.009, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Forced Quaity - Baseline (pp)", fontsize=14, labelpad=2)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=12)

    fig.supxlabel("Forced Quality Scores", fontsize=16, y=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.17, top=0.96, wspace=0.12, hspace=0.42)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    models = [str(model).strip() for model in args.models if str(model).strip()]
    if not models:
        raise RuntimeError("No models provided.")

    datasets = (
        [str(dataset).strip().lower() for dataset in args.datasets if str(dataset).strip()]
        if args.datasets
        else discover_datasets(input_root=input_root, models=models)
    )
    if not datasets:
        raise RuntimeError("No datasets found to plot.")

    frames: list[pd.DataFrame] = []
    for model in models:
        model_root = input_root / model
        if not model_root.exists():
            print(f"[WARN] Missing model root: {model_root}")
            continue
        for dataset in datasets:
            frame = collect_records_for_model_dataset(
                model_root=model_root,
                model_name=model,
                dataset=dataset,
                severity=int(args.severity),
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No forced-quality records found for selected models/datasets.")

    records = pd.concat(frames, ignore_index=True)
    records = records.sort_values(
        ["dataset", "target", "model", "method", "quality_1", "quality_2"]
    ).reset_index(drop=True)

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(csv_output, index=False)

    output_dir = Path(args.output_dir)
    metric = str(args.metric)
    severity_display = severity_name(int(args.severity))
    plotted = 0
    for dataset in datasets:
        subset = records[records["dataset"] == dataset].copy()
        if subset.empty:
            print(f"[INFO] Skipping {dataset}: no data.")
            continue
        output_path = output_dir / f"forced_quality_delta_{dataset}_s{severity_display}_{metric}.png"
        plot_dataset_figure(
            dataset_records=subset,
            dataset=dataset,
            models=models,
            metric=metric,
            severity=int(args.severity),
            output_path=output_path,
            dpi=int(args.dpi),
        )
        if output_path.exists():
            plotted += 1
            print(f"[OK] Wrote plot: {output_path}")

    print(f"[OK] Wrote CSV:  {csv_output}")
    print(f"[OK] Plots written: {plotted}")


if __name__ == "__main__":
    main()
