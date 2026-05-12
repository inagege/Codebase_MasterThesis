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

BASELINE_FILE_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
SEEDED_FILE_RE = re.compile(
    r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*).+_rand_seed_(?P<seed>\d+)\.csv$",
    re.IGNORECASE,
)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
AGG_KEYS = ["dataset", "modalities", "noise_modalities", "target", "method", "severity"]
MODALITY_ORDER = ["a", "i", "t", "v"]

DATASET_LABEL = {
    "imdb": "IMDB",
    "homeprice": "Austin Housing",
    "nejm": "NEJM",
    "marine": "Marine Animals",
    "voxceleb": "VoxCeleb2",
    "sentiment": "MELD Sentiment",
    "emotion": "MELD Emotion",
}
MODALITY_LABEL = {"a": "Audio", "i": "Image", "t": "Text", "v": "Video"}
METHOD_ORDER_HINTS = {
    "a": ["bandlimit", "bitcrushing", "clipping", "compress", "jitter", "mp3", "reverb", "snr_white"],
    "i": ["gaussian_noise", "jpeg", "motion_blur", "occlusion", "pixelate", "scale_down", "zoom_blur"],
    "t": ["char_delete", "char_replace", "keyboard", "ocr", "synonym_replace", "top4_paper"],
    "v": ["fps_drop", "gaussian_noise", "motion_blur", "moving_occlusion", "occlusion", "pixelate", "scale_down", "zoom_blur"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create placebo_rescaled rand-seed vs baseline heatmaps with model columns "
            "(e.g., Qwen 3B vs Qwen 7B) at a selected random seed."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model folders under out/ to plot (default: Qwen_3B Qwen_7B).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="placebo_rescaled",
        help="Source folder under out/<model>/ containing rand-seed predictions (default: placebo_rescaled).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include. If omitted, infer all datasets with seeded files.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Severity to include (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to plot across all selected models (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps/placebo_rescaled_seeds",
        help="Directory where heatmaps are saved.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="analysis/out/csv/placebo_rescaled_seed_vs_baseline_sentiment_emotion_s3.csv",
        help="CSV path for merged seed-vs-baseline records.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI.",
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


def has_prediction_files(directory: Path, seeded_only: bool = False) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    if seeded_only:
        return any(SEEDED_FILE_RE.match(path.name) for path in directory.glob("prediction_*.csv"))
    return any(BASELINE_FILE_RE.match(path.name) or SEEDED_FILE_RE.match(path.name) for path in directory.glob("prediction_*.csv"))


def collect_label_set_for_dir(directory: Path, seeded_only: bool = False) -> set[str]:
    labels: set[str] = set()
    if not has_prediction_files(directory=directory, seeded_only=seeded_only):
        return labels

    for path in sorted(directory.glob("prediction_*.csv")):
        match = SEEDED_FILE_RE.match(path.name) if seeded_only else (BASELINE_FILE_RE.match(path.name) or SEEDED_FILE_RE.match(path.name))
        if not match:
            continue
        modalities = (match.group("modalities") or "").lower()
        if len(modalities) <= 1:
            continue
        frame = read_prediction_file(path)
        labels.update(frame["label_norm"].tolist())
    return labels


def parse_baseline_filename(path: Path) -> tuple[str, str] | None:
    match = BASELINE_FILE_RE.match(path.name)
    if not match:
        return None
    modalities = (match.group("modalities") or "").lower()
    noise_modalities = (match.group("noise") or "").lower()
    return modalities, noise_modalities


def parse_seeded_filename(path: Path) -> tuple[str, str, int] | None:
    match = SEEDED_FILE_RE.match(path.name)
    if not match:
        return None
    modalities = (match.group("modalities") or "").lower()
    noise_modalities = (match.group("noise") or "").lower()
    seed = int(match.group("seed"))
    return modalities, noise_modalities, seed


def collect_metrics_for_file(
    dataset: str,
    path: Path,
    labels: list[str],
    seed: int | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    frame = read_prediction_file(path)
    if frame.empty:
        return pd.DataFrame.from_records(records)

    parsed = parse_seeded_filename(path) if seed is not None else parse_baseline_filename(path)
    if parsed is None:
        return pd.DataFrame.from_records(records)

    if seed is None:
        modalities, noise_modalities = parsed
    else:
        modalities, noise_modalities, _ = parsed

    if not modalities or len(modalities) <= 1 or len(noise_modalities) != 1 or noise_modalities not in modalities:
        return pd.DataFrame.from_records(records)

    for split, split_frame in frame.groupby("split", sort=True):
        split_meta = parse_split_metadata(split)
        if split_meta["is_unmodified"] or split_meta["target"] != noise_modalities or split_meta["severity"] is None:
            continue

        records.append(
            {
                "dataset": dataset,
                "modalities": "".join(sorted(modalities)),
                "noise_modalities": noise_modalities,
                "target": split_meta["target"],
                "method": split_meta["method"],
                "severity": int(split_meta["severity"]),
                "accuracy": float(split_frame["is_correct"].mean() * 100.0),
                "macro_f1": (
                    float("nan")
                    if str(dataset).casefold() == "nejm"
                    else float(macro_f1_score(split_frame["label_norm"], split_frame["prediction_norm"], labels) * 100.0)
                ),
                "seed": seed,
            }
        )
    return pd.DataFrame.from_records(records)


def collect_baseline_records(model_root: Path, datasets: list[str]) -> tuple[pd.DataFrame, list[str], dict[str, set[str]]]:
    notes: list[str] = []
    labels_by_dataset: dict[str, set[str]] = {}
    frames: list[pd.DataFrame] = []

    for dataset in datasets:
        dataset_dir = model_root / dataset
        if not has_prediction_files(dataset_dir):
            notes.append(f"[WARN] Missing baseline dataset directory: {dataset_dir}")
            continue

        labels = collect_label_set_for_dir(dataset_dir, seeded_only=False)
        labels_by_dataset[dataset] = labels
        label_list = sorted(labels)

        dataset_frames: list[pd.DataFrame] = []
        for path in sorted(dataset_dir.glob("prediction_*.csv")):
            if parse_baseline_filename(path) is None:
                continue
            file_frame = collect_metrics_for_file(dataset=dataset, path=path, labels=label_list, seed=None)
            if not file_frame.empty:
                dataset_frames.append(file_frame)
        if dataset_frames:
            frames.append(pd.concat(dataset_frames, ignore_index=True))
        else:
            notes.append(f"[WARN] No baseline multimodal rows for {dataset_dir}")

    if not frames:
        return pd.DataFrame(), notes, labels_by_dataset

    baseline_records = pd.concat(frames, ignore_index=True)
    baseline_agg = (
        baseline_records.groupby(AGG_KEYS, as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_baseline", "macro_f1": "macro_f1_baseline"})
    )
    return baseline_agg, notes, labels_by_dataset


def collect_seeded_records(
    model_root: Path,
    source: str,
    datasets: list[str],
    baseline_labels_by_dataset: dict[str, set[str]],
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    source_root = model_root / source
    if not source_root.exists():
        notes.append(f"[WARN] Missing source root: {source_root}")
        return pd.DataFrame(), notes

    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        dataset_dir = source_root / dataset
        if not has_prediction_files(dataset_dir, seeded_only=True):
            notes.append(f"[WARN] Missing seeded prediction files in: {dataset_dir}")
            continue

        labels = sorted(collect_label_set_for_dir(dataset_dir, seeded_only=True) | baseline_labels_by_dataset.get(dataset, set()))
        dataset_frames: list[pd.DataFrame] = []
        for path in sorted(dataset_dir.glob("prediction_*.csv")):
            parsed = parse_seeded_filename(path)
            if parsed is None:
                continue
            seed = int(parsed[2])
            file_frame = collect_metrics_for_file(dataset=dataset, path=path, labels=labels, seed=seed)
            if not file_frame.empty:
                dataset_frames.append(file_frame)
        if dataset_frames:
            frames.append(pd.concat(dataset_frames, ignore_index=True))
        else:
            notes.append(f"[WARN] No seeded multimodal rows for {dataset_dir}")

    if not frames:
        return pd.DataFrame(), notes

    seeded_records = pd.concat(frames, ignore_index=True)
    seeded_agg = (
        seeded_records.groupby(["seed", *AGG_KEYS], as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_source", "macro_f1": "macro_f1_source"})
    )
    return seeded_agg, notes


def ordered_methods(records: pd.DataFrame, modality: str) -> list[str]:
    methods = sorted(records.loc[records["target"] == modality, "method"].dropna().astype(str).unique().tolist())
    hints = METHOD_ORDER_HINTS.get(modality, [])
    in_hint = [method for method in hints if method in methods]
    extras = [method for method in methods if method not in in_hint]
    return in_hint + extras


def format_method_tick_label(method: str) -> str:
    label = str(method).replace("_", " ")
    if label == "gaussian noise":
        return "uniform noise"
    if label == "occlusion":
        return "static occlusion"
    return label


def severity_name(severity: int) -> int:
    severity_map = {3: 1, 5: 2}
    return int(severity_map.get(int(severity), int(severity)))


def source_display_label(source: str) -> str:
    source_norm = str(source).strip().casefold()
    if source_norm == "placebo_rescaled":
        return "Placebo rescaled"
    if source_norm == "placebo":
        return "Placebo"
    return str(source).replace("_", " ")


def resolve_dataset_order(records: pd.DataFrame, preferred: list[str] | None) -> list[str]:
    available = sorted(records["dataset"].dropna().astype(str).unique().tolist())
    if preferred:
        preferred_norm = [str(dataset).strip().lower() for dataset in preferred if str(dataset).strip()]
        chosen = [dataset for dataset in preferred_norm if dataset in set(available)]
        extras = [dataset for dataset in available if dataset not in set(chosen)]
        return chosen + extras
    return sorted(available, key=lambda dataset: DATASET_LABEL.get(dataset, dataset).casefold())


def modality_order_for_records(records: pd.DataFrame) -> list[str]:
    available_targets = [str(value) for value in records["target"].dropna().astype(str).unique().tolist()]
    ordered = [modality for modality in MODALITY_ORDER if modality in set(available_targets)]
    ordered.extend(sorted([modality for modality in available_targets if modality not in set(MODALITY_ORDER)]))
    return ordered


def dataset_order_for_modality(records: pd.DataFrame, modality: str, preferred: list[str] | None) -> list[str]:
    modality_records = records[records["target"] == modality].copy()
    available = sorted(modality_records["dataset"].dropna().astype(str).unique().tolist())
    if preferred:
        preferred_norm = [str(dataset).strip().lower() for dataset in preferred if str(dataset).strip()]
        chosen = [dataset for dataset in preferred_norm if dataset in set(available)]
        extras = [dataset for dataset in available if dataset not in set(chosen)]
        return chosen + extras
    return sorted(available, key=lambda dataset: DATASET_LABEL.get(dataset, dataset).casefold())


def dataset_target_columns(records: pd.DataFrame, dataset_order: list[str], modality_order: list[str]) -> list[tuple[str, str]]:
    available_pairs = {
        (str(row["dataset"]), str(row["target"]))
        for _, row in records[["dataset", "target"]].dropna().iterrows()
    }
    ordered: list[tuple[str, str]] = []
    for dataset in dataset_order:
        for modality in modality_order:
            if (dataset, modality) in available_pairs:
                ordered.append((dataset, modality))
    return ordered


def dataset_target_key(dataset: str, target: str) -> str:
    return f"{dataset}::{target}"


def row_key(target: str, method: str) -> str:
    return f"{target}::{method}"


def plot_seed_comparison_heatmap(
    records: pd.DataFrame,
    models: list[str],
    source: str,
    datasets: list[str] | None,
    seed: int,
    severity: int,
    metric_col: str,
    output_path: Path,
    dpi: int,
) -> None:
    if records.empty or not models:
        return

    model_order = [str(model).strip() for model in models if str(model).strip()]
    if not model_order:
        return

    figure_records = records[
        (records["seed"] == int(seed))
        & (records["severity"] == int(severity))
        & (records["model"].isin(model_order))
    ].copy()
    if figure_records.empty:
        return

    row_modalities = modality_order_for_records(figure_records)
    if not row_modalities:
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
    )
    norm = mcolors.TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)

    datasets_by_target = {
        modality: dataset_order_for_modality(figure_records, modality, datasets) for modality in row_modalities
    }
    if metric_col == "delta_macro_f1":
        datasets_by_target = {
            modality: [dataset for dataset in modality_datasets if str(dataset).casefold() != "nejm"]
            for modality, modality_datasets in datasets_by_target.items()
        }
    row_modalities = [modality for modality in row_modalities if len(datasets_by_target.get(modality, [])) > 0]
    if not row_modalities:
        return

    max_datasets_per_row = max((len(values) for values in datasets_by_target.values()), default=0)
    per_panel_width = max(4.8, 0.85 * max_datasets_per_row)
    fig_width = per_panel_width * len(model_order) + 2.7
    fig_height = max(5.2, 2.9 * len(row_modalities) + 1.8)
    fig, axes = plt.subplots(
        nrows=len(row_modalities),
        ncols=len(model_order),
        figsize=(fig_width, fig_height),
        sharey="row",
        constrained_layout=False,
    )
    if len(row_modalities) == 1 and len(model_order) == 1:
        axes = np.array([[axes]])
    elif len(row_modalities) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(model_order) == 1:
        axes = np.expand_dims(axes, axis=1)

    methods_by_target = {modality: ordered_methods(figure_records, modality) for modality in row_modalities}

    for row_idx, modality in enumerate(row_modalities):
        methods = methods_by_target.get(modality, [])
        row_datasets = datasets_by_target.get(modality, [])
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            subset = figure_records[
                (figure_records["target"] == modality)
                & (figure_records["model"] == model_name)
            ]
            matrix = (
                subset.pivot_table(index="dataset", columns="method", values=metric_col, aggfunc="mean")
                .reindex(index=row_datasets, columns=methods)
            )
            if matrix.empty:
                matrix = pd.DataFrame(index=row_datasets, columns=methods, dtype=float)

            if len(row_datasets) == 0 or len(methods) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            values = matrix.to_numpy(dtype=float)
            ax.imshow(values.T, cmap=cmap, norm=norm, aspect="auto")

            for y in range(len(methods) + 1):
                ax.axhline(y - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)
            for x in range(len(row_datasets) + 1):
                ax.axvline(x - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)

            for y_idx, method in enumerate(methods):
                for x_idx, dataset_name in enumerate(row_datasets):
                    value = matrix.at[dataset_name, method]
                    if pd.notna(value):
                        ax.text(x_idx, y_idx, f"{value:+.1f}", ha="center", va="center", fontsize=10.5, color="black")

            if row_idx == 0:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(MODALITY_LABEL.get(modality, modality.upper()), fontsize=13)
                ax.yaxis.set_label_coords(-0.62, 0.5)
            else:
                ax.set_ylabel("")

            x_tick_labels = [DATASET_LABEL.get(dataset, dataset) for dataset in row_datasets]
            ax.set_xticks(np.arange(len(row_datasets)))
            ax.set_xticklabels(x_tick_labels, rotation=28, ha="right", fontsize=9.8)

            y_tick_labels = [format_method_tick_label(method) for method in methods]
            ax.set_yticks(np.arange(len(methods)))
            if col_idx == 0:
                ax.set_yticklabels(y_tick_labels, fontsize=10.0)
            else:
                ax.tick_params(axis="y", labelleft=False, left=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.15, 0.009, 0.72])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Source - Baseline (pp)", fontsize=12, labelpad=2)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=10)

    metric_label = "Accuracy" if metric_col == "delta_accuracy" else "Macro F1"
    fig.suptitle("")
    fig.supxlabel("Dataset", fontsize=12, y=0.05)
    fig.subplots_adjust(left=0.16, right=0.90, top=0.90, bottom=0.20, wspace=0.08, hspace=0.48)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_csv(records: pd.DataFrame, csv_output: Path) -> Path:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "model",
        "source",
        "seed",
        "dataset",
        "modalities",
        "noise_modalities",
        "target",
        "method",
        "severity",
        "accuracy_baseline",
        "accuracy_source",
        "delta_accuracy",
        "macro_f1_baseline",
        "macro_f1_source",
        "delta_macro_f1",
    ]
    available_cols = [col for col in export_cols if col in records.columns]
    export = records[available_cols].copy()
    export["dataset_label"] = export["dataset"].map(lambda value: DATASET_LABEL.get(value, value))
    export["target_label"] = export["target"].map(lambda value: MODALITY_LABEL.get(value, value))
    export["method_label"] = export["method"].map(format_method_tick_label)
    export.to_csv(csv_output, index=False)
    return csv_output


def discover_seeded_datasets(model_root: Path, source: str) -> list[str]:
    source_root = model_root / source
    if not source_root.exists() or not source_root.is_dir():
        return []

    discovered: list[str] = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir():
            continue
        if has_prediction_files(path, seeded_only=True):
            discovered.append(path.name.casefold())
    return discovered


def main() -> None:
    args = parse_args()
    requested_datasets = (
        [str(dataset).strip().lower() for dataset in args.datasets if str(dataset).strip()]
        if args.datasets
        else None
    )

    collected_records: list[pd.DataFrame] = []
    processed_models = 0
    output_dir = Path(args.output_dir)
    source_slug = args.source.lower()
    severity_display = severity_name(int(args.severity))

    for model in args.models:
        model_name = str(model).strip()
        if not model_name:
            continue
        model_root = Path("../out") / model_name
        if not model_root.exists():
            print(f"[WARN] Skipping {model_name}: missing model root {model_root}")
            continue

        model_datasets = requested_datasets if requested_datasets else discover_seeded_datasets(model_root, args.source)
        if not model_datasets:
            print(f"[WARN] Skipping {model_name}: no datasets found for source '{args.source}'.")
            continue

        baseline_agg, baseline_notes, baseline_labels = collect_baseline_records(
            model_root=model_root,
            datasets=model_datasets,
        )
        source_agg, source_notes = collect_seeded_records(
            model_root=model_root,
            source=args.source,
            datasets=model_datasets,
            baseline_labels_by_dataset=baseline_labels,
        )
        for note in baseline_notes + source_notes:
            print(note)

        if baseline_agg.empty:
            print(f"[WARN] Skipping {model_name}: no baseline records found.")
            continue
        if source_agg.empty:
            print(f"[WARN] Skipping {model_name}: no seeded source records found.")
            continue

        records = source_agg.merge(baseline_agg, on=AGG_KEYS, how="inner")
        if records.empty:
            print(f"[WARN] Skipping {model_name}: no overlapping perturbation groups.")
            continue

        records["delta_accuracy"] = records["accuracy_source"] - records["accuracy_baseline"]
        records["delta_macro_f1"] = records["macro_f1_source"] - records["macro_f1_baseline"]
        records["model"] = model_name
        records["source"] = args.source
        records = records[records["severity"] == int(args.severity)].copy()
        if records.empty:
            print(f"[WARN] Skipping {model_name}: no records at severity {args.severity}.")
            continue

        available_seeds = sorted(records["seed"].dropna().astype(int).unique().tolist())
        if int(args.seed) not in set(available_seeds):
            print(f"[WARN] {model_name}: seed {args.seed} not found. Available seeds: {available_seeds}")
        collected_records.append(records.copy())
        processed_models += 1

    if processed_models == 0 or not collected_records:
        raise RuntimeError("No model produced plottable records. Check model names, datasets, source, and seeds.")

    combined_records = pd.concat(collected_records, ignore_index=True)
    combined_records = combined_records[combined_records["severity"] == int(args.severity)].copy()
    if combined_records.empty:
        raise RuntimeError(f"No records found at severity {args.severity}.")
    seed_records = combined_records[
        (combined_records["seed"] == int(args.seed))
        & (combined_records["model"].isin([str(model).strip() for model in args.models if str(model).strip()]))
    ].copy()
    if seed_records.empty:
        raise RuntimeError(
            f"No records found for seed {args.seed} at severity {args.severity} for models {args.models}."
        )

    datasets_used = resolve_dataset_order(seed_records, requested_datasets)
    dataset_slug = "_".join(datasets_used) if datasets_used else "all"
    models_slug = "_".join(str(model).lower().replace("_", "") for model in args.models if str(model).strip())
    seed_slug = str(int(args.seed))
    accuracy_path = (
        output_dir
        / f"{models_slug}_{source_slug}_seed_{seed_slug}_{dataset_slug}_s{severity_display}_accuracy.png"
    )
    macro_path = (
        output_dir
        / f"{models_slug}_{source_slug}_seed_{seed_slug}_{dataset_slug}_s{severity_display}_macro_f1.png"
    )
    plot_seed_comparison_heatmap(
        records=combined_records,
        models=args.models,
        source=args.source,
        datasets=requested_datasets,
        seed=int(args.seed),
        severity=int(args.severity),
        metric_col="delta_accuracy",
        output_path=accuracy_path,
        dpi=args.dpi,
    )
    plot_seed_comparison_heatmap(
        records=combined_records,
        models=args.models,
        source=args.source,
        datasets=requested_datasets,
        seed=int(args.seed),
        severity=int(args.severity),
        metric_col="delta_macro_f1",
        output_path=macro_path,
        dpi=args.dpi,
    )
    print(f"Wrote model-comparison heatmap (Accuracy): {accuracy_path}")
    print(f"Wrote model-comparison heatmap (Macro-F1): {macro_path}")

    csv_path = write_csv(records=combined_records, csv_output=Path(args.csv_output))
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
