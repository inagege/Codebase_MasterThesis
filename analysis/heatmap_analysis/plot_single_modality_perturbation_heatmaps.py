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

PREDICTION_FILE_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

MODALITY_ROWS = ["a", "i", "t", "v"]
MODALITY_LABEL = {"a": "Audio", "i": "Image", "t": "Text", "v": "Video"}

DATASETS_BY_MODALITY = {
    "a": ["sentiment", "emotion", "marine", "voxceleb"],
    "i": ["marine", "nejm", "imdb", "homeprice"],
    "t": ["nejm", "imdb", "homeprice"],
    "v": ["voxceleb", "sentiment", "emotion"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 4x4 heatmap pages for single-modality perturbations. "
            "Rows: modalities (audio/image/text/video), columns: model+severity."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model folders under out/ (default: Qwen_3B Qwen_7B).",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Noise severities shown as columns for each model (default: 3 5).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="out",
        help="Root directory containing out/<model>/<dataset>/prediction_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps",
        help="Directory where heatmaps are saved.",
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


def collect_label_sets(input_root: Path, models: list[str]) -> dict[tuple[str, str], list[str]]:
    labels_by_key: dict[tuple[str, str], set[str]] = {}
    for model in models:
        model_root = input_root / model
        if not model_root.exists():
            continue
        for dataset_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            dataset = dataset_dir.name.lower()
            key = (model, dataset)
            labels_by_key.setdefault(key, set())
            for path in dataset_dir.glob("prediction_*.csv"):
                match = PREDICTION_FILE_RE.match(path.name)
                if not match:
                    continue
                modalities = (match.group("modalities") or "").lower()
                if len(modalities) != 1:
                    continue
                frame = read_prediction_file(path)
                labels_by_key[key].update(frame["label_norm"].tolist())
    return {key: sorted(values) for key, values in labels_by_key.items()}


def compute_single_modality_records(
    input_root: Path,
    models: list[str],
    labels_by_key: dict[tuple[str, str], list[str]],
) -> pd.DataFrame:
    baselines: dict[tuple[str, str, str], dict[str, float]] = {}
    records: list[dict[str, object]] = []

    for model in models:
        model_root = input_root / model
        if not model_root.exists():
            continue

        for dataset_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            dataset = dataset_dir.name.lower()
            labels = labels_by_key.get((model, dataset), [])

            file_entries: list[tuple[Path, str, str]] = []
            for path in dataset_dir.glob("prediction_*.csv"):
                match = PREDICTION_FILE_RE.match(path.name)
                if not match:
                    continue
                modalities = (match.group("modalities") or "").lower()
                noise_modalities = (match.group("noise") or "").lower()
                if len(modalities) != 1:
                    continue
                file_entries.append((path, modalities, noise_modalities))

            # Pass 1: collect all baselines first so perturbation files can always reference them.
            for path, modality, noise_modalities in sorted(file_entries, key=lambda entry: entry[0].name):
                if noise_modalities != "":
                    continue

                frame = read_prediction_file(path)
                if frame.empty:
                    continue

                baseline_accuracy = float(frame["is_correct"].mean() * 100.0)
                baseline_macro_f1 = (
                    float("nan")
                    if dataset == "nejm"
                    else float(macro_f1_score(frame["label_norm"], frame["prediction_norm"], labels) * 100.0)
                )
                baselines[(model, dataset, modality)] = {
                    "baseline_accuracy": baseline_accuracy,
                    "baseline_macro_f1": baseline_macro_f1,
                }

            # Pass 2: process perturbation files against stored baselines.
            for path, modality, noise_modalities in sorted(file_entries, key=lambda entry: entry[0].name):
                if noise_modalities != modality:
                    continue

                baseline = baselines.get((model, dataset, modality))
                if baseline is None:
                    continue

                frame = read_prediction_file(path)
                if frame.empty:
                    continue

                for split, split_frame in frame.groupby("split", sort=True):
                    split_meta = parse_split_metadata(split)
                    if split_meta["is_unmodified"]:
                        continue
                    if split_meta["target"] != modality:
                        continue

                    accuracy = float(split_frame["is_correct"].mean() * 100.0)
                    macro_f1 = (
                        float("nan")
                        if dataset == "nejm"
                        else float(macro_f1_score(split_frame["label_norm"], split_frame["prediction_norm"], labels) * 100.0)
                    )

                    records.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "modality": modality,
                            "target": split_meta["target"],
                            "method": split_meta["method"],
                            "severity": split_meta["severity"],
                            "split": split,
                            "accuracy": accuracy,
                            "baseline_accuracy": baseline["baseline_accuracy"],
                            "delta_accuracy": accuracy - baseline["baseline_accuracy"],
                            "macro_f1": macro_f1,
                            "baseline_macro_f1": baseline["baseline_macro_f1"],
                            "delta_macro_f1": macro_f1 - baseline["baseline_macro_f1"]
                            if not (pd.isna(macro_f1) or pd.isna(baseline["baseline_macro_f1"]))
                            else float("nan"),
                        }
                    )

    return pd.DataFrame.from_records(records)


def ordered_methods(records: pd.DataFrame, modality: str) -> list[str]:
    methods = sorted(records.loc[records["modality"] == modality, "method"].dropna().astype(str).unique().tolist())
    hint = METHOD_ORDER_HINTS.get(modality, [])
    in_hint = [m for m in hint if m in methods]
    extra = [m for m in methods if m not in in_hint]
    return in_hint + extra


def dataset_order_for_modality(modality: str, include_nejm: bool) -> list[str]:
    datasets = list(DATASETS_BY_MODALITY[modality])
    if not include_nejm:
        datasets = [dataset for dataset in datasets if dataset != "nejm"]
    return sorted(datasets, key=lambda dataset: DATASET_LABEL.get(dataset, dataset).casefold())


def format_method_tick_label(method: str) -> str:
    label = str(method).replace("_", " ")
    if label == "gaussian noise":
        return "uniform noise"
    if label == "occlusion":
        return "static occlusion"
    return label


def format_column_header(model: str, severity: int) -> str:
    model_label = str(model).replace("_", " ")
    severity_map = {1: 1, 2: 2, 3: 1, 5: 2}
    severity_idx = severity_map.get(int(severity))
    if severity_idx is None:
        return f"{model_label}\nNoise Severity {severity}"
    return f"{model_label}\nNoise Severity {severity_idx}"


def plot_metric_page(
    records: pd.DataFrame,
    models: list[str],
    severities: list[int],
    metric_col: str,
    delta_col: str,
    output_path: Path,
    include_nejm: bool,
    dpi: int,
) -> None:
    if records.empty:
        return

    column_specs = [(model, severity) for model in models for severity in severities]
    row_modalities = MODALITY_ROWS

    figure_records = records.copy()
    if not include_nejm:
        figure_records = figure_records[figure_records["dataset"] != "nejm"].copy()

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

    fig, axes = plt.subplots(
        nrows=len(row_modalities),
        ncols=len(column_specs),
        figsize=(2.4 * len(column_specs) + 2.4, 3.9 * len(row_modalities)),
        sharey="row",
        constrained_layout=False,
    )

    for row_idx, modality in enumerate(row_modalities):
        methods = ordered_methods(figure_records, modality)
        datasets = dataset_order_for_modality(modality, include_nejm=include_nejm)

        for col_idx, (model, severity) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            subset = figure_records[
                (figure_records["model"] == model)
                & (figure_records["modality"] == modality)
                & (figure_records["severity"] == severity)
            ]

            metric_matrix = (
                subset.pivot_table(index="dataset", columns="method", values=metric_col, aggfunc="mean")
                .reindex(index=datasets, columns=methods)
            )
            delta_matrix = (
                subset.pivot_table(index="dataset", columns="method", values=delta_col, aggfunc="mean")
                .reindex(index=datasets, columns=methods)
            )

            if metric_matrix.empty:
                metric_matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)
            if delta_matrix.empty:
                delta_matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)

            annotations = metric_matrix.copy()
            annotations = annotations.map(lambda value: "" if pd.isna(value) else f"{value:.1f}")

            if len(datasets) == 0 or len(methods) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            plot_values = delta_matrix.to_numpy(dtype=float)
            mask = np.isnan(plot_values)
            plot_values = np.where(mask, np.nan, plot_values)

            # Switch axes: x = datasets, y = perturbation methods.
            ax.imshow(plot_values.T, cmap=cmap, norm=norm, aspect="auto")

            # Draw light cell borders for readability.
            for y in range(len(methods) + 1):
                ax.axhline(y - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)
            for x in range(len(datasets) + 1):
                ax.axvline(x - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)

            for y_idx, method in enumerate(methods):
                for x_idx, dataset_name in enumerate(datasets):
                    text = annotations.at[dataset_name, method] if method in annotations.columns else ""
                    if text:
                        ax.text(
                            x_idx,
                            y_idx,
                            text,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="black",
                        )

            if row_idx == 0:
                ax.set_title("")
            else:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODALITY_LABEL[modality]}",
                    fontsize=15,
                    rotation=90,
                    ha="center",
                    va="center",
                )
                # Align all row labels at the same horizontal anchor.
                ax.yaxis.set_label_coords(-0.72, 0.5)
            else:
                ax.set_ylabel("")

            dataset_tick_labels = [DATASET_LABEL.get(value, value) for value in datasets]
            ax.set_xticks(np.arange(len(datasets)))
            ax.set_xticklabels(dataset_tick_labels, rotation=30, ha="right", fontsize=11)
            ax.tick_params(axis="x", pad=1)
            method_tick_labels = [format_method_tick_label(method) for method in methods]
            ax.set_yticks(np.arange(len(methods)))
            if col_idx == 0:
                ax.set_yticklabels(method_tick_labels, rotation=0, fontsize=11)
                ax.tick_params(axis="y", labelleft=True, left=True)
            else:
                # With shared y-axes, hide labels per-axis without clearing shared labels.
                ax.tick_params(axis="y", labelleft=False, left=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.926, 0.135, 0.008, 0.73])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Pertubation-Baseline (pp)", fontsize=14, labelpad=2)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=12)

    metric_title = "Accuracy" if metric_col == "accuracy" else "Macro F1"
    fig.suptitle("")
    fig.supxlabel("Dataset", fontsize=15, y=0.04)
    fig.subplots_adjust(left=0.12, right=0.882, top=0.900, bottom=0.120, wspace=0.08, hspace=0.39)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    labels_by_key = collect_label_sets(input_root=input_root, models=args.models)
    records = compute_single_modality_records(
        input_root=input_root,
        models=args.models,
        labels_by_key=labels_by_key,
    )

    if records.empty:
        raise RuntimeError("No single-modality perturbation records found.")

    accuracy_path = output_dir / "single_modality_heatmaps_accuracy.png"
    macro_f1_path = output_dir / "single_modality_heatmaps_macro_f1.png"

    plot_metric_page(
        records=records,
        models=args.models,
        severities=args.severities,
        metric_col="accuracy",
        delta_col="delta_accuracy",
        output_path=accuracy_path,
        include_nejm=True,
        dpi=args.dpi,
    )
    plot_metric_page(
        records=records,
        models=args.models,
        severities=args.severities,
        metric_col="macro_f1",
        delta_col="delta_macro_f1",
        output_path=macro_f1_path,
        include_nejm=False,
        dpi=args.dpi,
    )

    print(f"Wrote accuracy heatmap page to {accuracy_path}")
    print(f"Wrote macro-F1 heatmap page to {macro_f1_path}")


if __name__ == "__main__":
    main()
