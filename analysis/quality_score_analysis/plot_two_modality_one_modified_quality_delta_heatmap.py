import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITY_FILE_RE = re.compile(r"^quality_scores_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)")

MODALITY_ROWS = ["a", "i", "t", "v"]
MODALITY_LABEL = {"a": "Audio", "i": "Image", "t": "Text", "v": "Video"}
MODALITY_SHORT = {"a": "A", "i": "I", "t": "T", "v": "V"}

QUALITY_COL_BY_MODALITY = {
    "a": "audio_raw_quality",
    "i": "image_raw_quality",
    "t": "text_raw_quality",
    "v": "video_raw_quality",
}

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
    "a": ["bandlimit", "bitcrushing", "compress", "jitter", "mp3", "reverb", "snr_white"],
    "i": ["gaussian_noise", "jpeg", "motion_blur", "occlusion", "pixelate", "scale_down", "zoom_blur"],
    "t": ["char_delete", "char_replace", "keyboard", "ocr", "synonym_replace", "top4_paper"],
    "v": ["fps_drop", "gaussian_noise", "motion_blur", "moving_occlusion", "occlusion", "pixelate", "scale_down", "zoom_blur"],
}

EXCLUDED_METHODS_BY_MODALITY = {
    "a": {"clipping"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create heatmaps for two-modality settings with one perturbed modality, "
            "showing only the perturbed modality quality score."
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
        help="Root directory containing out/<model>/qwen_scored/<dataset>/quality_scores_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps",
        help="Directory where heatmap is saved.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="analysis/out/csv/two_modality_one_modified_quality_scores.csv",
        help="CSV path for extracted quality-score records.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI.",
    )
    return parser.parse_args()


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


def ordered_methods(records: pd.DataFrame, modality: str) -> list[str]:
    methods = sorted(records.loc[records["target"] == modality, "method"].dropna().astype(str).unique().tolist())
    excluded_methods = EXCLUDED_METHODS_BY_MODALITY.get(modality, set())
    methods = [method for method in methods if method not in excluded_methods]
    hint = METHOD_ORDER_HINTS.get(modality, [])
    in_hint = [method for method in hint if method in methods]
    extra = [method for method in methods if method not in in_hint]
    return in_hint + extra


def dataset_order_for_modality(modality: str) -> list[str]:
    datasets = list(DATASETS_BY_MODALITY[modality])
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


def format_modalities_pair(pair: str) -> str:
    return "+".join(MODALITY_SHORT.get(modality, modality.upper()) for modality in str(pair))


def build_dataset_tick_labels(datasets: list[str], subset: pd.DataFrame) -> list[str]:
    pair_map: dict[str, list[str]] = {}
    if not subset.empty and "modalities_pair" in subset.columns:
        per_dataset = (
            subset.dropna(subset=["modalities_pair"])
            .groupby("dataset")["modalities_pair"]
            .agg(lambda values: sorted({str(value) for value in values if str(value)}))
        )
        pair_map = per_dataset.to_dict()

    labels: list[str] = []
    for dataset in datasets:
        base_label = DATASET_LABEL.get(dataset, dataset)
        pairs = pair_map.get(dataset, [])
        if not pairs:
            labels.append(base_label)
            continue
        pair_label = " / ".join(format_modalities_pair(pair) for pair in pairs)
        labels.append(f"{base_label} ({pair_label})")
    return labels


def compute_records(input_root: Path, models: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for model in models:
        scored_root = input_root / model / "qwen_scored"
        if not scored_root.exists():
            continue

        for dataset in sorted({dataset for values in DATASETS_BY_MODALITY.values() for dataset in values}):
            dataset_dir = scored_root / dataset
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                continue

            for path in sorted(dataset_dir.glob("quality_scores_*.csv")):
                match = QUALITY_FILE_RE.match(path.name)
                if not match:
                    continue

                modalities = (match.group("modalities") or "").lower()
                noise_modalities = (match.group("noise") or "").lower()

                if len(modalities) != 2:
                    continue
                if len(noise_modalities) != 1:
                    continue
                if noise_modalities not in modalities:
                    continue

                target = noise_modalities
                quality_col = QUALITY_COL_BY_MODALITY.get(target)
                if quality_col is None:
                    continue

                frame = pd.read_csv(path)
                if "split" not in frame.columns or quality_col not in frame.columns:
                    continue

                frame = frame.dropna(subset=["split"]).copy()
                frame["quality_score"] = pd.to_numeric(frame[quality_col], errors="coerce")
                frame = frame.dropna(subset=["quality_score"]).copy()
                if frame.empty:
                    continue

                for split, split_frame in frame.groupby("split", sort=True):
                    split_meta = parse_split_metadata(split)
                    if split_meta["is_unmodified"]:
                        continue
                    if split_meta["target"] != target:
                        continue
                    if split_meta["severity"] is None:
                        continue

                    records.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "modalities_pair": "".join(sorted(modalities)),
                            "target": target,
                            "method": split_meta["method"],
                            "severity": split_meta["severity"],
                            "split": split,
                            "perturbed_quality_score": float(split_frame["quality_score"].mean()),
                        }
                    )

    return pd.DataFrame.from_records(records)


def plot_quality_heatmap(
    records: pd.DataFrame,
    models: list[str],
    severities: list[int],
    output_path: Path,
    dpi: int,
) -> None:
    if records.empty:
        raise RuntimeError("No records to plot.")

    row_modalities = MODALITY_ROWS
    column_specs = [(model, severity) for model in models for severity in severities]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_to_white_quality",
        ["#b2182b", "#f7f7f7"],
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(
        nrows=len(row_modalities),
        ncols=len(column_specs),
        figsize=(2.4 * len(column_specs) + 2.4, 4.8 * len(row_modalities)),
        sharey="row",
        constrained_layout=False,
    )

    if len(column_specs) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, modality in enumerate(row_modalities):
        methods = ordered_methods(records, modality)
        datasets = dataset_order_for_modality(modality)

        for col_idx, (model, severity) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            subset = records[
                (records["model"] == model)
                & (records["target"] == modality)
                & (records["severity"] == severity)
            ]

            matrix = (
                subset.pivot_table(index="dataset", columns="method", values="perturbed_quality_score", aggfunc="mean")
                .reindex(index=datasets, columns=methods)
            )
            if matrix.empty:
                matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)

            annotations = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

            if len(datasets) == 0 or len(methods) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            plot_values = matrix.to_numpy(dtype=float)
            plot_values = np.where(np.isnan(plot_values), np.nan, plot_values)
            ax.imshow(plot_values.T, cmap=cmap, norm=norm, aspect="auto")

            for y in range(len(methods) + 1):
                ax.axhline(y - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)
            for x in range(len(datasets) + 1):
                ax.axvline(x - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)

            for y_idx, method in enumerate(methods):
                for x_idx, dataset_name in enumerate(datasets):
                    text = annotations.at[dataset_name, method] if method in annotations.columns else ""
                    if text:
                        ax.text(x_idx, y_idx, text, ha="center", va="center", fontsize=12, color="black")

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
                ax.yaxis.set_label_coords(-0.72, 0.5)
            else:
                ax.set_ylabel("")

            dataset_tick_labels = build_dataset_tick_labels(datasets=datasets, subset=subset)
            ax.set_xticks(np.arange(len(datasets)))
            ax.set_xticklabels(dataset_tick_labels, rotation=30, ha="right", fontsize=11)
            ax.tick_params(axis="x", pad=2)

            method_tick_labels = [format_method_tick_label(method) for method in methods]
            ax.set_yticks(np.arange(len(methods)))
            if col_idx == 0:
                ax.set_yticklabels(method_tick_labels, rotation=0, fontsize=11)
                ax.tick_params(axis="y", labelleft=True, left=True)
            else:
                ax.tick_params(axis="y", labelleft=False, left=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.926, 0.135, 0.008, 0.73])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Perturbed modality quality score", fontsize=14, labelpad=2)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle("")
    fig.supxlabel("Dataset", fontsize=15, y=0.06)
    fig.subplots_adjust(left=0.12, right=0.882, top=0.900, bottom=0.155, wspace=0.08, hspace=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    csv_output = Path(args.csv_output)

    records = compute_records(input_root=input_root, models=args.models)
    if records.empty:
        raise RuntimeError("No two-modality one-perturbed quality-score records found.")

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(csv_output, index=False)

    output_path = output_dir / "two_modality_one_modified_heatmap_perturbed_quality_scores.png"
    plot_quality_heatmap(
        records=records,
        models=args.models,
        severities=args.severities,
        output_path=output_path,
        dpi=args.dpi,
    )

    print(f"Wrote CSV to {csv_output}")
    print(f"Wrote heatmap to {output_path}")


if __name__ == "__main__":
    main()
