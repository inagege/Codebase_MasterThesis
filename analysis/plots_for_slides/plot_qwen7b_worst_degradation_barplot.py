import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from result_plot_parsing import parse_modalities_from_prediction_filename, parse_split_metadata

DATASET_ORDER = ["sentiment", "emotion", "homeprice", "marine", "voxceleb"]
DATASET_LABELS = {
    "sentiment": "MELD-Sentiment",
    "emotion": "MELD-Emotion",
    "homeprice": "Austin Housing",
    "marine": "Marine Animals",
    "voxceleb": "Voxceleb2",
}

SEVERITY_MAP = {3: "Severity 1", 5: "Severity 2"}
SEVERITY_ORDER = [3, 5]

TARGET_LABELS = {
    "a": "audio",
    "v": "video",
    "t": "text",
    "i": "image",
}

# KIT-inspired colors (green + blue)
KIT_GREEN = "#009682"
KIT_BLUE = "#4664AA"


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def compute_macro_f1(path: Path) -> float:
    data = pd.read_csv(path)
    required = {"prediction", "label"}
    if not required.issubset(data.columns):
        return float("nan")

    temp = data[["prediction", "label"]].copy()
    temp.dropna(subset=["prediction", "label"], inplace=True)
    if temp.empty:
        return float("nan")
    temp["prediction_norm"] = temp["prediction"].map(normalize_text)
    temp["label_norm"] = temp["label"].map(normalize_text)
    return float(
        f1_score(
            temp["label_norm"],
            temp["prediction_norm"],
            average="macro",
            zero_division=0,
        )
    )


def format_target(target: str) -> str:
    tokens = [part.strip().lower() for part in str(target).split("+") if part.strip()]
    if not tokens:
        return "unknown"
    return " / ".join(TARGET_LABELS.get(token, token) for token in tokens)


def format_method(method: str) -> str:
    if not method:
        return "unknown"
    return method.replace("_", " ")


def collect_degradation_records(model_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for dataset in DATASET_ORDER:
        dataset_dir = model_root / dataset
        if not dataset_dir.exists():
            continue

        for noisy_file in sorted(dataset_dir.glob("prediction_*_noise_*.csv")):
            file_meta = parse_modalities_from_prediction_filename(noisy_file)
            modalities = file_meta.get("modalities", "")
            noise_modalities = file_meta.get("noise_modalities", "")
            if not modalities or not noise_modalities:
                continue

            baseline_file = dataset_dir / f"prediction_{modalities}_noise_.csv"
            if not baseline_file.exists():
                continue

            baseline_macro_f1 = compute_macro_f1(baseline_file)
            if pd.isna(baseline_macro_f1):
                continue

            noisy_df = pd.read_csv(noisy_file)
            if not {"split", "prediction", "label"}.issubset(noisy_df.columns):
                continue

            noisy_df = noisy_df[["split", "prediction", "label"]].dropna(subset=["split", "prediction", "label"])
            if noisy_df.empty:
                continue

            noisy_df["prediction_norm"] = noisy_df["prediction"].map(normalize_text)
            noisy_df["label_norm"] = noisy_df["label"].map(normalize_text)
            split_f1_rows: list[dict[str, object]] = []
            for split_name, split_df in noisy_df.groupby("split"):
                macro_f1 = f1_score(
                    split_df["label_norm"],
                    split_df["prediction_norm"],
                    average="macro",
                    zero_division=0,
                )
                split_f1_rows.append({"split": split_name, "macro_f1": float(macro_f1)})

            for row in split_f1_rows:
                split_name = str(row["split"])
                split_meta = parse_split_metadata(split_name)
                severity = split_meta.get("severity")
                if severity not in SEVERITY_MAP:
                    continue

                perturbed_macro_f1 = float(row["macro_f1"])
                degradation = baseline_macro_f1 - perturbed_macro_f1
                records.append(
                    {
                        "dataset": dataset,
                        "severity": int(severity),
                        "severity_label": SEVERITY_MAP[int(severity)],
                        "drop_macro_f1": degradation,
                        "baseline_macro_f1": baseline_macro_f1,
                        "perturbed_macro_f1": perturbed_macro_f1,
                        "perturbation_target": str(split_meta.get("perturbation_target", "")),
                        "perturbation_method": str(split_meta.get("perturbation_method", "")),
                        "noisy_file": noisy_file.name,
                    }
                )

    return pd.DataFrame.from_records(records)


def build_worst_summary(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    idx = records.groupby(["dataset", "severity"])["drop_macro_f1"].idxmax()
    summary = records.loc[idx].copy()
    return summary.sort_values(["dataset", "severity"])


def plot_worst_degradation(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    base_fontsize = 18

    y = np.arange(len(DATASET_ORDER))
    bar_width = 0.36
    offsets = {3: -bar_width / 2, 5: bar_width / 2}
    colors = {3: KIT_GREEN, 5: KIT_BLUE}
    max_drop = float(summary["drop_macro_f1"].max()) if not summary.empty else 0.0
    annotation_pad = max(0.008, 0.03 * max_drop)
    min_annotation_x = max(0.02, 0.12 * max_drop)

    bar_values: dict[tuple[str, int], float] = {}
    bar_texts: dict[tuple[str, int], str] = {}

    for severity in SEVERITY_ORDER:
        dataset_to_row = {
            row["dataset"]: row
            for row in summary[summary["severity"] == severity].to_dict(orient="records")
        }
        widths = []
        labels = []
        for dataset in DATASET_ORDER:
            row = dataset_to_row.get(dataset)
            if row is None:
                widths.append(np.nan)
                labels.append("")
                continue
            width = float(row["drop_macro_f1"])
            widths.append(width)
            target_label = format_target(str(row["perturbation_target"]))
            method_label = format_method(str(row["perturbation_method"]))
            label_text = f"{target_label} - {method_label}"
            labels.append(label_text)
            bar_values[(dataset, severity)] = width
            bar_texts[(dataset, severity)] = label_text

        bars = ax.barh(
            y + offsets[severity],
            widths,
            height=bar_width,
            color=colors[severity],
            label=SEVERITY_MAP[severity],
        )

    # Add annotations dataset-wise to avoid overlaps (especially identical labels across severities)
    for idx, dataset in enumerate(DATASET_ORDER):
        sev3_text = bar_texts.get((dataset, 3), "")
        sev5_text = bar_texts.get((dataset, 5), "")
        sev3_width = bar_values.get((dataset, 3), np.nan)
        sev5_width = bar_values.get((dataset, 5), np.nan)

        if sev3_text and sev5_text and sev3_text == sev5_text and not np.isnan(sev3_width) and not np.isnan(sev5_width):
            shared_x = max(sev3_width, sev5_width) + annotation_pad
            shared_x = max(shared_x, min_annotation_x)
            ax.text(
                shared_x,
                idx,
                sev3_text,
                ha="left",
                va="center",
                fontsize=base_fontsize - 1,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                zorder=3,
            )
            continue

        for severity in SEVERITY_ORDER:
            text = bar_texts.get((dataset, severity), "")
            width = bar_values.get((dataset, severity), np.nan)
            if not text or np.isnan(width):
                continue

            text_x = width + (annotation_pad if width >= 0 else -annotation_pad)
            if width >= 0:
                text_x = max(text_x, min_annotation_x)
            else:
                text_x = min(text_x, -min_annotation_x)
            ha = "left" if width >= 0 else "right"
            y_pos = idx + offsets[severity]
            ax.text(
                text_x,
                y_pos,
                text,
                ha=ha,
                va="center",
                fontsize=base_fontsize - 1,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                zorder=3,
            )

    ax.axvline(0.0, color="black", linewidth=1, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [DATASET_LABELS.get(dataset, dataset) for dataset in DATASET_ORDER],
        rotation=0,
        fontsize=base_fontsize,
    )
    ax.set_xlabel("Macro-F1 drop vs unmodified baseline", fontsize=base_fontsize + 1)
    ax.set_ylabel("Dataset", fontsize=base_fontsize + 1)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=base_fontsize)
    ax.legend(title="Severity", fontsize=base_fontsize, title_fontsize=base_fontsize)
    ax.grid(axis="x", alpha=0.25)
    right_limit = max(0.34, max_drop + 0.12)
    ax.set_xlim(0.0, right_limit)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped bar plot of worst perturbation degradation per dataset for Qwen 7B."
    )
    parser.add_argument("--model-root", type=str, default="out/Qwen_7B", help="Model output root folder.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="analysis/out/Qwen_7B/plots/worst_degradation_by_input_perturbation.png",
        help="Output path for the plot image.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="analysis/out/Qwen_7B/plots/worst_degradation_by_input_perturbation.csv",
        help="Output path for summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)

    records = collect_degradation_records(model_root=model_root)
    summary = build_worst_summary(records=records)
    if summary.empty:
        raise RuntimeError(f"No valid perturbation rows found under {model_root}")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    plot_worst_degradation(summary=summary, output_path=Path(args.output_path))

    print(f"Saved plot: {args.output_path}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
