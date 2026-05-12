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
import pandas as pd

from heatmap_analysis.plot_qaa_vs_baseline_multimodal_delta_heatmaps import compute_qaa_minus_baseline

DATASET_LABELS = {
    "sentiment": "MELD-Sentiment",
    "emotion": "MELD-Emotion",
    "homeprice": "Austin Housing",
    "marine": "Marine Animals",
    "voxceleb": "Voxceleb2",
    "imdb": "IMDB",
    "nejm": "NEJM",
}

NOISE_LABELS = {
    "a": "noisy audio",
    "v": "noisy video",
    "t": "noisy text",
    "i": "noisy image",
}

SEVERITY_TARGET = 3
SEVERITY_LABEL = "Severity 1"


def dataset_display_name(dataset: str) -> str:
    return DATASET_LABELS.get(dataset.lower(), dataset)


def noise_display_name(noise_modalities: str) -> str:
    parts = [p for p in str(noise_modalities).lower() if p.strip()]
    if not parts:
        return "noisy unknown"
    labels = [NOISE_LABELS.get(p, f"noisy {p}") for p in parts]
    return " + ".join(labels)


def collect_rows(qaa_root: Path, baseline_root: Path) -> pd.DataFrame:
    model_name = baseline_root.name
    condition_df = compute_qaa_minus_baseline(
        baseline_root=baseline_root,
        qaa_root=qaa_root,
        model_name=model_name,
        significance_alpha=0.05,
        significance_min_paired=25,
        macro_f1_permutations=0,
        significance_seed=42,
    )
    if condition_df.empty:
        return pd.DataFrame()

    severity_df = condition_df[condition_df["severity"] == SEVERITY_TARGET].copy()
    if severity_df.empty:
        return pd.DataFrame()

    grouped = (
        severity_df.groupby(["dataset", "noise_modalities"], as_index=False)
        .agg(
            f1_baseline=("macro_f1_baseline", "mean"),
            f1_qaa=("macro_f1_qaa", "mean"),
            delta_f1_pp=("delta_macro_f1", "mean"),
            n_conditions=("method", "nunique"),
            n_paired=("n_paired", "sum"),
        )
        .dropna(subset=["delta_f1_pp"])
    )

    grouped["dataset_name"] = grouped["dataset"].map(dataset_display_name)
    grouped["noise_name"] = grouped["noise_modalities"].map(noise_display_name)
    grouped["dataset_noise_mix"] = grouped["dataset_name"] + " " + grouped["noise_name"]
    grouped["severity_raw"] = SEVERITY_TARGET
    grouped["severity"] = SEVERITY_LABEL
    return grouped


def plot_delta_bars(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values(["dataset_name", "noise_name"]).reset_index(drop=True)

    side = max(12, 0.85 * len(plot_df) + 4)
    fig, ax = plt.subplots(figsize=(side * 1.6, side))
    colors = ["#2E8B57" if v >= 0 else "#B22222" for v in plot_df["delta_f1_pp"]]

    y = range(len(plot_df))
    ax.barh(y, plot_df["delta_f1_pp"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["dataset_noise_mix"], fontsize=18)
    ax.set_xlabel("Delta Macro-F1 vs baseline (pp)", fontsize=20)
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=17)
    ax.grid(axis="x", alpha=0.25)

    max_abs = float(plot_df["delta_f1_pp"].abs().max()) if not plot_df.empty else 0.0
    limit = max(0.2, max_abs * 1.08)
    ax.set_xlim(-limit, limit)

    plt.tight_layout()
    fig.subplots_adjust(left=0.34, right=0.98)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barplot of QAA macro-F1 delta vs baseline for Qwen_7B at severity 3 (labeled Severity 1)."
    )
    parser.add_argument(
        "--qaa-root",
        type=str,
        default="out/Qwen_7B/qwen_scored",
        help="Root folder containing QAA-scored noisy predictions.",
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="out/Qwen_7B",
        help="Root folder containing baseline dataset folders.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="analysis/out/Qwen_7B/plots/qaa_qwen7b_delta_f1_severity1_barplot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="analysis/out/Qwen_7B/plots/qaa_qwen7b_delta_f1_severity1_barplot.csv",
        help="Output summary CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_rows(qaa_root=Path(args.qaa_root), baseline_root=Path(args.baseline_root))
    if summary.empty:
        raise RuntimeError("No valid rows found. Check input folders and prediction file formats.")

    summary = summary.sort_values(["dataset_name", "noise_name"]).reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    plot_delta_bars(summary=summary, output_path=Path(args.output_plot))

    print(f"Saved plot: {args.output_plot}")
    print(f"Saved summary CSV: {args.output_csv}")
    print(f"Rows: {len(summary)}")


if __name__ == "__main__":
    main()
