import argparse
import math
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = None
    spearmanr = None

PREDICTION_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
QUALITY_RE = re.compile(r"^quality_scores_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
KV_PATTERN = re.compile(r"([A-Za-z]+)=([^=]+?)(?=_[A-Za-z]+=|$)")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

QUALITY_COL_BY_MODALITY = {
    "a": "audio_raw_quality",
    "i": "image_raw_quality",
    "t": "text_raw_quality",
    "v": "video_raw_quality",
}

PLOT_COLORS = {
    "Qwen_3B": "#1f77b4",
    "Qwen_7B": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate quality scores with performance degradation. "
            "Degradation is baseline_accuracy - noisy_accuracy, where baseline is prediction_<mods>_noise_.csv "
            "and noisy is prediction_<mods>_noise_<target>.csv."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model directories under out/.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="out",
        help="Root folder containing model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/quality_vs_degradation",
        help="Directory for csv summaries and plots.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of split-level points to compute a correlation.",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
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
    split_lower = split_text.lower()

    if split_lower in {"all", "test_all", "dev"} or "unmodified" in split_lower:
        return {
            "is_unmodified": True,
            "severity": None,
            "target": "",
            "method": "unmodified",
        }

    pairs = {key.upper(): value for key, value in KV_PATTERN.findall(split_text)}
    severity = None
    if "S" in pairs:
        try:
            severity = int(pairs["S"])
        except ValueError:
            severity = None

    method_values = []
    target_values = []
    for key, value in pairs.items():
        if key == "S":
            continue
        target_values.append(key.lower())
        method_values.append(value.lower())

    return {
        "is_unmodified": False,
        "severity": severity,
        "target": "+".join(target_values),
        "method": "+".join(method_values) if method_values else "unknown",
    }


def prediction_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    needed = {"split", "sample_id", "file", "prediction", "label"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame(
            columns=[
                "split",
                "sample_id",
                "file",
                "prediction_norm",
                "label_norm",
                "is_correct",
            ]
        )

    out = frame.loc[:, ["split", "sample_id", "file", "prediction", "label"]].copy()
    out = out.dropna(subset=["split", "sample_id", "file", "prediction", "label"])
    out["prediction_norm"] = out["prediction"].map(normalize_label)
    out["label_norm"] = out["label"].map(normalize_label)
    out["is_correct"] = out["prediction_norm"].eq(out["label_norm"]).astype(int)
    return out


def quality_table(path: Path, target_modality: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    quality_col = QUALITY_COL_BY_MODALITY.get(target_modality)
    if quality_col is None:
        return pd.DataFrame(columns=["split", "sample_id", "file", "quality"])
    needed = {"split", "sample_id", "file", quality_col}
    if not needed.issubset(frame.columns):
        return pd.DataFrame(columns=["split", "sample_id", "file", "quality"])

    out = frame.loc[:, ["split", "sample_id", "file", quality_col]].copy()
    out["quality"] = pd.to_numeric(out[quality_col], errors="coerce")
    out = out.dropna(subset=["split", "sample_id", "file", "quality"])
    return out.loc[:, ["split", "sample_id", "file", "quality"]]


def _build_baseline_keyed_table(baseline_pred: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    baseline_pred = baseline_pred.copy()
    baseline_pred["split_meta"] = baseline_pred["split"].map(parse_split_metadata)
    baseline_pred["is_unmodified_split"] = baseline_pred["split_meta"].map(lambda meta: bool(meta["is_unmodified"]))
    baseline_rows = baseline_pred[baseline_pred["is_unmodified_split"]].copy()
    if baseline_rows.empty:
        return pd.DataFrame(columns=["sample_id", "file", "baseline_correct"]), float("nan")

    baseline_rows = baseline_rows.loc[:, ["sample_id", "file", "is_correct"]].copy()
    baseline_rows = baseline_rows.sort_values(["sample_id", "file"]).drop_duplicates(["sample_id", "file"], keep="first")
    baseline_rows = baseline_rows.rename(columns={"is_correct": "baseline_correct"})
    baseline_accuracy = float(baseline_rows["baseline_correct"].mean() * 100.0)
    return baseline_rows, baseline_accuracy


def discover_split_records_for_model(model_dir: Path, scored_dir: Path, model_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    if not model_dir.exists() or not scored_dir.exists():
        return pd.DataFrame()

    dataset_dirs = [path for path in sorted(scored_dir.iterdir()) if path.is_dir()]
    for dataset_scored_dir in dataset_dirs:
        dataset = dataset_scored_dir.name
        baseline_dataset_dir = model_dir / dataset
        if not baseline_dataset_dir.exists() or not baseline_dataset_dir.is_dir():
            continue

        for quality_path in sorted(dataset_scored_dir.glob("quality_scores_*.csv")):
            match = QUALITY_RE.match(quality_path.name)
            if not match:
                continue

            modalities = (match.group("modalities") or "").lower()
            noisy = (match.group("noise") or "").lower()
            if len(modalities) != 2:
                continue
            if len(noisy) != 1:
                continue
            if noisy not in modalities:
                continue

            baseline_pred_path = baseline_dataset_dir / f"prediction_{modalities}_noise_.csv"
            noisy_pred_path = baseline_dataset_dir / f"prediction_{modalities}_noise_{noisy}.csv"
            if not baseline_pred_path.exists() or not noisy_pred_path.exists():
                continue

            baseline_pred = prediction_table(baseline_pred_path)
            noisy_pred = prediction_table(noisy_pred_path)
            quality = quality_table(quality_path, noisy)
            if baseline_pred.empty or noisy_pred.empty or quality.empty:
                continue

            baseline_keyed, baseline_accuracy = _build_baseline_keyed_table(baseline_pred)
            if baseline_keyed.empty or math.isnan(baseline_accuracy):
                continue

            noisy_pred["split_meta"] = noisy_pred["split"].map(parse_split_metadata)
            noisy_pred["is_unmodified_split"] = noisy_pred["split_meta"].map(lambda meta: bool(meta["is_unmodified"]))
            noisy_pred = noisy_pred[~noisy_pred["is_unmodified_split"]].copy()
            if noisy_pred.empty:
                continue

            noisy_pred["target"] = noisy_pred["split_meta"].map(lambda meta: str(meta["target"]))
            noisy_pred = noisy_pred[noisy_pred["target"] == noisy].copy()
            if noisy_pred.empty:
                continue

            noisy_pred["method"] = noisy_pred["split_meta"].map(lambda meta: str(meta["method"]))
            noisy_pred["severity"] = noisy_pred["split_meta"].map(lambda meta: meta["severity"])

            quality_grouped = (
                quality.groupby("split", as_index=False)
                .agg(
                    quality_mean=("quality", "mean"),
                    quality_std=("quality", "std"),
                    quality_n=("quality", "size"),
                )
                .sort_values("split")
            )
            noisy_grouped = (
                noisy_pred.groupby("split", as_index=False)
                .agg(
                    noisy_accuracy=("is_correct", lambda x: float(np.mean(x) * 100.0)),
                    n_samples=("is_correct", "size"),
                    method=("method", "first"),
                    severity=("severity", "first"),
                )
                .sort_values("split")
            )
            merged = noisy_grouped.merge(quality_grouped, on="split", how="inner")
            if merged.empty:
                continue

            for row in merged.to_dict(orient="records"):
                noisy_accuracy = float(row["noisy_accuracy"])
                records.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "modalities_pair": "".join(sorted(modalities)),
                        "target_modality": noisy,
                        "method": row["method"],
                        "severity": row["severity"],
                        "split": row["split"],
                        "baseline_accuracy": baseline_accuracy,
                        "noisy_accuracy": noisy_accuracy,
                        "degradation": baseline_accuracy - noisy_accuracy,
                        "quality_mean": float(row["quality_mean"]),
                        "quality_std": float(row["quality_std"]) if pd.notna(row["quality_std"]) else float("nan"),
                        "n_samples": int(row["n_samples"]),
                        "quality_n": int(row["quality_n"]),
                    }
                )

    return pd.DataFrame.from_records(records)


def correlation_stats(frame: pd.DataFrame, min_points: int) -> dict[str, object] | None:
    clean = frame.loc[:, ["quality_mean", "degradation"]].copy()
    clean = clean.dropna()
    n = len(clean)
    if n < min_points:
        return None

    if pearsonr is not None and spearmanr is not None:
        pearson_result = pearsonr(clean["quality_mean"], clean["degradation"])
        spearman_result = spearmanr(clean["quality_mean"], clean["degradation"])
        pearson = float(pearson_result.statistic)
        pearson_p = float(pearson_result.pvalue)
        spearman = float(spearman_result.statistic)
        spearman_p = float(spearman_result.pvalue)
    else:
        pearson = clean["quality_mean"].corr(clean["degradation"], method="pearson")
        spearman = clean["quality_mean"].corr(clean["degradation"], method="spearman")
        pearson_p = float("nan")
        spearman_p = float("nan")

    quality_mean = float(clean["quality_mean"].mean())
    degradation_mean = float(clean["degradation"].mean())
    degradation_improved_share = float((clean["degradation"] < 0).mean())

    return {
        "n_points": int(n),
        "pearson_r": float(pearson) if pd.notna(pearson) else float("nan"),
        "pearson_p": pearson_p,
        "spearman_rho": float(spearman) if pd.notna(spearman) else float("nan"),
        "spearman_p": spearman_p,
        "mean_quality": quality_mean,
        "mean_degradation": degradation_mean,
        "share_improvement": degradation_improved_share,
    }


def build_correlation_tables(records: pd.DataFrame, min_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    breakdown_rows: list[dict[str, object]] = []

    for model, model_frame in records.groupby("model"):
        stats = correlation_stats(model_frame, min_points=min_points)
        if stats is not None:
            summary_rows.append(
                {
                    "scope": "model_overall",
                    "model": model,
                    "dataset": "ALL",
                    "target_modality": "ALL",
                    **stats,
                }
            )

        for dataset, frame in model_frame.groupby("dataset"):
            stats = correlation_stats(frame, min_points=min_points)
            if stats is None:
                continue
            breakdown_rows.append(
                {
                    "scope": "model_dataset",
                    "model": model,
                    "dataset": dataset,
                    "target_modality": "ALL",
                    **stats,
                }
            )

        for target, frame in model_frame.groupby("target_modality"):
            stats = correlation_stats(frame, min_points=min_points)
            if stats is None:
                continue
            breakdown_rows.append(
                {
                    "scope": "model_modality",
                    "model": model,
                    "dataset": "ALL",
                    "target_modality": target,
                    **stats,
                }
            )

        for (dataset, target), frame in model_frame.groupby(["dataset", "target_modality"]):
            stats = correlation_stats(frame, min_points=min_points)
            if stats is None:
                continue
            breakdown_rows.append(
                {
                    "scope": "model_dataset_modality",
                    "model": model,
                    "dataset": dataset,
                    "target_modality": target,
                    **stats,
                }
            )

    overall_stats = correlation_stats(records, min_points=min_points)
    if overall_stats is not None:
        summary_rows.append(
            {
                "scope": "all_models_overall",
                "model": "ALL",
                "dataset": "ALL",
                "target_modality": "ALL",
                **overall_stats,
            }
        )

    summary = pd.DataFrame(summary_rows)
    breakdown = pd.DataFrame(breakdown_rows)
    return summary, breakdown


def scatter_plot(records: pd.DataFrame, output_path: Path) -> None:
    if records.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, squeeze=False)
    models = ["Qwen_3B", "Qwen_7B"]
    model_titles = {"Qwen_3B": "Qwen 3B", "Qwen_7B": "Qwen 7B"}
    base_title_fs = 12
    base_label_fs = 10
    base_tick_fs = 10
    title_fs = base_title_fs + 4
    label_fs = base_label_fs + 4
    tick_fs = base_tick_fs + 4

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_frame = records[records["model"] == model].dropna(subset=["quality_mean", "degradation"])
        if model_frame.empty:
            ax.set_title(f"{model_titles.get(model, model)} (no data)", fontsize=title_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(alpha=0.2)
            continue

        color = PLOT_COLORS.get(model, "#4c78a8")
        ax.scatter(
            model_frame["quality_mean"],
            model_frame["degradation"],
            alpha=0.75,
            s=24,
            c=color,
            edgecolors="none",
        )

        if len(model_frame) >= 2:
            coeffs = np.polyfit(model_frame["quality_mean"], model_frame["degradation"], 1)
            x_vals = np.linspace(model_frame["quality_mean"].min(), model_frame["quality_mean"].max(), 100)
            y_vals = coeffs[0] * x_vals + coeffs[1]
            ax.plot(x_vals, y_vals, color="black", linewidth=1.25, alpha=0.85)

        ax.set_title(model_titles.get(model, model), fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(alpha=0.2)

    fig.supxlabel("Mean quality score", fontsize=label_fs)
    fig.supylabel("Performance degradation (pp)", fontsize=label_fs)
    plt.tight_layout(rect=[0.04, 0.05, 1.0, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_text_report(
    summary: pd.DataFrame,
    breakdown: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("QUALITY VS PERFORMANCE DEGRADATION CORRELATION")
    lines.append("")

    if summary.empty:
        lines.append("No correlation results (not enough data points).")
    else:
        lines.append("OVERALL:")
        lines.append(summary.to_string(index=False))
        lines.append("")

    if breakdown.empty:
        lines.append("No breakdown rows (not enough data points in subgroups).")
    else:
        lines.append("BREAKDOWN:")
        lines.append(breakdown.sort_values(["scope", "model", "dataset", "target_modality"]).to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[pd.DataFrame] = []
    for model in args.models:
        model_root = out_root / model
        scored_root = model_root / "qwen_scored"
        model_records = discover_split_records_for_model(model_root, scored_root, model)
        if not model_records.empty:
            all_records.append(model_records)

    if not all_records:
        raise RuntimeError("No valid quality/prediction pairs found. Check directory structure and file names.")

    records = pd.concat(all_records, ignore_index=True)
    summary, breakdown = build_correlation_tables(records=records, min_points=args.min_points)

    records.to_csv(output_dir / "split_level_quality_vs_degradation.csv", index=False)
    summary.to_csv(output_dir / "correlation_summary.csv", index=False)
    breakdown.to_csv(output_dir / "correlation_breakdown.csv", index=False)
    scatter_plot(records=records, output_path=output_dir / "quality_vs_degradation_scatter.png")
    write_text_report(
        summary=summary,
        breakdown=breakdown,
        output_path=output_dir / "quality_vs_degradation_report.txt",
    )

    print(f"Wrote {len(records)} split-level rows to {output_dir}")


if __name__ == "__main__":
    main()
