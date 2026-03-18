from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / f"mplconfig_{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib
import numpy as np
import pandas as pd

from utils.calibration.quality_calibration import SUPPORTED_MODALITIES, load_percentile_calibration

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze calibration quality-score distributions and their calibrated percentiles. "
            "Produces CSV summaries and per-modality plots."
        )
    )
    parser.add_argument(
        "--scores-csv",
        action="append",
        required=True,
        help="Input score CSV path(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--calibration-json",
        type=str,
        default="",
        help=(
            "Optional percentile calibration JSON. If provided, percentiles are computed "
            "against this frozen reference. Otherwise, percentiles are derived from input scores."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for CSV summaries and plots.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="text,audio,image,video",
        help="Comma-separated modalities to analyze.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="1,5,10,25,50,75,90,95,99",
        help="Comma-separated percentile values (0..100) used for threshold tables and summaries.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation and only write CSV outputs.",
    )
    return parser.parse_args()


def _parse_modalities(modalities_arg: str) -> list[str]:
    modalities = [token.strip().lower() for token in modalities_arg.split(",") if token.strip()]
    bad = sorted(set(modalities) - set(SUPPORTED_MODALITIES))
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Supported: {list(SUPPORTED_MODALITIES)}")
    if not modalities:
        raise ValueError("No modalities selected for analysis.")
    return modalities


def _parse_percentiles(percentiles_arg: str) -> list[float]:
    raw_values = [token.strip() for token in percentiles_arg.split(",") if token.strip()]
    if not raw_values:
        raise ValueError("No percentile values provided.")
    percentiles: list[float] = []
    for token in raw_values:
        value = float(token)
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"Percentile must be in [0, 100], got {value}.")
        percentiles.append(value)
    return sorted(set(percentiles))


def _detect_score_column(df: pd.DataFrame, modality: str) -> str | None:
    candidates = [
        f"{modality}_raw_quality",
        f"{modality}_quality",
        modality,
    ]
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _collect_long_scores(
    df: pd.DataFrame,
    modalities: list[str],
    source_csv: Path,
) -> pd.DataFrame:
    if "dataset" in df.columns:
        dataset_values = df["dataset"].fillna("unknown").astype(str)
    else:
        dataset_values = pd.Series(["unknown"] * len(df), index=df.index, dtype="string")

    rows = []
    for modality in modalities:
        score_column = _detect_score_column(df, modality)
        if score_column is None:
            continue
        numeric_scores = pd.to_numeric(df[score_column], errors="coerce")
        valid_mask = numeric_scores.notna()
        if not valid_mask.any():
            continue
        rows.append(
            pd.DataFrame(
                {
                    "source_csv": str(source_csv),
                    "dataset": dataset_values[valid_mask].values,
                    "modality": modality,
                    "raw_score": numeric_scores[valid_mask].values.astype(float),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["source_csv", "dataset", "modality", "raw_score"])
    return pd.concat(rows, ignore_index=True)


def _calibrate_scores(raw_scores: np.ndarray, sorted_scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    if sorted_scores.size < 1:
        raise ValueError("Cannot calibrate against an empty score reference.")
    ranks_right = np.searchsorted(sorted_scores, raw_scores, side="right")
    calibrated = (ranks_right.astype(float) + 0.5) / (sorted_scores.size + 1.0)
    if not higher_is_better:
        calibrated = 1.0 - calibrated
    return np.clip(calibrated, 0.0, 1.0)


def _percentile_label(percentile: float) -> str:
    if abs(percentile - round(percentile)) < 1e-9:
        p_int = int(round(percentile))
        return f"p{p_int:02d}" if p_int < 100 else "p100"
    return f"p{str(percentile).replace('.', '_')}"


def _quantiles(scores: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    values = np.percentile(scores, percentiles, method="linear")
    return {_percentile_label(percentile): float(value) for percentile, value in zip(percentiles, values)}


def _summarize_group(group: pd.DataFrame, percentiles: list[float]) -> dict[str, float]:
    raw_scores = group["raw_score"].to_numpy(dtype=float)
    calibrated = group["calibrated_percentile"].to_numpy(dtype=float)
    summary = {
        "count": int(raw_scores.size),
        "raw_mean": float(np.mean(raw_scores)),
        "raw_std": float(np.std(raw_scores)),
        "raw_min": float(np.min(raw_scores)),
        "raw_max": float(np.max(raw_scores)),
        "calibrated_percentile_mean": float(np.mean(calibrated)),
        "calibrated_percentile_std": float(np.std(calibrated)),
        "calibrated_percentile_min": float(np.min(calibrated)),
        "calibrated_percentile_max": float(np.max(calibrated)),
    }
    raw_quantiles = _quantiles(raw_scores, percentiles)
    for label, value in raw_quantiles.items():
        summary[f"raw_{label}"] = value
    return summary


def _thresholds_for_calibrated_percentiles(
    sorted_scores: np.ndarray,
    higher_is_better: bool,
    percentiles: list[float],
) -> np.ndarray:
    qs = np.asarray(percentiles, dtype=float) / 100.0
    if not higher_is_better:
        qs = 1.0 - qs
    return np.quantile(sorted_scores, qs, method="linear")


def _plot_modality(
    modality: str,
    scores: np.ndarray,
    higher_is_better: bool,
    threshold_rows: pd.DataFrame,
    out_path: Path,
):
    sorted_scores = np.sort(scores)
    ecdf = np.arange(1, sorted_scores.size + 1, dtype=float) / (sorted_scores.size + 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_hist, ax_ecdf = axes

    bins = max(20, min(120, int(np.sqrt(sorted_scores.size)) + 10))
    ax_hist.hist(sorted_scores, bins=bins, color="#23689b", alpha=0.9, edgecolor="white")
    ax_hist.set_title(f"{modality}: raw score distribution")
    ax_hist.set_xlabel("raw quality score")
    ax_hist.set_ylabel("count")

    marker_rows = threshold_rows[
        threshold_rows["percentile"].isin([5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0])
    ]
    if marker_rows.empty:
        marker_rows = threshold_rows.head(5)
    for _, row in marker_rows.iterrows():
        ax_hist.axvline(
            row["raw_score_threshold"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"p{int(round(row['percentile']))}",
        )
    if not marker_rows.empty:
        ax_hist.legend(loc="best", fontsize=8)

    ax_ecdf.plot(sorted_scores, ecdf, color="#ed553b", linewidth=2.0)
    ax_ecdf.set_title(
        f"{modality}: empirical CDF ({'higher' if higher_is_better else 'lower'} is better)"
    )
    ax_ecdf.set_xlabel("raw quality score")
    ax_ecdf.set_ylabel("cumulative probability")
    ax_ecdf.set_ylim(0.0, 1.0)
    ax_ecdf.grid(alpha=0.2)

    for _, row in marker_rows.iterrows():
        target_y = row["percentile"] / 100.0
        if not higher_is_better:
            target_y = 1.0 - target_y
        ax_ecdf.scatter(row["raw_score_threshold"], target_y, s=28, color="#173f5f", alpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    modalities = _parse_modalities(args.modalities)
    requested_percentiles = _parse_percentiles(args.percentiles)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in args.scores_csv]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Score CSV not found: {path}")

    long_frames = []
    for path in input_paths:
        df = pd.read_csv(path)
        long_df = _collect_long_scores(df, modalities, path)
        if not long_df.empty:
            long_frames.append(long_df)
    if not long_frames:
        raise ValueError("No usable quality-score rows found in provided CSV files.")

    scores_long = pd.concat(long_frames, ignore_index=True)
    scores_long["dataset"] = scores_long["dataset"].fillna("unknown")

    calibrators = {}
    calibration_path = Path(args.calibration_json) if args.calibration_json else None
    if calibration_path is not None:
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")
        calibrators = load_percentile_calibration(calibration_path)

    calibrated_groups = []
    threshold_rows: list[dict[str, float | str | bool]] = []
    for modality in modalities:
        modality_group = scores_long[scores_long["modality"] == modality].copy()
        if modality_group.empty:
            continue

        raw_scores = modality_group["raw_score"].to_numpy(dtype=float)
        if modality in calibrators:
            calibrator = calibrators[modality]
            sorted_reference = np.asarray(calibrator.sorted_scores, dtype=float)
            higher_is_better = bool(calibrator.higher_is_better)
            reference_name = "calibration_json"
        else:
            sorted_reference = np.sort(raw_scores)
            higher_is_better = True
            reference_name = "input_scores"

        modality_group["calibrated_percentile"] = _calibrate_scores(
            raw_scores,
            sorted_reference,
            higher_is_better,
        )
        modality_group["calibrated_percentile_0_100"] = modality_group["calibrated_percentile"] * 100.0
        calibrated_groups.append(modality_group)

        thresholds = _thresholds_for_calibrated_percentiles(
            sorted_reference,
            higher_is_better,
            requested_percentiles,
        )
        for percentile, threshold in zip(requested_percentiles, thresholds):
            threshold_rows.append(
                {
                    "modality": modality,
                    "percentile": percentile,
                    "raw_score_threshold": float(threshold),
                    "higher_is_better": higher_is_better,
                    "reference": reference_name,
                }
            )

    if not calibrated_groups:
        raise ValueError("No modality scores available for the selected modalities.")

    calibrated_scores = pd.concat(calibrated_groups, ignore_index=True)
    calibrated_scores = calibrated_scores.sort_values(
        by=["modality", "dataset", "raw_score"], ascending=[True, True, True]
    )

    thresholds_df = pd.DataFrame(threshold_rows).sort_values(by=["modality", "percentile"])
    thresholds_df.to_csv(out_dir / "raw_score_thresholds_by_percentile.csv", index=False)

    calibrated_scores.to_csv(out_dir / "score_percentiles.csv", index=False)

    modality_summaries = []
    for modality, group in calibrated_scores.groupby("modality", sort=True):
        summary = {"modality": modality}
        summary.update(_summarize_group(group, requested_percentiles))
        modality_summaries.append(summary)
    pd.DataFrame(modality_summaries).to_csv(out_dir / "summary_by_modality.csv", index=False)

    dataset_modality_summaries = []
    for (dataset, modality), group in calibrated_scores.groupby(["dataset", "modality"], sort=True):
        summary = {"dataset": dataset, "modality": modality}
        summary.update(_summarize_group(group, requested_percentiles))
        dataset_modality_summaries.append(summary)
    pd.DataFrame(dataset_modality_summaries).to_csv(
        out_dir / "summary_by_dataset_modality.csv",
        index=False,
    )

    if not args.skip_plots:
        plot_dir = out_dir / "plots"
        for modality, group in calibrated_scores.groupby("modality", sort=True):
            modality_thresholds = thresholds_df[thresholds_df["modality"] == modality]
            if modality_thresholds.empty:
                continue
            higher_is_better = bool(modality_thresholds["higher_is_better"].iloc[0])
            _plot_modality(
                modality=modality,
                scores=group["raw_score"].to_numpy(dtype=float),
                higher_is_better=higher_is_better,
                threshold_rows=modality_thresholds,
                out_path=plot_dir / f"{modality}_distribution_and_ecdf.png",
            )

    print(f"[INFO] Wrote analysis outputs to {out_dir}")
    print(f"[INFO] Rows analyzed: {len(calibrated_scores)}")
    for modality, group in calibrated_scores.groupby("modality", sort=True):
        print(f"[INFO] {modality}: {len(group)} rows", flush=True)


if __name__ == "__main__":
    main()
