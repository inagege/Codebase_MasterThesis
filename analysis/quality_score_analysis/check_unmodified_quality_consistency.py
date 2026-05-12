import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

QUALITY_FILE_RE = re.compile(r"^quality_scores_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[AVTI])=(?P<method>.+?)_S=(?P<severity>\d+)")

DATASET_LABELS = {
    "emotion": "MELD Emotion",
    "homeprice": "Austin Housing",
    "imdb": "IMDB",
    "marine": "Marine Animals",
    "nejm": "NEJM",
    "sentiment": "MELD Sentiment",
    "voxceleb": "VoxCeleb2",
}
ALLOWED_DATASETS = sorted(DATASET_LABELS.keys())

QUALITY_COL_BY_MODALITY = {
    "a": "audio_raw_quality",
    "v": "video_raw_quality",
    "i": "image_raw_quality",
    "t": "text_raw_quality",
}

MODALITY_LABEL = {"a": "Audio", "v": "Video", "i": "Image", "t": "Text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare consistency of unmodified-modality quality scores between "
            "clean-pair quality files and one-modality-perturbed quality files."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model folders under out/ to analyze.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(ALLOWED_DATASETS),
        help="Comma-separated datasets to include.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="analysis/out/quality_consistency",
        help="Output directory for CSV and markdown report.",
    )
    return parser.parse_args()


def parse_split_metadata(split_value: object) -> dict[str, object]:
    split_text = str(split_value or "").strip()
    split_lower = split_text.casefold()
    if split_lower in {"all", "dev", "test_all"} or "unmodified" in split_lower:
        return {"method": "unmodified", "severity": pd.NA, "target": ""}

    match = SPLIT_NOISE_RE.search(split_text)
    if not match:
        return {"method": "unknown", "severity": pd.NA, "target": ""}
    return {
        "method": str(match.group("method") or "").lower(),
        "severity": int(match.group("severity")),
        "target": str(match.group("target") or "").lower(),
    }


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "file", "split"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["file"] = out["file"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip()
    out = out[(out["sample_id"] != "") & (out["file"] != "")].copy()
    return out


def read_quality_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = normalize_df(frame)
    if frame.empty:
        return frame
    return frame


def _safe_stat_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return float("nan")
    return float(xv.corr(yv))


def summarize_consistency(merged: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in merged.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        diff = group["score_diff"]
        abs_diff = diff.abs()
        row["n"] = int(len(group))
        row["mean_diff"] = float(diff.mean())
        row["median_diff"] = float(diff.median())
        row["std_diff"] = float(diff.std(ddof=0))
        row["mean_abs_diff"] = float(abs_diff.mean())
        row["median_abs_diff"] = float(abs_diff.median())
        row["exact_match_rate_pct"] = float((abs_diff == 0).mean() * 100.0)
        row["within_0_05_rate_pct"] = float((abs_diff <= 0.05).mean() * 100.0)
        row["within_0_10_rate_pct"] = float((abs_diff <= 0.10).mean() * 100.0)
        row["pearson_corr"] = _safe_stat_corr(group["score_clean"], group["score_noisy_file_unmodified_modality"])
        row["spearman_corr"] = _safe_stat_corr(
            group["score_clean"].rank(method="average"),
            group["score_noisy_file_unmodified_modality"].rank(method="average"),
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=group_cols)
    return pd.DataFrame(rows)


def build_report_text(overall: pd.DataFrame, by_dataset: pd.DataFrame, by_method: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Unmodified-Modality Quality Score Consistency")
    lines.append("")
    lines.append(
        "Comparison: clean two-modality file (`noise_`) vs one-modality-perturbed file (`noise_<mod>`), "
        "using the score of the still-unmodified modality on the same sample (`sample_id`, `file`)."
    )
    lines.append("")
    lines.append("## Overall")
    if overall.empty:
        lines.append("- No data.")
    else:
        for _, row in overall.sort_values(["model", "unmodified_modality"]).iterrows():
            lines.append(
                "- "
                f"{row['model']} | {MODALITY_LABEL.get(str(row['unmodified_modality']), row['unmodified_modality'])}: "
                f"n={int(row['n'])}, "
                f"mean|diff|={row['mean_abs_diff']:.4f}, "
                f"exact={row['exact_match_rate_pct']:.2f}%, "
                f"<=0.05={row['within_0_05_rate_pct']:.2f}%, "
                f"Spearman={row['spearman_corr']:.4f}"
            )
    lines.append("")
    lines.append("## Worst Dataset Pairs by Mean Absolute Difference")
    if by_dataset.empty:
        lines.append("- No data.")
    else:
        worst = by_dataset.sort_values("mean_abs_diff", ascending=False).head(15)
        for _, row in worst.iterrows():
            lines.append(
                "- "
                f"{row['model']} | {row['dataset_label']} | "
                f"unmodified={MODALITY_LABEL.get(str(row['unmodified_modality']), row['unmodified_modality'])}: "
                f"mean|diff|={row['mean_abs_diff']:.4f}, "
                f"exact={row['exact_match_rate_pct']:.2f}%, "
                f"<=0.05={row['within_0_05_rate_pct']:.2f}%"
            )
    lines.append("")
    lines.append("## Worst Method/Severity Pairs by Mean Absolute Difference")
    if by_method.empty:
        lines.append("- No data.")
    else:
        worst = by_method.sort_values("mean_abs_diff", ascending=False).head(20)
        for _, row in worst.iterrows():
            sev = "-" if pd.isna(row["severity"]) else int(row["severity"])
            lines.append(
                "- "
                f"{row['model']} | {row['dataset_label']} | "
                f"modified={MODALITY_LABEL.get(str(row['modified_modality']), row['modified_modality'])} | "
                f"unmodified={MODALITY_LABEL.get(str(row['unmodified_modality']), row['unmodified_modality'])} | "
                f"{row['method']} S={sev}: "
                f"mean|diff|={row['mean_abs_diff']:.4f}, "
                f"exact={row['exact_match_rate_pct']:.2f}%, "
                f"<=0.05={row['within_0_05_rate_pct']:.2f}%"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
    if not dataset_filter:
        dataset_filter = set(ALLOWED_DATASETS)

    comparison_rows: list[dict[str, object]] = []

    for model in args.models:
        model_root = Path("../out") / model / "qwen_scored"
        if not model_root.exists():
            print(f"[WARN] Missing root: {model_root}")
            continue

        for dataset in sorted(dataset_filter):
            dataset_dir = model_root / dataset
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                continue

            files = list(dataset_dir.glob("quality_scores_*.csv"))
            if not files:
                continue

            parsed_files: dict[tuple[str, str], Path] = {}
            for path in files:
                match = QUALITY_FILE_RE.match(path.name)
                if not match:
                    continue
                modalities = "".join(sorted((match.group("modalities") or "").lower()))
                noise = (match.group("noise") or "").lower()
                parsed_files[(modalities, noise)] = path

            pair_modalities = sorted({modalities for modalities, noise in parsed_files if noise == "" and len(modalities) >= 2})
            for modalities in pair_modalities:
                clean_path = parsed_files.get((modalities, ""))
                if clean_path is None:
                    continue
                clean_df = read_quality_file(clean_path)
                if clean_df.empty:
                    continue

                for modified_modality in modalities:
                    noisy_path = parsed_files.get((modalities, modified_modality))
                    if noisy_path is None:
                        continue

                    noisy_df = read_quality_file(noisy_path)
                    if noisy_df.empty:
                        continue

                    # Baseline mapping on same samples.
                    clean_base = clean_df[["sample_id", "file"]].copy()
                    for letter in modalities:
                        col = QUALITY_COL_BY_MODALITY[letter]
                        clean_base[f"clean_{letter}"] = pd.to_numeric(clean_df.get(col, np.nan), errors="coerce")

                    clean_base = clean_base.groupby(["sample_id", "file"], as_index=False).mean(numeric_only=True)

                    split_meta = noisy_df["split"].map(parse_split_metadata).apply(pd.Series)
                    noisy_work = pd.concat([noisy_df.reset_index(drop=True), split_meta.reset_index(drop=True)], axis=1)
                    noisy_work = noisy_work[noisy_work["target"].fillna("").astype(str).str.lower() == modified_modality].copy()
                    if noisy_work.empty:
                        continue

                    unmodified_modalities = [letter for letter in modalities if letter != modified_modality]
                    for unmodified_modality in unmodified_modalities:
                        noisy_col = QUALITY_COL_BY_MODALITY[unmodified_modality]
                        noisy_work["score_noisy_file_unmodified_modality"] = pd.to_numeric(
                            noisy_work.get(noisy_col, np.nan), errors="coerce"
                        )

                        merged = noisy_work.merge(
                            clean_base[["sample_id", "file", f"clean_{unmodified_modality}"]],
                            on=["sample_id", "file"],
                            how="inner",
                        ).rename(
                            columns={f"clean_{unmodified_modality}": "score_clean"}
                        )

                        merged = merged.dropna(subset=["score_clean", "score_noisy_file_unmodified_modality"]).copy()
                        if merged.empty:
                            continue

                        merged["score_diff"] = (
                            merged["score_noisy_file_unmodified_modality"] - merged["score_clean"]
                        )

                        for row in merged.itertuples(index=False):
                            comparison_rows.append(
                                {
                                    "model": model,
                                    "dataset": dataset,
                                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                                    "modalities_pair": modalities,
                                    "modified_modality": modified_modality,
                                    "unmodified_modality": unmodified_modality,
                                    "method": getattr(row, "method"),
                                    "severity": getattr(row, "severity"),
                                    "sample_id": getattr(row, "sample_id"),
                                    "file": getattr(row, "file"),
                                    "score_clean": getattr(row, "score_clean"),
                                    "score_noisy_file_unmodified_modality": getattr(row, "score_noisy_file_unmodified_modality"),
                                    "score_diff": getattr(row, "score_diff"),
                                    "score_abs_diff": abs(getattr(row, "score_diff")),
                                }
                            )

    if not comparison_rows:
        raise RuntimeError("No consistency rows were generated.")

    raw_df = pd.DataFrame(comparison_rows)
    raw_path = out_dir / "unmodified_quality_consistency_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_overall = summarize_consistency(raw_df, ["model", "unmodified_modality"])
    summary_overall_path = out_dir / "unmodified_quality_consistency_overall.csv"
    summary_overall.to_csv(summary_overall_path, index=False)

    summary_by_dataset = summarize_consistency(raw_df, ["model", "dataset", "dataset_label", "unmodified_modality"])
    summary_by_dataset_path = out_dir / "unmodified_quality_consistency_by_dataset.csv"
    summary_by_dataset.to_csv(summary_by_dataset_path, index=False)

    summary_by_method = summarize_consistency(
        raw_df,
        ["model", "dataset", "dataset_label", "modified_modality", "unmodified_modality", "method", "severity"],
    )
    summary_by_method_path = out_dir / "unmodified_quality_consistency_by_method.csv"
    summary_by_method.to_csv(summary_by_method_path, index=False)

    report_text = build_report_text(
        overall=summary_overall,
        by_dataset=summary_by_dataset,
        by_method=summary_by_method,
    )
    report_path = out_dir / "unmodified_quality_consistency_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_overall_path}")
    print(f"Wrote {summary_by_dataset_path}")
    print(f"Wrote {summary_by_method_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

