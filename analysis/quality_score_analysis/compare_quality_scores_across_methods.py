import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_quality_score_distributions_by_noise import discover_quality_score_files, to_long_quality_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare quality score distributions across qwen_scored methods."
    )
    parser.add_argument(
        "--methods-root",
        type=str,
        default="out/Qwen_7B/qwen_scored",
        help="Root directory containing method folders.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "steps_defined_unspecific",
            "steps_refined_specific",
            "steps_refined_specific_cap01",
            "task_specific",
            "imdb",
        ],
        help="Method folder names under methods-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/Qwen_7B/quality_method_comparison",
        help="Directory for csv/report outputs.",
    )
    return parser.parse_args()


def summarize_scores(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, as_index=False)
    summary = grouped["quality_score"].agg(
        n_scores="size",
        mean_score="mean",
        median_score="median",
        std_score="std",
        min_score="min",
        q25_score=lambda s: s.quantile(0.25),
        q75_score=lambda s: s.quantile(0.75),
        max_score="max",
    )
    zeros = grouped["quality_score"].apply(lambda s: int((s == 0).sum())).rename(columns={"quality_score": "zero_count"})
    merged = summary.merge(zeros, on=group_cols, how="left")
    merged["zero_count"] = merged["zero_count"].fillna(0).astype(int)
    merged["zero_pct"] = merged["zero_count"] / merged["n_scores"] * 100.0
    return merged


def main() -> None:
    args = parse_args()
    methods_root = Path(args.methods_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_methods = []

    for method in args.methods:
        method_dir = methods_root / method
        if not method_dir.exists():
            missing_methods.append(method)
            continue

        files = discover_quality_score_files(method_dir)
        for relpath, path in files:
            long_df = to_long_quality_scores(path, relpath)
            if long_df.empty:
                continue
            long_df = long_df.copy()
            long_df["method"] = method
            records.append(long_df)

    if not records:
        raise ValueError("No quality-score data found for requested methods.")

    all_scores = pd.concat(records, ignore_index=True)

    overall = summarize_scores(all_scores, ["method"]).sort_values(["zero_pct", "median_score", "mean_score"], ascending=[True, False, False])
    by_dataset = summarize_scores(all_scores, ["method", "dataset"]).sort_values(["dataset", "zero_pct", "median_score"], ascending=[True, True, False])
    by_setting = summarize_scores(
        all_scores,
        ["method", "source_relpath", "noise_level", "perturbation_method", "scored_modality"],
    ).sort_values(["source_relpath", "noise_level", "perturbation_method", "zero_pct"], ascending=[True, True, True, True])

    all_scores.to_csv(output_dir / "quality_scores_long.csv", index=False)
    overall.to_csv(output_dir / "quality_scores_summary_overall.csv", index=False)
    by_dataset.to_csv(output_dir / "quality_scores_summary_by_dataset.csv", index=False)
    by_setting.to_csv(output_dir / "quality_scores_summary_by_setting.csv", index=False)

    lines = [
        "QUALITY SCORES ACROSS METHODS",
        "",
        "OVERALL (sorted by least zero_pct)",
        overall.to_string(index=False),
    ]
    if missing_methods:
        lines.extend(["", "MISSING_METHOD_FOLDERS", ", ".join(missing_methods)])

    (output_dir / "quality_scores_method_report.txt").write_text("\n".join(lines) + "\n")
    print(output_dir / "quality_scores_method_report.txt")


if __name__ == "__main__":
    main()

