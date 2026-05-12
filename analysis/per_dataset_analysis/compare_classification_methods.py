import argparse
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


JOIN_COLUMN_PRIORITY = ["dataset", "split", "sample_id", "file"]
PREDICTION_GLOB = "prediction_*.csv"
SEVERITY_RE = re.compile(r"(?:^|_)s=(\d+)(?:_|$)", re.IGNORECASE)
LOW_CLASS_THRESHOLD = 10
HIGH_CLASS_THRESHOLD = 150
TOP_CLASS_COUNT = 9

MODALITY_LABELS = {
    "a": "Audio",
    "v": "Video",
    "t": "Text",
    "i": "Image",
    "av": "Audio, Video",
    "at": "Audio, Text",
    "ai": "Audio, Image",
    "tv": "Video, Text",
    "ti": "Image, Text",
    "it": "Image, Text",
    "atv": "Audio, Video, Text",
    "avi": "Audio, Video, Image",
    "ati": "Audio, Text, Image",
    "tvi": "Video, Text, Image",
}

MODALITY_COLORS = {
    "a": "#1f77b4",
    "v": "#ff7f0e",
    "t": "#2ca02c",
    "i": "#d62728",
    "av": "#9467bd",
    "at": "#8c564b",
    "ai": "#e377c2",
    "tv": "#7f7f7f",
    "ti": "#bcbd22",
    "it": "#bcbd22",
    "atv": "#17becf",
    "avi": "#4e79a7",
    "ati": "#f28e2b",
    "tvi": "#59a14f",
}


@dataclass
class PairComparisonResult:
    pair_id: str
    dataset: str
    severity_level: str
    configuration: str
    join_key: str
    n_baseline: int
    n_candidate: int
    n_paired: int
    coverage_baseline: float
    coverage_candidate: float
    accuracy_baseline: float
    accuracy_candidate: float
    delta_accuracy: float
    delta_ci_low: float
    delta_ci_high: float
    f1_macro_baseline: float
    f1_macro_candidate: float
    delta_f1_macro: float
    f1_weighted_baseline: float
    f1_weighted_candidate: float
    delta_f1_weighted: float
    precision_weighted_baseline: float
    precision_weighted_candidate: float
    delta_precision_weighted: float
    recall_weighted_baseline: float
    recall_weighted_candidate: float
    delta_recall_weighted: float
    mcc_baseline: float
    mcc_candidate: float
    delta_mcc: float
    wins_candidate: int
    wins_baseline: int
    ties: int
    mcnemar_pvalue: float
    mcnemar_discordant: int
    label_mismatch_count: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired method comparison for classification predictions. "
            "Compares matching prediction CSVs between baseline and candidate directories."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="out",
        help="Root directory for baseline prediction CSVs.",
    )
    parser.add_argument(
        "--candidate-root",
        type=str,
        default="out/qwen_scored",
        help="Root directory for candidate prediction CSVs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional model folder under out/. When set and roots are left as defaults, "
            "uses out/<model>/ and out/<model>/qwen_scored/. Supports out/<model>/(task/)<dataset>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/method_comparison",
        help="Output directory for comparison tables.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations for delta-accuracy confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrapping.",
    )
    parser.add_argument(
        "--min-paired",
        type=int,
        default=25,
        help="Minimum number of paired samples required to report a comparison.",
    )
    parser.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="Optional regex on relative path to include only specific file pairs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset names used to filter pair comparisons. "
            "Examples: imdb,nejm"
        ),
    )
    parser.add_argument(
        "--plot-datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset names for grouped metric plots. "
            "If omitted, plots are generated for all datasets present in the comparison."
        ),
    )
    parser.add_argument(
        "--plot-dir-name",
        type=str,
        default="plots",
        help="Subdirectory name under output-dir for generated plots.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="baseline",
        help="Legend label for baseline bars in comparison plots.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="qwen_scored",
        help="Legend label for candidate bars in comparison plots.",
    )
    parser.add_argument(
        "--plotting-util-style-compare",
        action="store_true",
        help=(
            "Generate plotting_util-style per-configuration plots with grouped bars "
            "for baseline vs qwen_scored."
        ),
    )
    parser.add_argument(
        "--plotting-util-style-datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated datasets for plotting_util-style comparison plots. "
            "Defaults to datasets found in pairwise comparison."
        ),
    )
    parser.add_argument(
        "--plotting-util-style-dir-name",
        type=str,
        default="plotting_util_style_comparison",
        help="Subdirectory name under output-dir for plotting_util-style comparison outputs.",
    )
    parser.add_argument(
        "--baseline-bar-palette",
        type=str,
        default="ocean_r",
        help="Matplotlib colormap name used for baseline bars.",
    )
    parser.add_argument(
        "--candidate-bar-palette",
        type=str,
        default="magma",
        help="Matplotlib colormap name used for candidate bars.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return normalized.strip().casefold()


def normalize_split_for_join(value: object) -> str:
    split = normalize_text(value)
    if split == "test_all":
        return "all"
    if split.startswith("test_"):
        return split[len("test_") :]
    return split


def discover_prediction_files(root: Path) -> dict[str, Path]:
    files = {}
    if not root.exists():
        return files
    for path in root.rglob(PREDICTION_GLOB):
        rel = path.relative_to(root).as_posix()
        files[rel] = path
    return files


def extract_dataset_from_relpath(relpath: str) -> str:
    path = Path(relpath)
    parent_name = path.parent.name.strip().lower()
    if parent_name:
        return parent_name
    parts = path.parts
    if not parts:
        return "unknown"
    if str(parts[0]).startswith("prediction_"):
        return "unknown"
    return str(parts[0]).lower()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [part.strip().lower() for part in raw_value.split(",") if part.strip()]


def extract_configuration_from_filename(path: Path) -> str:
    match = re.match(r"^prediction_(.+)\.csv$", path.name)
    return match.group(1) if match else path.stem


def extract_modality_token_from_configuration(configuration: str) -> str | None:
    normalized = str(configuration).strip().lower()
    noise_match = re.match(r"^([avti]+)_noise_.*$", normalized)
    if noise_match:
        token = noise_match.group(1)
    elif re.fullmatch(r"[avti]+", normalized):
        token = normalized
    else:
        return None

    canonical = "".join(ch for ch in "avti" if ch in token)
    return canonical if canonical else None


def configuration_is_multi_modality(configuration: str) -> bool:
    token = extract_modality_token_from_configuration(configuration)
    return token is not None and len(token) > 1


def determine_join_columns(df_baseline: pd.DataFrame, df_candidate: pd.DataFrame) -> list[str]:
    columns = [col for col in JOIN_COLUMN_PRIORITY if col in df_baseline.columns and col in df_candidate.columns]
    if "sample_id" in columns and "split" in columns:
        return ["split", "sample_id"]
    if "sample_id" in columns:
        return ["sample_id"]
    if "file" in columns and "split" in columns:
        return ["split", "file"]
    if "file" in columns:
        return ["file"]
    if "dataset" in columns and "split" in columns:
        return ["dataset", "split"]
    return columns[:1]


def preprocess_frame(df: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for col in join_columns:
        if col == "split":
            frame[col] = frame[col].map(normalize_split_for_join)
        else:
            frame[col] = frame[col].map(normalize_text)
    frame["label"] = frame["label"].map(normalize_text)
    frame["prediction"] = frame["prediction"].map(normalize_text)
    frame = frame[(frame["label"] != "") & (frame["prediction"] != "")]
    return frame


def deduplicate_by_join_key(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    if not join_columns:
        frame = frame.reset_index(drop=False).rename(columns={"index": "_row_id"})
        return frame
    return frame.drop_duplicates(subset=join_columns, keep="first")


def split_severity_key(split_value: object) -> str:
    split_text = str(split_value or "").strip().lower()
    match = SEVERITY_RE.search(split_text)
    if match:
        return f"S={int(match.group(1))}"
    return "__NO_SEVERITY__"


def filter_to_shared_severity_rows(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in baseline.columns or "split" not in candidate.columns:
        return baseline, candidate

    baseline = baseline.copy()
    candidate = candidate.copy()
    baseline["__severity_key"] = baseline["split"].map(split_severity_key)
    candidate["__severity_key"] = candidate["split"].map(split_severity_key)

    shared_severities = set(baseline["__severity_key"].unique()).intersection(set(candidate["__severity_key"].unique()))
    if not shared_severities:
        return baseline.iloc[0:0].copy(), candidate.iloc[0:0].copy()

    baseline = baseline[baseline["__severity_key"].isin(shared_severities)].copy()
    candidate = candidate[candidate["__severity_key"].isin(shared_severities)].copy()
    return baseline, candidate


def filter_splits_to_shared_severities(
    baseline_values: dict[str, float],
    candidate_values: dict[str, float],
) -> list[str]:
    all_splits = sorted(set(baseline_values).union(set(candidate_values)))
    if not all_splits:
        return []

    baseline_severities = {split_severity_key(split) for split in baseline_values}
    candidate_severities = {split_severity_key(split) for split in candidate_values}
    shared_severities = baseline_severities.intersection(candidate_severities)
    if not shared_severities:
        return []

    return [split for split in all_splits if split_severity_key(split) in shared_severities]


def safe_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        value = metric_fn(y_true, y_pred)
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def bootstrap_delta_accuracy(
    delta_per_sample: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(delta_per_sample)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        value = float(delta_per_sample[0])
        return value, value
    indices = rng.integers(0, n, size=(iterations, n))
    samples = delta_per_sample[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def mcnemar_exact_pvalue(wins_candidate: int, wins_baseline: int) -> float:
    discordant = wins_candidate + wins_baseline
    if discordant == 0:
        return 1.0
    k = min(wins_candidate, wins_baseline)
    return float(binomtest(k, n=discordant, p=0.5, alternative="two-sided").pvalue)


def compare_single_pair(
    relpath: str,
    baseline_path: Path,
    candidate_path: Path,
    bootstrap_iterations: int,
    rng: np.random.Generator,
    min_paired: int,
) -> list[PairComparisonResult]:
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    required = {"prediction", "label"}
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        return []

    configuration = extract_configuration_from_filename(baseline_path)

    join_columns = determine_join_columns(baseline, candidate)
    if not join_columns:
        return []

    baseline = preprocess_frame(baseline, join_columns)
    candidate = preprocess_frame(candidate, join_columns)
    baseline, candidate = filter_to_shared_severity_rows(baseline, candidate)
    if baseline.empty or candidate.empty:
        return []

    if "__severity_key" in baseline.columns and "__severity_key" in candidate.columns and "__severity_key" not in join_columns:
        join_columns = [*join_columns, "__severity_key"]

    baseline = deduplicate_by_join_key(baseline, join_columns)
    candidate = deduplicate_by_join_key(candidate, join_columns)

    paired = baseline.merge(
        candidate,
        on=join_columns,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    if paired.empty:
        return []

    join_key = ",".join(join_columns)
    dataset = extract_dataset_from_relpath(relpath)
    severity_keys = sorted(paired["__severity_key"].astype(str).unique().tolist())
    results: list[PairComparisonResult] = []

    for severity_key in severity_keys:
        paired_severity = paired[paired["__severity_key"].astype(str) == severity_key].copy()
        if len(paired_severity) < min_paired:
            continue

        baseline_severity = baseline[baseline["__severity_key"].astype(str) == severity_key]
        candidate_severity = candidate[candidate["__severity_key"].astype(str) == severity_key]
        n_baseline_severity = int(len(baseline_severity))
        n_candidate_severity = int(len(candidate_severity))
        if n_baseline_severity == 0 or n_candidate_severity == 0:
            continue

        y_true_baseline = paired_severity["label_baseline"].to_numpy()
        y_true_candidate = paired_severity["label_candidate"].to_numpy()
        y_pred_baseline = paired_severity["prediction_baseline"].to_numpy()
        y_pred_candidate = paired_severity["prediction_candidate"].to_numpy()

        label_mismatch = int((y_true_baseline != y_true_candidate).sum())
        y_true = y_true_baseline

        correct_baseline = (y_pred_baseline == y_true)
        correct_candidate = (y_pred_candidate == y_true)

        accuracy_baseline = float(correct_baseline.mean())
        accuracy_candidate = float(correct_candidate.mean())
        delta_accuracy = accuracy_candidate - accuracy_baseline

        delta_per_sample = correct_candidate.astype(np.int8) - correct_baseline.astype(np.int8)
        delta_ci_low, delta_ci_high = bootstrap_delta_accuracy(delta_per_sample, bootstrap_iterations, rng)

        f1_macro_baseline = safe_metric(lambda yt, yp: f1_score(yt, yp, average="macro"), y_true, y_pred_baseline)
        f1_macro_candidate = safe_metric(lambda yt, yp: f1_score(yt, yp, average="macro"), y_true, y_pred_candidate)
        f1_weighted_baseline = safe_metric(
            lambda yt, yp: f1_score(yt, yp, average="weighted"), y_true, y_pred_baseline
        )
        f1_weighted_candidate = safe_metric(
            lambda yt, yp: f1_score(yt, yp, average="weighted"), y_true, y_pred_candidate
        )
        precision_weighted_baseline = safe_metric(
            lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0),
            y_true,
            y_pred_baseline,
        )
        precision_weighted_candidate = safe_metric(
            lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0),
            y_true,
            y_pred_candidate,
        )
        recall_weighted_baseline = safe_metric(
            lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
            y_true,
            y_pred_baseline,
        )
        recall_weighted_candidate = safe_metric(
            lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
            y_true,
            y_pred_candidate,
        )
        mcc_baseline = safe_metric(matthews_corrcoef, y_true, y_pred_baseline)
        mcc_candidate = safe_metric(matthews_corrcoef, y_true, y_pred_candidate)

        wins_candidate = int((~correct_baseline & correct_candidate).sum())
        wins_baseline = int((correct_baseline & ~correct_candidate).sum())
        ties = int((correct_baseline == correct_candidate).sum())
        mcnemar_p = mcnemar_exact_pvalue(wins_candidate, wins_baseline)

        results.append(
            PairComparisonResult(
                pair_id=f"{relpath}::{severity_key}",
                dataset=dataset,
                severity_level=severity_key,
                configuration=configuration,
                join_key=join_key,
                n_baseline=n_baseline_severity,
                n_candidate=n_candidate_severity,
                n_paired=len(paired_severity),
                coverage_baseline=len(paired_severity) / n_baseline_severity,
                coverage_candidate=len(paired_severity) / n_candidate_severity,
                accuracy_baseline=accuracy_baseline,
                accuracy_candidate=accuracy_candidate,
                delta_accuracy=delta_accuracy,
                delta_ci_low=delta_ci_low,
                delta_ci_high=delta_ci_high,
                f1_macro_baseline=f1_macro_baseline,
                f1_macro_candidate=f1_macro_candidate,
                delta_f1_macro=f1_macro_candidate - f1_macro_baseline,
                f1_weighted_baseline=f1_weighted_baseline,
                f1_weighted_candidate=f1_weighted_candidate,
                delta_f1_weighted=f1_weighted_candidate - f1_weighted_baseline,
                precision_weighted_baseline=precision_weighted_baseline,
                precision_weighted_candidate=precision_weighted_candidate,
                delta_precision_weighted=precision_weighted_candidate - precision_weighted_baseline,
                recall_weighted_baseline=recall_weighted_baseline,
                recall_weighted_candidate=recall_weighted_candidate,
                delta_recall_weighted=recall_weighted_candidate - recall_weighted_baseline,
                mcc_baseline=mcc_baseline,
                mcc_candidate=mcc_candidate,
                delta_mcc=mcc_candidate - mcc_baseline,
                wins_candidate=wins_candidate,
                wins_baseline=wins_baseline,
                ties=ties,
                mcnemar_pvalue=mcnemar_p,
                mcnemar_discordant=wins_candidate + wins_baseline,
                label_mismatch_count=label_mismatch,
            )
        )

    return results


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    for dataset, group in df.groupby("dataset"):
        weighted_n = group["n_paired"].sum()
        if weighted_n <= 0:
            continue
        weighted_delta_acc = float((group["delta_accuracy"] * group["n_paired"]).sum() / weighted_n)
        weighted_delta_f1 = float((group["delta_f1_weighted"] * group["n_paired"]).sum() / weighted_n)
        weighted_delta_precision = float((group["delta_precision_weighted"] * group["n_paired"]).sum() / weighted_n)
        weighted_delta_recall = float((group["delta_recall_weighted"] * group["n_paired"]).sum() / weighted_n)
        weighted_delta_mcc = float((group["delta_mcc"] * group["n_paired"]).sum() / weighted_n)
        total_wins_candidate = int(group["wins_candidate"].sum())
        total_wins_baseline = int(group["wins_baseline"].sum())
        total_discordant = total_wins_candidate + total_wins_baseline
        rows.append(
            {
                "dataset": dataset,
                "pairs": int(len(group)),
                "total_paired_samples": int(weighted_n),
                "weighted_delta_accuracy": weighted_delta_acc,
                "weighted_delta_precision_weighted": weighted_delta_precision,
                "weighted_delta_recall_weighted": weighted_delta_recall,
                "weighted_delta_f1_weighted": weighted_delta_f1,
                "weighted_delta_mcc": weighted_delta_mcc,
                "total_wins_candidate": total_wins_candidate,
                "total_wins_baseline": total_wins_baseline,
                "total_discordant": total_discordant,
                "aggregate_mcnemar_pvalue": mcnemar_exact_pvalue(total_wins_candidate, total_wins_baseline),
            }
        )
    return pd.DataFrame(rows).sort_values(by="weighted_delta_accuracy", ascending=False)


def severity_output_dirname(severity_level: str) -> str:
    level = str(severity_level or "").strip()
    if not level:
        return "__NO_SEVERITY__"
    if level == "__NO_SEVERITY__":
        return level
    return level.replace("/", "_")


def parse_dataset_selection(raw_value: str | None, available_datasets: list[str]) -> list[str]:
    if raw_value is None:
        return available_datasets
    requested = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    if not requested:
        return available_datasets
    available_set = {dataset.lower() for dataset in available_datasets}
    return [dataset for dataset in requested if dataset in available_set]


def plot_grouped_metric_comparison(
    result_df: pd.DataFrame,
    output_dir: Path,
    datasets: list[str],
    baseline_label: str,
    candidate_label: str,
):
    if result_df.empty or not datasets:
        return

    metric_specs = [
        ("accuracy", "accuracy_baseline", "accuracy_candidate", (0.0, 1.0)),
        ("precision", "precision_weighted_baseline", "precision_weighted_candidate", (0.0, 1.0)),
        ("recall", "recall_weighted_baseline", "recall_weighted_candidate", (0.0, 1.0)),
        ("f1", "f1_weighted_baseline", "f1_weighted_candidate", (0.0, 1.0)),
        ("mcc", "mcc_baseline", "mcc_candidate", (-1.0, 1.0)),
    ]
    baseline_color = "#1f77b4"
    candidate_color = "#ff7f0e"

    for dataset in datasets:
        dataset_rows = result_df[result_df["dataset"].str.lower() == dataset.lower()].copy()
        if dataset_rows.empty:
            continue

        dataset_rows = dataset_rows.sort_values(by="configuration")
        if dataset_rows["configuration"].duplicated().any():
            x_labels = dataset_rows["pair_id"].astype(str).tolist()
        else:
            x_labels = dataset_rows["configuration"].astype(str).tolist()
        x = np.arange(len(x_labels))
        width = 0.38

        dataset_plot_dir = output_dir / dataset.lower()
        dataset_plot_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, baseline_col, candidate_col, y_limits in metric_specs:
            fig, ax = plt.subplots(figsize=(max(7, len(x_labels) * 1.2), 5))
            baseline_vals = dataset_rows[baseline_col].to_numpy(dtype=float)
            candidate_vals = dataset_rows[candidate_col].to_numpy(dtype=float)

            ax.bar(x - width / 2, baseline_vals, width, label=baseline_label, color=baseline_color)
            ax.bar(x + width / 2, candidate_vals, width, label=candidate_label, color=candidate_color)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_xlabel("configuration")
            ax.set_title(f"{dataset.upper()} - {metric_name}: {baseline_label} vs {candidate_label}")
            ax.set_ylim(*y_limits)
            ax.grid(axis="y")
            ax.legend()
            fig.tight_layout()

            out_path = dataset_plot_dir / f"{metric_name}_comparison.svg"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


def colormap_colors(name: str, n: int) -> list:
    if n <= 0:
        return []
    cmap = plt.get_cmap(name)
    if n == 1:
        return [cmap(0.6)]
    return [cmap(i / (n - 1)) for i in range(n)]


def is_reference_split(split: str) -> bool:
    split = str(split)
    return ("unmodified" in split) or ("all/" in split)


def split_modality_token(split: str) -> str:
    normalized = str(split).lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    token = "".join(ch for ch in "avti" if ch in normalized)
    return token if token else "unknown"


def split_modality_label(split: str) -> str:
    token = split_modality_token(split)
    return MODALITY_LABELS.get(token, "Unknown")


def split_modality_color(split: str) -> str:
    token = split_modality_token(split)
    return MODALITY_COLORS.get(token, "#000000")


def parse_config_from_filename(name: str, prefix: str) -> tuple[str, str] | None:
    match = re.match(rf"^{prefix}_(.+)_noise_(.*)\.csv$", name)
    if not match:
        return None
    return match.group(1), match.group(2)


def discover_configs_in_prepared_dir(prepared_dir: Path) -> set[tuple[str, str]]:
    configs: set[tuple[str, str]] = set()
    if not prepared_dir.exists():
        return configs
    for path in prepared_dir.glob("prepared_*.csv"):
        parsed = parse_config_from_filename(path.name, "prepared")
        if parsed is None:
            continue
        configs.add(parsed)
    for path in prepared_dir.glob("accuracy_*.csv"):
        parsed = parse_config_from_filename(path.name, "accuracy")
        if parsed is None:
            continue
        configs.add(parsed)
    return configs


def load_prepared_frame(task: str, modality_token: str, noise_token: str) -> pd.DataFrame:
    path = Path("analysis") / "out" / task / "prepared_data" / f"prepared_{modality_token}_noise_{noise_token}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame["split"] = frame["split"].astype(str)
    return frame


def load_accuracy_frame(task: str, modality_token: str, noise_token: str) -> pd.DataFrame:
    accuracy_path = Path("analysis") / "out" / task / "prepared_data" / f"accuracy_{modality_token}_noise_{noise_token}.csv"
    if accuracy_path.exists():
        frame = pd.read_csv(accuracy_path)
        if {"split", "accuracy"}.issubset(frame.columns):
            out = frame[["split", "accuracy"]].copy()
            out["split"] = out["split"].astype(str)
            out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce")
            return out.dropna(subset=["accuracy"])

    prepared = load_prepared_frame(task, modality_token, noise_token)
    required = {"split", "TP", "FN"}
    if prepared.empty or not required.issubset(prepared.columns):
        return pd.DataFrame(columns=["split", "accuracy"])

    tmp = prepared.copy()
    tmp["TP"] = pd.to_numeric(tmp["TP"], errors="coerce").fillna(0)
    tmp["FN"] = pd.to_numeric(tmp["FN"], errors="coerce").fillna(0)
    tmp["total"] = tmp["TP"] + tmp["FN"]
    grouped = tmp.groupby("split", as_index=False).agg(correct=("TP", "sum"), total=("total", "sum"))
    grouped["accuracy"] = grouped.apply(lambda row: row["correct"] / row["total"] if row["total"] > 0 else np.nan, axis=1)
    return grouped[["split", "accuracy"]].dropna(subset=["accuracy"])


def calculate_cm_metric(metric: str, tp: float, fp: float, tn: float, fn: float) -> float:
    tp = float(tp)
    fp = float(fp)
    tn = float(tn)
    fn = float(fn)
    if metric == "precision":
        denom = tp + fp
        return 0.0 if denom == 0 else tp / denom
    if metric == "recall":
        denom = tp + fn
        return 0.0 if denom == 0 else tp / denom
    if metric == "f1":
        precision = calculate_cm_metric("precision", tp, fp, tn, fn)
        recall = calculate_cm_metric("recall", tp, fp, tn, fn)
        denom = precision + recall
        return 0.0 if denom == 0 else (2 * precision * recall) / denom
    if metric == "mcc":
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if denom == 0 else ((tp * tn) - (fp * fn)) / denom
    raise ValueError(f"Unknown metric: {metric}")


def class_count_for_prepared(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "class_count" in frame.columns and not frame["class_count"].dropna().empty:
        return int(float(frame["class_count"].dropna().iloc[0]))
    if "class" in frame.columns:
        return int(frame["class"].astype(str).nunique())
    return 0


def top_classes_for_task(task: str, modality_token: str, noise_token: str, prepared: pd.DataFrame) -> list[str]:
    counts_path = Path("analysis") / "out" / task / "prepared_data" / f"true_label_count_{modality_token}_noise_{noise_token}.csv"
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        if {"class", "true_label_count"}.issubset(counts.columns):
            counts["class"] = counts["class"].astype(str)
            counts["true_label_count"] = pd.to_numeric(counts["true_label_count"], errors="coerce").fillna(0)
            sorted_classes = counts.sort_values("true_label_count", ascending=False)["class"].tolist()
            return sorted_classes[:TOP_CLASS_COUNT]

    if prepared.empty or not {"class", "TP", "FN"}.issubset(prepared.columns):
        return []
    fallback = prepared.copy()
    fallback["TP"] = pd.to_numeric(fallback["TP"], errors="coerce").fillna(0)
    fallback["FN"] = pd.to_numeric(fallback["FN"], errors="coerce").fillna(0)
    fallback["support"] = fallback["TP"] + fallback["FN"]
    class_support = fallback.groupby("class", as_index=False)["support"].sum()
    class_support["class"] = class_support["class"].astype(str)
    sorted_classes = class_support.sort_values("support", ascending=False)["class"].tolist()
    return sorted_classes[:TOP_CLASS_COUNT]


def score_for_split_and_class(
    split_frame: pd.DataFrame,
    metric: str,
    class_name: str,
    other_classes: list[str],
) -> float | None:
    required = {"class", "TP", "FP", "TN", "FN"}
    if split_frame.empty or not required.issubset(split_frame.columns):
        return None
    tmp = split_frame.copy()
    for col in ["TP", "FP", "TN", "FN"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
    tmp["class"] = tmp["class"].astype(str)

    if class_name == "others":
        relevant = tmp[tmp["class"].isin(other_classes)].copy()
        if relevant.empty:
            return None
        relevant["support"] = relevant["TP"] + relevant["FN"]
        total_support = relevant["support"].sum()
        per_class_scores = relevant.apply(
            lambda row: calculate_cm_metric(metric, row["TP"], row["FP"], row["TN"], row["FN"]),
            axis=1,
        )
        if total_support <= 0:
            return float(per_class_scores.mean()) if not per_class_scores.empty else None
        return float((per_class_scores * relevant["support"]).sum() / total_support)

    row = tmp[tmp["class"] == class_name]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(calculate_cm_metric(metric, r["TP"], r["FP"], r["TN"], r["FN"]))


def classes_to_plot_for_config(
    baseline_prepared: pd.DataFrame,
    candidate_prepared: pd.DataFrame,
    baseline_task: str,
    modality_token: str,
    noise_token: str,
) -> tuple[list[str], list[str], int]:
    class_count = max(class_count_for_prepared(baseline_prepared), class_count_for_prepared(candidate_prepared))
    if class_count == 0:
        return [], [], class_count

    all_classes = sorted(
        set(baseline_prepared.get("class", pd.Series(dtype=str)).astype(str).tolist())
        .union(set(candidate_prepared.get("class", pd.Series(dtype=str)).astype(str).tolist()))
    )
    if class_count < LOW_CLASS_THRESHOLD:
        return all_classes, [], class_count

    top_classes = top_classes_for_task(baseline_task, modality_token, noise_token, baseline_prepared)
    top_classes = [c for c in top_classes if c in all_classes]
    if len(top_classes) < TOP_CLASS_COUNT:
        for class_name in all_classes:
            if class_name not in top_classes:
                top_classes.append(class_name)
            if len(top_classes) == TOP_CLASS_COUNT:
                break

    others = [c for c in all_classes if c not in top_classes]
    classes_to_plot = top_classes.copy()
    if others:
        classes_to_plot.append("others")
    return classes_to_plot, others, class_count


def plot_accuracy_comparison_for_config(
    dataset: str,
    modality_token: str,
    noise_token: str,
    baseline_task: str,
    candidate_task: str,
    out_root: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_palette: str,
    candidate_palette: str,
):
    baseline_accuracy = load_accuracy_frame(baseline_task, modality_token, noise_token)
    candidate_accuracy = load_accuracy_frame(candidate_task, modality_token, noise_token)
    if baseline_accuracy.empty or candidate_accuracy.empty:
        return

    baseline_map = {str(row["split"]): float(row["accuracy"]) for _, row in baseline_accuracy.iterrows()}
    candidate_map = {str(row["split"]): float(row["accuracy"]) for _, row in candidate_accuracy.iterrows()}
    all_splits = filter_splits_to_shared_severities(
        baseline_values=baseline_map,
        candidate_values=candidate_map,
    )
    if not all_splits:
        return
    bar_splits = [split for split in all_splits if not is_reference_split(split)]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(bar_splits)), 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy over dataset")

    line_handles = []
    line_labels = []
    for split in all_splits:
        if not is_reference_split(split):
            continue
        color = split_modality_color(split)
        label = split_modality_label(split)
        baseline_value = baseline_map.get(split)
        candidate_value = candidate_map.get(split)
        if baseline_value is not None:
            ax.axhline(y=baseline_value, color=color, linestyle="--", alpha=0.95)
        if candidate_value is not None:
            ax.axhline(y=candidate_value, color=color, linestyle=":", alpha=0.95)
        if label not in line_labels:
            line_handles.append(Line2D([0], [0], color=color, linestyle="--"))
            line_labels.append(label)

    x = np.arange(len(bar_splits))
    width = 0.38
    baseline_scores = [baseline_map.get(split, np.nan) for split in bar_splits]
    candidate_scores = [candidate_map.get(split, np.nan) for split in bar_splits]
    baseline_colors = colormap_colors(baseline_palette, len(bar_splits))
    candidate_colors = colormap_colors(candidate_palette, len(bar_splits))

    ax.bar(x - width / 2, baseline_scores, width=width, color=baseline_colors)
    ax.bar(x + width / 2, candidate_scores, width=width, color=candidate_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_splits, rotation=45, ha="right")
    ax.grid(axis="y")

    method_handles = [
        Patch(facecolor=baseline_colors[0] if baseline_colors else "#1f77b4", label=baseline_label),
        Patch(facecolor=candidate_colors[0] if candidate_colors else "#ff7f0e", label=candidate_label),
        Line2D([0], [0], color="#555555", linestyle="--", label="reference baseline"),
        Line2D([0], [0], color="#555555", linestyle=":", label="reference qwen_scored"),
    ]
    fig.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    if line_handles:
        fig.legend(handles=line_handles, labels=line_labels, loc="upper right", bbox_to_anchor=(0.98, 0.78), title="Reference modality")
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    out_dir = out_root / dataset / "accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"accuracy_{modality_token}_noise_{noise_token}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison_for_config(
    dataset: str,
    modality_token: str,
    noise_token: str,
    baseline_task: str,
    candidate_task: str,
    out_root: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_palette: str,
    candidate_palette: str,
    metric: str,
):
    baseline_prepared = load_prepared_frame(baseline_task, modality_token, noise_token)
    candidate_prepared = load_prepared_frame(candidate_task, modality_token, noise_token)
    required_columns = {"split", "class", "TP", "FP", "TN", "FN"}
    if baseline_prepared.empty or candidate_prepared.empty:
        return
    if not required_columns.issubset(baseline_prepared.columns):
        return
    if not required_columns.issubset(candidate_prepared.columns):
        return

    classes_to_plot, other_classes, class_count = classes_to_plot_for_config(
        baseline_prepared=baseline_prepared,
        candidate_prepared=candidate_prepared,
        baseline_task=baseline_task,
        modality_token=modality_token,
        noise_token=noise_token,
    )
    if class_count > HIGH_CLASS_THRESHOLD or not classes_to_plot:
        return

    baseline_split_names = baseline_prepared["split"].astype(str).tolist()
    candidate_split_names = candidate_prepared["split"].astype(str).tolist()
    split_names = filter_splits_to_shared_severities(
        baseline_values={split: 1.0 for split in baseline_split_names},
        candidate_values={split: 1.0 for split in candidate_split_names},
    )
    if not split_names:
        return
    bar_splits = [split for split in split_names if not is_reference_split(split)]
    if not bar_splits:
        return

    baseline_split_map = {
        split: baseline_prepared[baseline_prepared["split"].astype(str) == split].copy() for split in split_names
    }
    candidate_split_map = {
        split: candidate_prepared[candidate_prepared["split"].astype(str) == split].copy() for split in split_names
    }

    fig, axes = plt.subplots(1, len(classes_to_plot), sharey=True, figsize=(4 * len(classes_to_plot), 5))
    if hasattr(axes, "ravel"):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for ax in axes:
        ax.set_ylim((-1, 1) if metric == "mcc" else (0, 1))
    axes[0].set_ylabel(metric)

    x = np.arange(len(bar_splits))
    width = 0.38
    baseline_colors = colormap_colors(baseline_palette, len(bar_splits))
    candidate_colors = colormap_colors(candidate_palette, len(bar_splits))
    line_handles = []
    line_labels = []

    for idx, class_name in enumerate(classes_to_plot):
        ax = axes[idx]
        baseline_scores = []
        candidate_scores = []

        for split in bar_splits:
            baseline_score = score_for_split_and_class(
                baseline_split_map.get(split, pd.DataFrame()),
                metric=metric,
                class_name=class_name,
                other_classes=other_classes,
            )
            candidate_score = score_for_split_and_class(
                candidate_split_map.get(split, pd.DataFrame()),
                metric=metric,
                class_name=class_name,
                other_classes=other_classes,
            )
            baseline_scores.append(np.nan if baseline_score is None else baseline_score)
            candidate_scores.append(np.nan if candidate_score is None else candidate_score)

        for split in split_names:
            if not is_reference_split(split):
                continue
            color = split_modality_color(split)
            label = split_modality_label(split)
            baseline_ref = score_for_split_and_class(
                baseline_split_map.get(split, pd.DataFrame()),
                metric=metric,
                class_name=class_name,
                other_classes=other_classes,
            )
            candidate_ref = score_for_split_and_class(
                candidate_split_map.get(split, pd.DataFrame()),
                metric=metric,
                class_name=class_name,
                other_classes=other_classes,
            )
            if baseline_ref is not None:
                ax.axhline(y=baseline_ref, color=color, linestyle="--", alpha=0.95)
            if candidate_ref is not None:
                ax.axhline(y=candidate_ref, color=color, linestyle=":", alpha=0.95)
            if label not in line_labels:
                line_handles.append(Line2D([0], [0], color=color, linestyle="--"))
                line_labels.append(label)

        ax.bar(x - width / 2, baseline_scores, width=width, color=baseline_colors)
        ax.bar(x + width / 2, candidate_scores, width=width, color=candidate_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_splits, rotation=45, ha="right")
        ax.set_title(class_name, fontsize=10)
        ax.grid(axis="y")

    method_handles = [
        Patch(facecolor=baseline_colors[0] if baseline_colors else "#1f77b4", label=baseline_label),
        Patch(facecolor=candidate_colors[0] if candidate_colors else "#ff7f0e", label=candidate_label),
        Line2D([0], [0], color="#555555", linestyle="--", label="reference baseline"),
        Line2D([0], [0], color="#555555", linestyle=":", label="reference qwen_scored"),
    ]
    fig.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    if line_handles:
        fig.legend(
            handles=line_handles,
            labels=line_labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.78),
            title="Reference modality",
        )
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    out_dir = out_root / dataset / metric
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{metric}_{modality_token}_noise_{noise_token}.svg", bbox_inches="tight")
    plt.close(fig)


def run_plotting_util_style_comparison(
    datasets: list[str],
    out_root: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_palette: str,
    candidate_palette: str,
    model: str | None = None,
):
    metrics = ["mcc", "f1", "recall", "precision"]
    for dataset in datasets:
        dataset = dataset.lower()
        baseline_task = dataset
        candidate_task = f"qwen_scored/{dataset}"
        if model:
            baseline_task = f"{model}/{baseline_task}"
            candidate_task = f"{model}/{candidate_task}"
        baseline_prepared_dir = Path("analysis") / "out" / baseline_task / "prepared_data"
        candidate_prepared_dir = Path("analysis") / "out" / candidate_task / "prepared_data"
        if not baseline_prepared_dir.exists() or not candidate_prepared_dir.exists():
            print(
                f"Skipping plotting_util-style comparison for '{dataset}': "
                "missing prepared_data directory for baseline or qwen_scored."
            )
            continue

        baseline_configs = discover_configs_in_prepared_dir(baseline_prepared_dir)
        candidate_configs = discover_configs_in_prepared_dir(candidate_prepared_dir)
        shared_configs = sorted(baseline_configs.intersection(candidate_configs))
        if not shared_configs:
            print(f"Skipping plotting_util-style comparison for '{dataset}': no shared configurations.")
            continue

        for modality_token, noise_token in shared_configs:
            plot_accuracy_comparison_for_config(
                dataset=dataset,
                modality_token=modality_token,
                noise_token=noise_token,
                baseline_task=baseline_task,
                candidate_task=candidate_task,
                out_root=out_root,
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                baseline_palette=baseline_palette,
                candidate_palette=candidate_palette,
            )
            for metric in metrics:
                plot_metric_comparison_for_config(
                    dataset=dataset,
                    modality_token=modality_token,
                    noise_token=noise_token,
                    baseline_task=baseline_task,
                    candidate_task=candidate_task,
                    out_root=out_root,
                    baseline_label=baseline_label,
                    candidate_label=candidate_label,
                    baseline_palette=baseline_palette,
                    candidate_palette=candidate_palette,
                    metric=metric,
                )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    baseline_root = Path(args.baseline_root)
    candidate_root = Path(args.candidate_root)
    if args.model:
        if baseline_root == Path("../out"):
            baseline_root = Path("../out") / args.model
        if candidate_root == Path("out/qwen_scored"):
            candidate_root = Path("../out") / args.model / "qwen_scored"

    output_dir = Path(args.output_dir)
    if args.model and output_dir == Path("analysis/out/method_comparison"):
        output_dir = Path("analysis") / "out" / args.model / "method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_files = discover_prediction_files(baseline_root)
    candidate_files = discover_prediction_files(candidate_root)

    shared_relpaths = sorted(set(baseline_files).intersection(candidate_files))
    requested_datasets = set(parse_csv_list(args.datasets))
    if requested_datasets:
        shared_relpaths = [
            rel for rel in shared_relpaths if extract_dataset_from_relpath(rel).lower() in requested_datasets
        ]
    if args.include_regex:
        pattern = re.compile(args.include_regex)
        shared_relpaths = [rel for rel in shared_relpaths if pattern.search(rel)]

    if not shared_relpaths:
        print("No matching prediction CSV files found.")
        print(f"Baseline root: {baseline_root}")
        print(f"Candidate root: {candidate_root}")
        return

    results = []
    skipped = 0
    for relpath in shared_relpaths:
        pair_results = compare_single_pair(
            relpath=relpath,
            baseline_path=baseline_files[relpath],
            candidate_path=candidate_files[relpath],
            bootstrap_iterations=args.bootstrap_iterations,
            rng=rng,
            min_paired=args.min_paired,
        )
        if not pair_results:
            skipped += 1
            continue
        results.extend([result.to_dict() for result in pair_results])

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("No valid pair comparisons generated.")
        print(f"Matched files: {len(shared_relpaths)}, skipped: {skipped}")
        return

    result_df = result_df.sort_values(by=["severity_level", "dataset", "delta_accuracy"], ascending=[True, True, False])
    dataset_summary = build_dataset_summary(result_df)

    pair_out_path = output_dir / "pairwise_comparison.csv"
    dataset_out_path = output_dir / "dataset_summary.csv"
    result_df.to_csv(pair_out_path, index=False)
    dataset_summary.to_csv(dataset_out_path, index=False)

    severity_dirs = []
    for severity_level, severity_df in result_df.groupby("severity_level", sort=True):
        severity_dir = output_dir / severity_output_dirname(str(severity_level))
        severity_dir.mkdir(parents=True, exist_ok=True)
        severity_pairwise = severity_df.sort_values(by=["dataset", "delta_accuracy"], ascending=[True, False])
        severity_summary = build_dataset_summary(severity_pairwise)
        severity_pairwise.to_csv(severity_dir / "pairwise_comparison.csv", index=False)
        severity_summary.to_csv(severity_dir / "dataset_summary.csv", index=False)
        severity_dirs.append(severity_dir)

    available_datasets = sorted(result_df["dataset"].astype(str).str.lower().unique().tolist())
    selected_datasets = parse_dataset_selection(args.plot_datasets, available_datasets)
    plots_dir = output_dir / args.plot_dir_name
    plot_grouped_metric_comparison(
        result_df=result_df,
        output_dir=plots_dir,
        datasets=selected_datasets,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )

    plotting_util_style_out_dir = output_dir / args.plotting_util_style_dir_name
    if args.plotting_util_style_compare:
        plotting_util_style_datasets = parse_dataset_selection(
            args.plotting_util_style_datasets,
            available_datasets,
        )
        run_plotting_util_style_comparison(
            datasets=plotting_util_style_datasets,
            out_root=plotting_util_style_out_dir,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            baseline_palette=args.baseline_bar_palette,
            candidate_palette=args.candidate_bar_palette,
            model=args.model,
        )

    print(f"Wrote pairwise results: {pair_out_path}")
    print(f"Wrote dataset summary: {dataset_out_path}")
    if severity_dirs:
        print("Wrote per-severity summaries:")
        for severity_dir in severity_dirs:
            print(f"  - {severity_dir}")
    print(f"Wrote plots under: {plots_dir}")
    if args.plotting_util_style_compare:
        print(f"Wrote plotting_util-style comparison plots under: {plotting_util_style_out_dir}")
    print(f"Matched files: {len(shared_relpaths)} | compared: {len(result_df)} | skipped: {skipped}")
    print()
    print("Top 10 improvements by delta accuracy:")
    print(
        result_df.nlargest(10, "delta_accuracy")[
            ["severity_level", "pair_id", "n_paired", "delta_accuracy", "mcnemar_pvalue"]
        ].to_string(index=False)
    )
    print()
    print("Top 10 degradations by delta accuracy:")
    print(
        result_df.nsmallest(10, "delta_accuracy")[
            ["severity_level", "pair_id", "n_paired", "delta_accuracy", "mcnemar_pvalue"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
