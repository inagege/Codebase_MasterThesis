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
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
JOIN_COLUMN_PRIORITY = ["dataset", "split", "sample_id", "file"]
AGG_KEYS = ["dataset", "modalities", "noise_modalities", "target", "method", "severity"]

MODALITY_ROWS = ["a", "i", "t", "v"]
MODALITY_LABEL = {"a": "Audio", "i": "Image", "t": "Text", "v": "Video"}
MODALITY_SHORT = {"a": "A", "i": "I", "t": "T", "v": "V"}

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
            "Create QAA-vs-baseline multimodal delta heatmaps under noisy conditions "
            "(more than one modality input, maximum one perturbed modality)."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Models under out/ to compare (default: Qwen_3B Qwen_7B).",
    )
    parser.add_argument(
        "--qaa-subdir",
        type=str,
        default="qwen_scored",
        help="QAA result subdirectory under out/<model>/ (default: qwen_scored).",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Noise severities shown as columns (default: 3 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps",
        help="Directory where heatmaps are saved.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="analysis/out/csv",
        help="Directory where CSV files are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.05,
        help="P-value threshold for marking significant QAA-vs-baseline changes (default: 0.05).",
    )
    parser.add_argument(
        "--significance-min-paired",
        type=int,
        default=25,
        help="Minimum paired samples required before significance markers are shown (default: 25).",
    )
    parser.add_argument(
        "--macro-f1-permutations",
        type=int,
        default=500,
        help="Number of paired-permutation samples for macro-F1 significance (default: 500).",
    )
    parser.add_argument(
        "--significance-seed",
        type=int,
        default=42,
        help="Random seed used for permutation-based significance tests.",
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


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def normalize_split_for_join(value: object) -> str:
    split = normalize_text(value)
    if split == "test_all":
        return "all"
    if split.startswith("test_"):
        return split[len("test_") :]
    return split


def determine_join_columns(df_baseline: pd.DataFrame, df_qaa: pd.DataFrame) -> list[str]:
    columns = [col for col in JOIN_COLUMN_PRIORITY if col in df_baseline.columns and col in df_qaa.columns]
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


def preprocess_frame_for_pairing(df: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for col in join_columns:
        if col == "split":
            frame[col] = frame[col].map(normalize_split_for_join)
        else:
            frame[col] = frame[col].map(normalize_text)
    return frame


def deduplicate_by_join_key(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    if not join_columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=join_columns, keep="first")


def mcnemar_exact_pvalue(wins_qaa: int, wins_baseline: int) -> float:
    discordant = wins_qaa + wins_baseline
    if discordant <= 0:
        return 1.0

    smaller_tail = min(wins_qaa, wins_baseline)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(smaller_tail + 1)
    ]
    max_log = max(log_terms)
    tail_prob = math.exp(max_log) * sum(math.exp(value - max_log) for value in log_terms)
    return float(min(1.0, 2.0 * tail_prob))


def encode_labels(values: np.ndarray, label_to_index: dict[str, int], unknown_index: int) -> np.ndarray:
    as_list = values.tolist()
    return np.fromiter((label_to_index.get(value, unknown_index) for value in as_list), dtype=np.int32, count=len(as_list))


def macro_f1_percent_from_encoded(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    n_labels: int,
) -> float:
    if n_labels <= 0 or y_true_idx.size == 0:
        return float("nan")

    class_count = n_labels + 1
    flat_index = y_true_idx.astype(np.int64) * class_count + y_pred_idx.astype(np.int64)
    confusion = np.bincount(flat_index, minlength=class_count * class_count).reshape(class_count, class_count)
    tp = np.diag(confusion[:n_labels, :n_labels]).astype(float)
    support_true = confusion[:n_labels, :].sum(axis=1).astype(float)
    support_pred = confusion[:, :n_labels].sum(axis=0).astype(float)
    denom = support_true + support_pred
    f1_per_label = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0.0)
    return float(np.mean(f1_per_label) * 100.0)


def paired_permutation_macro_f1_pvalue(
    y_true_idx: np.ndarray,
    y_pred_baseline_idx: np.ndarray,
    y_pred_qaa_idx: np.ndarray,
    n_labels: int,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if permutations <= 0 or y_true_idx.size == 0 or n_labels <= 0:
        return float("nan")

    observed_delta = (
        macro_f1_percent_from_encoded(y_true_idx, y_pred_qaa_idx, n_labels)
        - macro_f1_percent_from_encoded(y_true_idx, y_pred_baseline_idx, n_labels)
    )
    if not math.isfinite(observed_delta):
        return float("nan")

    abs_observed = abs(observed_delta)
    extreme_count = 0
    for _ in range(permutations):
        swap_mask = rng.random(y_true_idx.size) < 0.5
        perm_baseline = np.where(swap_mask, y_pred_qaa_idx, y_pred_baseline_idx)
        perm_qaa = np.where(swap_mask, y_pred_baseline_idx, y_pred_qaa_idx)
        delta = (
            macro_f1_percent_from_encoded(y_true_idx, perm_qaa, n_labels)
            - macro_f1_percent_from_encoded(y_true_idx, perm_baseline, n_labels)
        )
        if abs(delta) >= abs_observed - 1e-12:
            extreme_count += 1

    return float((extreme_count + 1) / (permutations + 1))


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


def dataset_universe() -> list[str]:
    all_datasets: set[str] = set()
    for values in DATASETS_BY_MODALITY.values():
        all_datasets.update(values)
    return sorted(all_datasets)


def collect_dataset_labels(
    baseline_root: Path,
    qaa_root: Path,
) -> dict[str, list[str]]:
    labels_by_dataset: dict[str, set[str]] = {}
    for dataset in dataset_universe():
        labels_by_dataset.setdefault(dataset, set())
        for root in (baseline_root, qaa_root):
            dataset_dir = root / dataset
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                continue
            for path in dataset_dir.glob("prediction_*.csv"):
                match = PREDICTION_FILE_RE.match(path.name)
                if not match:
                    continue
                modalities = (match.group("modalities") or "").lower()
                if len(modalities) <= 1:
                    continue
                frame = read_prediction_file(path)
                labels_by_dataset[dataset].update(frame["label_norm"].tolist())
    return {dataset: sorted(values) for dataset, values in labels_by_dataset.items()}


def collect_multimodal_noisy_metrics_for_root(
    model_root: Path,
    labels_by_dataset: dict[str, list[str]],
    source_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for dataset in dataset_universe():
        dataset_dir = model_root / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue

        labels = labels_by_dataset.get(dataset, [])

        for path in sorted(dataset_dir.glob("prediction_*.csv")):
            match = PREDICTION_FILE_RE.match(path.name)
            if not match:
                continue

            modalities = (match.group("modalities") or "").lower()
            noise_modalities = (match.group("noise") or "").lower()

            # Require multimodal input and at most one perturbed modality.
            if len(modalities) <= 1:
                continue
            if len(noise_modalities) > 1:
                continue
            if len(noise_modalities) == 0:
                continue
            if noise_modalities not in modalities:
                continue

            frame = read_prediction_file(path)
            if frame.empty:
                continue

            for split, split_frame in frame.groupby("split", sort=True):
                split_meta = parse_split_metadata(split)
                if split_meta["is_unmodified"]:
                    continue
                if split_meta["target"] != noise_modalities:
                    continue
                if split_meta["severity"] is None:
                    continue

                accuracy = float(split_frame["is_correct"].mean() * 100.0)
                macro_f1 = (
                    float("nan")
                    if dataset == "nejm"
                    else float(macro_f1_score(split_frame["label_norm"], split_frame["prediction_norm"], labels) * 100.0)
                )

                records.append(
                    {
                        "source": source_name,
                        "dataset": dataset,
                        "modalities": "".join(sorted(modalities)),
                        "noise_modalities": noise_modalities,
                        "target": split_meta["target"],
                        "method": split_meta["method"],
                        "severity": split_meta["severity"],
                        "split": split,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                    }
                )

    return pd.DataFrame.from_records(records)


def collect_paired_significance(
    baseline_root: Path,
    qaa_root: Path,
    labels_by_dataset: dict[str, list[str]],
    alpha: float,
    min_paired: int,
    macro_f1_permutations: int,
    significance_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(significance_seed)
    label_to_index_by_dataset = {
        dataset: {label: idx for idx, label in enumerate(labels)}
        for dataset, labels in labels_by_dataset.items()
    }

    paired_records: list[pd.DataFrame] = []
    summary_columns = [
        *AGG_KEYS,
        "n_paired",
        "wins_qaa",
        "wins_baseline",
        "discordant_pairs",
        "pvalue_accuracy",
        "significant_accuracy",
        "pvalue_macro_f1",
        "significant_macro_f1",
    ]

    for dataset in dataset_universe():
        baseline_dataset_dir = baseline_root / dataset
        qaa_dataset_dir = qaa_root / dataset
        if not baseline_dataset_dir.exists() or not qaa_dataset_dir.exists():
            continue

        for baseline_path in sorted(baseline_dataset_dir.glob("prediction_*.csv")):
            qaa_path = qaa_dataset_dir / baseline_path.name
            if not qaa_path.exists():
                continue

            match = PREDICTION_FILE_RE.match(baseline_path.name)
            if not match:
                continue

            modalities = (match.group("modalities") or "").lower()
            noise_modalities = (match.group("noise") or "").lower()

            if len(modalities) <= 1:
                continue
            if len(noise_modalities) != 1:
                continue
            if noise_modalities not in modalities:
                continue

            baseline_frame = read_prediction_file(baseline_path)
            qaa_frame = read_prediction_file(qaa_path)
            if baseline_frame.empty or qaa_frame.empty:
                continue

            join_columns = determine_join_columns(baseline_frame, qaa_frame)
            if not join_columns:
                continue

            baseline_prepared = deduplicate_by_join_key(
                preprocess_frame_for_pairing(baseline_frame, join_columns), join_columns
            )
            qaa_prepared = deduplicate_by_join_key(
                preprocess_frame_for_pairing(qaa_frame, join_columns), join_columns
            )
            if baseline_prepared.empty or qaa_prepared.empty:
                continue

            paired = baseline_prepared.merge(
                qaa_prepared,
                on=join_columns,
                how="inner",
                suffixes=("_baseline", "_qaa"),
            )
            if paired.empty:
                continue

            split_column = "split" if "split" in join_columns else "split_baseline"
            if split_column not in paired.columns:
                continue

            paired = paired[
                paired["label_norm_baseline"].eq(paired["label_norm_qaa"])
                & paired["label_norm_baseline"].ne("")
                & paired["label_norm_qaa"].ne("")
            ].copy()
            if paired.empty:
                continue

            split_meta = paired[split_column].map(parse_split_metadata).apply(pd.Series)
            paired["target"] = split_meta["target"]
            paired["method"] = split_meta["method"]
            paired["severity"] = split_meta["severity"]
            paired["is_unmodified"] = split_meta["is_unmodified"]
            paired = paired[
                (~paired["is_unmodified"].astype(bool))
                & paired["severity"].notna()
                & paired["target"].eq(noise_modalities)
            ].copy()
            if paired.empty:
                continue

            paired["dataset"] = dataset
            paired["modalities"] = "".join(sorted(modalities))
            paired["noise_modalities"] = noise_modalities
            paired["severity"] = paired["severity"].astype(int)
            paired["correct_baseline"] = paired["prediction_norm_baseline"].eq(paired["label_norm_baseline"])
            paired["correct_qaa"] = paired["prediction_norm_qaa"].eq(paired["label_norm_baseline"])
            paired_records.append(
                paired[
                    [
                        *AGG_KEYS,
                        "correct_baseline",
                        "correct_qaa",
                        "label_norm_baseline",
                        "prediction_norm_baseline",
                        "prediction_norm_qaa",
                    ]
                ].copy()
            )

    if not paired_records:
        return pd.DataFrame(columns=summary_columns)

    paired_all = pd.concat(paired_records, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    grouped = paired_all.groupby(AGG_KEYS, dropna=False, as_index=False)
    for key_values, group in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)

        correct_baseline = group["correct_baseline"].to_numpy(dtype=bool)
        correct_qaa = group["correct_qaa"].to_numpy(dtype=bool)
        wins_qaa = int((~correct_baseline & correct_qaa).sum())
        wins_baseline = int((correct_baseline & ~correct_qaa).sum())
        discordant_pairs = wins_qaa + wins_baseline
        pvalue = mcnemar_exact_pvalue(wins_qaa, wins_baseline)
        n_paired = int(len(group))
        dataset_name = str(group["dataset"].iloc[0])
        labels = labels_by_dataset.get(dataset_name, [])
        label_to_index = label_to_index_by_dataset.get(dataset_name, {})
        unknown_index = len(labels)
        if labels and dataset_name != "nejm" and macro_f1_permutations > 0:
            y_true_idx = encode_labels(group["label_norm_baseline"].to_numpy(dtype=object), label_to_index, unknown_index)
            y_pred_baseline_idx = encode_labels(
                group["prediction_norm_baseline"].to_numpy(dtype=object), label_to_index, unknown_index
            )
            y_pred_qaa_idx = encode_labels(group["prediction_norm_qaa"].to_numpy(dtype=object), label_to_index, unknown_index)
            pvalue_macro_f1 = paired_permutation_macro_f1_pvalue(
                y_true_idx=y_true_idx,
                y_pred_baseline_idx=y_pred_baseline_idx,
                y_pred_qaa_idx=y_pred_qaa_idx,
                n_labels=len(labels),
                permutations=macro_f1_permutations,
                rng=rng,
            )
        else:
            pvalue_macro_f1 = float("nan")

        row = dict(zip(AGG_KEYS, key_values))
        row.update(
            {
                "n_paired": n_paired,
                "wins_qaa": wins_qaa,
                "wins_baseline": wins_baseline,
                "discordant_pairs": discordant_pairs,
                "pvalue_accuracy": pvalue,
                "significant_accuracy": bool(n_paired >= min_paired and pvalue < alpha),
                "pvalue_macro_f1": pvalue_macro_f1,
                "significant_macro_f1": bool(
                    n_paired >= min_paired and pd.notna(pvalue_macro_f1) and float(pvalue_macro_f1) < alpha
                ),
            }
        )
        summary_rows.append(row)

    return pd.DataFrame.from_records(summary_rows, columns=summary_columns)


def compute_qaa_minus_baseline(
    baseline_root: Path,
    qaa_root: Path,
    model_name: str,
    significance_alpha: float,
    significance_min_paired: int,
    macro_f1_permutations: int,
    significance_seed: int,
) -> pd.DataFrame:
    labels_by_dataset = collect_dataset_labels(baseline_root=baseline_root, qaa_root=qaa_root)
    baseline_records = collect_multimodal_noisy_metrics_for_root(
        model_root=baseline_root,
        labels_by_dataset=labels_by_dataset,
        source_name="baseline",
    )
    qaa_records = collect_multimodal_noisy_metrics_for_root(
        model_root=qaa_root,
        labels_by_dataset=labels_by_dataset,
        source_name="qaa",
    )

    if baseline_records.empty or qaa_records.empty:
        return pd.DataFrame()

    significance_records = collect_paired_significance(
        baseline_root=baseline_root,
        qaa_root=qaa_root,
        labels_by_dataset=labels_by_dataset,
        alpha=significance_alpha,
        min_paired=significance_min_paired,
        macro_f1_permutations=macro_f1_permutations,
        significance_seed=significance_seed,
    )

    # Some QAA files use prefixed split names (e.g., "test_V=...") while baseline may use "V=...".
    # Aggregate by parsed condition keys and merge on those keys instead of raw split text.
    baseline_agg = (
        baseline_records.groupby(AGG_KEYS, as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_baseline", "macro_f1": "macro_f1_baseline"})
    )
    qaa_agg = (
        qaa_records.groupby(AGG_KEYS, as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_qaa", "macro_f1": "macro_f1_qaa"})
    )

    merged = baseline_agg.merge(
        qaa_agg,
        on=AGG_KEYS,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    if not significance_records.empty:
        merged = merged.merge(significance_records, on=AGG_KEYS, how="left")
    else:
        merged["n_paired"] = 0
        merged["wins_qaa"] = 0
        merged["wins_baseline"] = 0
        merged["discordant_pairs"] = 0
        merged["pvalue_accuracy"] = np.nan
        merged["significant_accuracy"] = False
        merged["pvalue_macro_f1"] = np.nan
        merged["significant_macro_f1"] = False

    merged["delta_accuracy"] = merged["accuracy_qaa"] - merged["accuracy_baseline"]
    merged["delta_macro_f1"] = merged["macro_f1_qaa"] - merged["macro_f1_baseline"]
    merged["n_paired"] = merged["n_paired"].fillna(0).astype(int)
    merged["wins_qaa"] = merged["wins_qaa"].fillna(0).astype(int)
    merged["wins_baseline"] = merged["wins_baseline"].fillna(0).astype(int)
    merged["discordant_pairs"] = merged["discordant_pairs"].fillna(0).astype(int)
    merged["significant_accuracy"] = merged["significant_accuracy"].fillna(False).astype(bool)
    merged["significant_macro_f1"] = merged["significant_macro_f1"].fillna(False).astype(bool)
    merged["model"] = model_name
    return merged


def compute_qaa_minus_baseline_legacy(
    baseline_root: Path,
    qaa_root: Path,
) -> pd.DataFrame:
    labels_by_dataset = collect_dataset_labels(baseline_root=baseline_root, qaa_root=qaa_root)
    baseline_records = collect_multimodal_noisy_metrics_for_root(
        model_root=baseline_root,
        labels_by_dataset=labels_by_dataset,
        source_name="baseline",
    )
    qaa_records = collect_multimodal_noisy_metrics_for_root(
        model_root=qaa_root,
        labels_by_dataset=labels_by_dataset,
        source_name="qaa",
    )

    if baseline_records.empty or qaa_records.empty:
        return pd.DataFrame()

    merge_keys = ["dataset", "modalities", "noise_modalities", "target", "method", "severity", "split"]
    merged = baseline_records.merge(
        qaa_records,
        on=merge_keys,
        how="inner",
        suffixes=("_baseline", "_qaa"),
    )

    if merged.empty:
        return pd.DataFrame()

    merged["delta_accuracy"] = merged["accuracy_qaa"] - merged["accuracy_baseline"]
    merged["delta_macro_f1"] = merged["macro_f1_qaa"] - merged["macro_f1_baseline"]
    return merged


def ordered_methods(records: pd.DataFrame, modality: str) -> list[str]:
    methods = sorted(records.loc[records["target"] == modality, "method"].dropna().astype(str).unique().tolist())
    hint = METHOD_ORDER_HINTS.get(modality, [])
    in_hint = [method for method in hint if method in methods]
    extra = [method for method in methods if method not in in_hint]
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


def format_modalities_pair(pair: str) -> str:
    return "+".join(MODALITY_SHORT.get(modality, modality.upper()) for modality in str(pair))


def build_dataset_tick_labels(datasets: list[str], subset: pd.DataFrame) -> list[str]:
    pair_map: dict[str, list[str]] = {}
    if not subset.empty and "modalities" in subset.columns:
        per_dataset = (
            subset.dropna(subset=["modalities"])
            .groupby("dataset")["modalities"]
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


def write_csv_outputs(records: pd.DataFrame, csv_dir: Path) -> tuple[Path, Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "model",
        "dataset",
        "modalities",
        "noise_modalities",
        "target",
        "method",
        "severity",
        "split",
        "n_paired",
        "wins_qaa",
        "wins_baseline",
        "discordant_pairs",
        "pvalue_accuracy",
        "significant_accuracy",
        "pvalue_macro_f1",
        "significant_macro_f1",
        "accuracy_baseline",
        "accuracy_qaa",
        "delta_accuracy",
        "macro_f1_baseline",
        "macro_f1_qaa",
        "delta_macro_f1",
    ]
    available_cols = [col for col in base_cols if col in records.columns]
    export_df = records[available_cols].copy()
    export_df["dataset_label"] = export_df["dataset"].map(lambda dataset: DATASET_LABEL.get(dataset, dataset))
    export_df["target_label"] = export_df["target"].map(lambda modality: MODALITY_LABEL.get(modality, modality))
    export_df["method_label"] = export_df["method"].map(format_method_tick_label)

    accuracy_path = csv_dir / "qaa_vs_baseline_multimodal_delta_accuracy.csv"
    macro_f1_path = csv_dir / "qaa_vs_baseline_multimodal_delta_macro_f1.csv"

    accuracy_cols = [
        "model",
        "dataset",
        "dataset_label",
        "modalities",
        "noise_modalities",
        "target",
        "target_label",
        "method",
        "method_label",
        "severity",
        "n_paired",
        "pvalue_accuracy",
        "significant_accuracy",
        "pvalue_macro_f1",
        "significant_macro_f1",
        "accuracy_baseline",
        "accuracy_qaa",
        "delta_accuracy",
    ]
    macro_cols = [
        "model",
        "dataset",
        "dataset_label",
        "modalities",
        "noise_modalities",
        "target",
        "target_label",
        "method",
        "method_label",
        "severity",
        "n_paired",
        "pvalue_accuracy",
        "significant_accuracy",
        "pvalue_macro_f1",
        "significant_macro_f1",
        "macro_f1_baseline",
        "macro_f1_qaa",
        "delta_macro_f1",
    ]
    export_df[[col for col in accuracy_cols if col in export_df.columns]].to_csv(accuracy_path, index=False)
    export_df[[col for col in macro_cols if col in export_df.columns]].to_csv(macro_f1_path, index=False)
    return accuracy_path, macro_f1_path


def plot_metric_page(
    records: pd.DataFrame,
    models: list[str],
    severities: list[int],
    metric_col: str,
    output_path: Path,
    include_nejm: bool,
    significance_alpha: float,
    significance_min_paired: int,
    macro_f1_permutations: int,
    dpi: int,
) -> None:
    row_modalities = MODALITY_ROWS
    if records.empty:
        return

    figure_records = records.copy()
    if not include_nejm:
        figure_records = figure_records[figure_records["dataset"] != "nejm"].copy()

    base_cmap = sns.color_palette("coolwarm", as_cmap=True) if sns is not None else plt.get_cmap("coolwarm")
    # Keep linear value scaling, but make near-zero colors change faster.
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

    column_specs = [(model, severity) for model in models for severity in severities]
    fig, axes = plt.subplots(
        nrows=len(row_modalities),
        ncols=len(column_specs),
        figsize=(2.4 * len(column_specs) + 2.4, 4.8 * len(row_modalities)),
        sharey="row",
        constrained_layout=False,
    )

    # Ensure 2D indexing for single-column scenarios.
    if len(column_specs) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, modality in enumerate(row_modalities):
        methods = ordered_methods(figure_records, modality)
        datasets = dataset_order_for_modality(modality, include_nejm=include_nejm)

        for col_idx, (model, severity) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            subset = figure_records[
                (figure_records["model"] == model)
                & (figure_records["target"] == modality)
                & (figure_records["severity"] == severity)
            ]

            metric_matrix = (
                subset.pivot_table(index="dataset", columns="method", values=metric_col, aggfunc="mean")
                .reindex(index=datasets, columns=methods)
            )

            if metric_matrix.empty:
                metric_matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)

            annotations = metric_matrix.copy()
            annotations = annotations.map(lambda value: "" if pd.isna(value) else f"{value:+.1f}")
            significance_col = "significant_accuracy" if metric_col == "delta_accuracy" else "significant_macro_f1"
            significance_matrix = pd.DataFrame(index=datasets, columns=methods, dtype=object)
            if significance_col in subset.columns:
                significance_matrix = (
                    subset.pivot_table(index="dataset", columns="method", values=significance_col, aggfunc="max")
                    .reindex(index=datasets, columns=methods)
                )

            if len(datasets) == 0 or len(methods) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            plot_values = metric_matrix.to_numpy(dtype=float)
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
                        is_significant = False
                        if method in significance_matrix.columns and dataset_name in significance_matrix.index:
                            sig_value = significance_matrix.at[dataset_name, method]
                            is_significant = bool(sig_value) if pd.notna(sig_value) else False
                        ax.text(
                            x_idx,
                            y_idx,
                            text,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="black",
                            fontweight="bold" if is_significant else "normal",
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
    cbar.set_label("QAA - Baseline (pp)", fontsize=14, labelpad=2)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=12)

    metric_title = "Accuracy" if metric_col == "delta_accuracy" else "Macro F1"
    fig.suptitle("")
    fig.supxlabel("Dataset", fontsize=15, y=0.06)
    fig.subplots_adjust(left=0.12, right=0.882, top=0.900, bottom=0.155, wspace=0.08, hspace=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_dir = Path(args.csv_dir)

    record_frames: list[pd.DataFrame] = []
    for model in args.models:
        baseline_root = Path("../out") / model
        qaa_root = baseline_root / args.qaa_subdir
        if not baseline_root.exists() or not qaa_root.exists():
            print(f"[WARN] Skipping {model}: missing baseline or qaa root.")
            continue

        model_records = compute_qaa_minus_baseline(
            baseline_root=baseline_root,
            qaa_root=qaa_root,
            model_name=model,
            significance_alpha=args.significance_alpha,
            significance_min_paired=args.significance_min_paired,
            macro_f1_permutations=args.macro_f1_permutations,
            significance_seed=args.significance_seed,
        )
        if model_records.empty:
            print(f"[WARN] No overlapping noisy multimodal records found for {model}.")
            continue
        record_frames.append(model_records)

    if not record_frames:
        raise RuntimeError("No overlapping noisy multimodal records found for any requested model.")

    records = pd.concat(record_frames, ignore_index=True)
    available_models = [model for model in args.models if model in set(records["model"].astype(str).unique())]

    csv_accuracy, csv_macro_f1 = write_csv_outputs(records=records, csv_dir=csv_dir)

    accuracy_heatmap = output_dir / "qaa_vs_baseline_multimodal_delta_heatmap_accuracy.png"
    macro_f1_heatmap = output_dir / "qaa_vs_baseline_multimodal_delta_heatmap_macro_f1.png"

    plot_metric_page(
        records=records,
        models=available_models,
        severities=args.severities,
        metric_col="delta_accuracy",
        output_path=accuracy_heatmap,
        include_nejm=True,
        significance_alpha=args.significance_alpha,
        significance_min_paired=args.significance_min_paired,
        macro_f1_permutations=args.macro_f1_permutations,
        dpi=args.dpi,
    )
    plot_metric_page(
        records=records,
        models=available_models,
        severities=args.severities,
        metric_col="delta_macro_f1",
        output_path=macro_f1_heatmap,
        include_nejm=False,
        significance_alpha=args.significance_alpha,
        significance_min_paired=args.significance_min_paired,
        macro_f1_permutations=args.macro_f1_permutations,
        dpi=args.dpi,
    )

    print(f"Wrote CSV (Accuracy) to {csv_accuracy}")
    print(f"Wrote CSV (Macro-F1) to {csv_macro_f1}")
    print(f"Wrote heatmap (Accuracy) to {accuracy_heatmap}")
    print(f"Wrote heatmap (Macro-F1) to {macro_f1_heatmap}")


if __name__ == "__main__":
    main()
