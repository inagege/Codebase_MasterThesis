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

PREDICTION_FILE_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)", re.IGNORECASE)
SPLIT_NOISE_RE = re.compile(r"(?P<target>[A-Za-z])=(?P<method>.+?)_S=(?P<severity>\d+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
AGG_KEYS = ["dataset", "modalities", "noise_modalities", "target", "method", "severity"]
JOIN_COLUMN_PRIORITY = ["dataset", "split", "sample_id", "file"]

DATASET_LABEL = {
    "imdb": "IMDB",
    "sentiment": "MELD Sentiment",
    "emotion": "MELD Emotion",
}
MODALITY_LABEL = {"a": "Audio", "i": "Image", "t": "Text", "v": "Video"}
MODALITY_ORDER = ["a", "i", "t", "v"]
METHOD_ORDER_HINTS = {
    "a": ["bandlimit", "bitcrushing", "clipping", "compress", "jitter", "mp3", "reverb", "snr_white"],
    "i": ["gaussian_noise", "jpeg", "motion_blur", "occlusion", "pixelate", "scale_down", "zoom_blur"],
    "t": ["char_delete", "char_replace", "keyboard", "ocr", "synonym_replace", "top4_paper"],
    "v": ["fps_drop", "gaussian_noise", "motion_blur", "moving_occlusion", "occlusion", "pixelate", "scale_down", "zoom_blur"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create severity-3 heatmaps for qwen_scored/placebo vs baseline across multimodal datasets "
            "with corruption-method splits of the modified modality."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen_3B", "Qwen_7B"],
        help="Model folders under out/ to compare (default: Qwen_3B Qwen_7B).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["qwen_scored", "placebo"],
        help="Source folders under out/<model>/ to compare against baseline (default: qwen_scored placebo).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sentiment", "emotion"],
        help="Datasets to include (default: sentiment emotion).",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Severity to include (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/out/plots/perturbation_heatmaps",
        help="Directory where heatmaps are saved.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="analysis/out/csv/qwen_scored_placebo_vs_baseline_sentiment_emotion_s3.csv",
        help="CSV path for merged source-vs-baseline records.",
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
        help="P-value threshold for qwen_scored-vs-placebo significance markers (default: 0.05).",
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


def source_display_label(source: str) -> str:
    source_norm = str(source).strip().casefold()
    if source_norm == "qwen_scored":
        return "Qwen-scored"
    if source_norm == "qwen_scored_rescaled":
        return "Qwen-scored rescaled"
    if source_norm == "placebo":
        return "Placebo"
    return str(source).replace("_", " ")


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


def determine_join_columns(df_left: pd.DataFrame, df_right: pd.DataFrame) -> list[str]:
    columns = [col for col in JOIN_COLUMN_PRIORITY if col in df_left.columns and col in df_right.columns]
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


def preprocess_frame_for_pairing(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    processed = frame.copy()
    for col in join_columns:
        if col == "split":
            processed[col] = processed[col].map(normalize_split_for_join)
        else:
            processed[col] = processed[col].map(normalize_text)
    return processed


def deduplicate_by_join_key(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    if not join_columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=join_columns, keep="first")


def mcnemar_exact_pvalue(wins_a: int, wins_b: int) -> float:
    discordant = wins_a + wins_b
    if discordant <= 0:
        return 1.0

    smaller_tail = min(wins_a, wins_b)
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
    values_list = values.tolist()
    return np.fromiter((label_to_index.get(value, unknown_index) for value in values_list), dtype=np.int32, count=len(values_list))


def macro_f1_percent_from_encoded(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_labels: int) -> float:
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
    y_pred_a_idx: np.ndarray,
    y_pred_b_idx: np.ndarray,
    n_labels: int,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if permutations <= 0 or y_true_idx.size == 0 or n_labels <= 0:
        return float("nan")

    observed_delta = (
        macro_f1_percent_from_encoded(y_true_idx, y_pred_b_idx, n_labels)
        - macro_f1_percent_from_encoded(y_true_idx, y_pred_a_idx, n_labels)
    )
    if not math.isfinite(observed_delta):
        return float("nan")

    abs_observed = abs(observed_delta)
    extreme_count = 0
    for _ in range(permutations):
        swap_mask = rng.random(y_true_idx.size) < 0.5
        perm_a = np.where(swap_mask, y_pred_b_idx, y_pred_a_idx)
        perm_b = np.where(swap_mask, y_pred_a_idx, y_pred_b_idx)
        delta = (
            macro_f1_percent_from_encoded(y_true_idx, perm_b, n_labels)
            - macro_f1_percent_from_encoded(y_true_idx, perm_a, n_labels)
        )
        if abs(delta) >= abs_observed - 1e-12:
            extreme_count += 1

    return float((extreme_count + 1) / (permutations + 1))


def parse_prediction_filename(path: Path) -> tuple[str, str]:
    match = PREDICTION_FILE_RE.match(path.name)
    if not match:
        return "", ""
    modalities = (match.group("modalities") or "").lower()
    noise_modalities = (match.group("noise") or "").lower()
    return modalities, noise_modalities


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


def read_prediction_file_for_pairing(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"split", "prediction", "label"}.issubset(frame.columns):
        return pd.DataFrame()

    frame = frame.dropna(subset=["split", "prediction", "label"]).copy()
    frame["prediction_norm"] = frame["prediction"].map(normalize_value)
    frame["label_norm"] = frame["label"].map(normalize_value)
    frame = frame[(frame["label_norm"] != "") & (frame["label_norm"] != "unknown")].copy()
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


def has_prediction_files(directory: Path) -> bool:
    return directory.exists() and directory.is_dir() and any(directory.glob("prediction_*.csv"))


def collect_label_set_for_dir(directory: Path) -> set[str]:
    labels: set[str] = set()
    if not has_prediction_files(directory):
        return labels

    for path in sorted(directory.glob("prediction_*.csv")):
        modalities, _ = parse_prediction_filename(path)
        if len(modalities) <= 1:
            continue
        frame = read_prediction_file(path)
        labels.update(frame["label_norm"].tolist())
    return labels


def label_overlap_score(candidate_labels: set[str], baseline_labels: set[str]) -> float:
    baseline_clean = {label for label in baseline_labels if label != "unknown"}
    candidate_clean = {label for label in candidate_labels if label != "unknown"}
    if not baseline_clean:
        return 0.0
    return len(baseline_clean & candidate_clean) / len(baseline_clean)


def resolve_source_dataset_dir(
    source_root: Path,
    dataset: str,
    baseline_labels: set[str],
) -> tuple[Path | None, str | None]:
    direct = source_root / dataset
    direct_exists = has_prediction_files(direct)
    candidates: list[Path] = []
    if direct_exists:
        candidates.append(direct)

    if source_root.exists() and source_root.is_dir():
        for path in sorted(source_root.iterdir()):
            if not path.is_dir() or path == direct:
                continue
            if has_prediction_files(path):
                candidates.append(path)

    if not candidates:
        return None, f"missing source dataset directory under {source_root} for '{dataset}'"

    if not baseline_labels:
        chosen = direct if direct_exists else candidates[0]
        if not direct_exists and chosen.name.casefold() != dataset.casefold():
            return chosen, f"using {chosen.name} for dataset '{dataset}' (direct folder missing)"
        return chosen, None

    scored: list[tuple[float, Path]] = []
    for path in candidates:
        labels = collect_label_set_for_dir(path)
        scored.append((label_overlap_score(labels, baseline_labels), path))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = scored[0]

    if best_score < 0.5:
        if direct_exists:
            return direct, f"low label-overlap for '{dataset}' in {source_root} (best score {best_score:.2f}); using direct path"
        return None, f"low label-overlap for '{dataset}' in {source_root} (best score {best_score:.2f}); skipping dataset"

    if direct_exists and best_path != direct:
        return best_path, f"using {best_path.name} for dataset '{dataset}' (label-overlap match)"

    if not direct_exists and best_path.name.casefold() != dataset.casefold():
        return best_path, f"using {best_path.name} for dataset '{dataset}' (label-overlap match)"

    return best_path, None


def collect_baseline_label_sets(model_root: Path, datasets: list[str]) -> dict[str, set[str]]:
    labels_by_dataset: dict[str, set[str]] = {dataset: set() for dataset in datasets}
    for dataset in datasets:
        dataset_dir = model_root / dataset
        labels_by_dataset[dataset] = collect_label_set_for_dir(dataset_dir)
    return labels_by_dataset


def collect_multimodal_noisy_metrics_for_dir(
    dataset: str,
    dataset_dir: Path,
    labels: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if not has_prediction_files(dataset_dir):
        return pd.DataFrame.from_records(records)

    for path in sorted(dataset_dir.glob("prediction_*.csv")):
        modalities, noise_modalities = parse_prediction_filename(path)
        if not modalities:
            continue
        if len(modalities) <= 1:
            continue
        if len(noise_modalities) != 1:
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
            macro_f1 = float(macro_f1_score(split_frame["label_norm"], split_frame["prediction_norm"], labels) * 100.0)
            records.append(
                {
                    "dataset": dataset,
                    "modalities": "".join(sorted(modalities)),
                    "noise_modalities": noise_modalities,
                    "target": split_meta["target"],
                    "method": split_meta["method"],
                    "severity": int(split_meta["severity"]),
                    "split": split,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )

    return pd.DataFrame.from_records(records)


def build_prediction_file_index(dataset_dir: Path) -> dict[tuple[str, str], list[Path]]:
    indexed: dict[tuple[str, str], list[Path]] = {}
    if not has_prediction_files(dataset_dir):
        return indexed

    for path in sorted(dataset_dir.glob("prediction_*.csv")):
        modalities, noise_modalities = parse_prediction_filename(path)
        if not modalities:
            continue
        if len(modalities) <= 1:
            continue
        if len(noise_modalities) != 1:
            continue
        if noise_modalities not in modalities:
            continue
        indexed.setdefault((modalities, noise_modalities), []).append(path)
    return indexed


def collect_qwen_placebo_significance_for_model(
    model_root: Path,
    model: str,
    datasets: list[str],
    baseline_label_sets: dict[str, set[str]],
    alpha: float,
    min_paired: int,
    macro_f1_permutations: int,
    significance_seed: int,
    source_a: str = "qwen_scored",
    source_b: str = "placebo",
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    source_a_root = model_root / source_a
    source_b_root = model_root / source_b
    if not source_a_root.exists() or not source_b_root.exists():
        notes.append(f"[INFO] Missing {source_a}/{source_b} root for significance in {model}; skipping markers.")
        return pd.DataFrame(), notes

    rng = np.random.default_rng(significance_seed)
    labels_by_dataset: dict[str, list[str]] = {}
    paired_records: list[pd.DataFrame] = []

    for dataset in datasets:
        baseline_labels = baseline_label_sets.get(dataset, set())
        dir_a, note_a = resolve_source_dataset_dir(
            source_root=source_a_root,
            dataset=dataset,
            baseline_labels=baseline_labels,
        )
        dir_b, note_b = resolve_source_dataset_dir(
            source_root=source_b_root,
            dataset=dataset,
            baseline_labels=baseline_labels,
        )
        if note_a:
            notes.append(f"[INFO] {model}/{source_a}/{dataset}: {note_a}")
        if note_b:
            notes.append(f"[INFO] {model}/{source_b}/{dataset}: {note_b}")
        if dir_a is None or dir_b is None:
            continue

        labels_by_dataset[dataset] = sorted(
            collect_label_set_for_dir(dir_a) | collect_label_set_for_dir(dir_b) | baseline_labels
        )
        index_a = build_prediction_file_index(dir_a)
        index_b = build_prediction_file_index(dir_b)
        common_keys = sorted(set(index_a.keys()) & set(index_b.keys()))
        if not common_keys:
            notes.append(f"[WARN] No overlapping prediction settings for significance in {model}/{dataset}.")
            continue

        for modalities, noise_modalities in common_keys:
            paths_a = index_a.get((modalities, noise_modalities), [])
            paths_b = index_b.get((modalities, noise_modalities), [])
            pair_count = min(len(paths_a), len(paths_b))
            if pair_count == 0:
                continue
            if len(paths_a) != len(paths_b):
                notes.append(
                    f"[INFO] Uneven file counts for {model}/{dataset}/{modalities}_noise_{noise_modalities} "
                    f"({source_a}={len(paths_a)}, {source_b}={len(paths_b)}); using first {pair_count} pair(s)."
                )

            for idx in range(pair_count):
                frame_a = read_prediction_file_for_pairing(paths_a[idx])
                frame_b = read_prediction_file_for_pairing(paths_b[idx])
                if frame_a.empty or frame_b.empty:
                    continue

                join_columns = determine_join_columns(frame_a, frame_b)
                if not join_columns:
                    continue

                a_prepared = deduplicate_by_join_key(
                    preprocess_frame_for_pairing(frame_a, join_columns), join_columns
                )
                b_prepared = deduplicate_by_join_key(
                    preprocess_frame_for_pairing(frame_b, join_columns), join_columns
                )
                if a_prepared.empty or b_prepared.empty:
                    continue

                paired = a_prepared.merge(
                    b_prepared,
                    on=join_columns,
                    how="inner",
                    suffixes=("_a", "_b"),
                )
                if paired.empty:
                    continue

                split_column = "split" if "split" in join_columns else "split_a"
                if split_column not in paired.columns:
                    continue

                paired = paired[
                    paired["label_norm_a"].eq(paired["label_norm_b"])
                    & paired["label_norm_a"].ne("")
                    & paired["label_norm_b"].ne("")
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
                paired["correct_a"] = paired["prediction_norm_a"].eq(paired["label_norm_a"])
                paired["correct_b"] = paired["prediction_norm_b"].eq(paired["label_norm_a"])
                paired_records.append(
                    paired[
                        [
                            *AGG_KEYS,
                            "correct_a",
                            "correct_b",
                            "label_norm_a",
                            "prediction_norm_a",
                            "prediction_norm_b",
                        ]
                    ].copy()
                )

    if not paired_records:
        return pd.DataFrame(), notes

    label_to_index_by_dataset = {
        dataset: {label: i for i, label in enumerate(labels)}
        for dataset, labels in labels_by_dataset.items()
    }
    paired_all = pd.concat(paired_records, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for key_values, group in paired_all.groupby(AGG_KEYS, as_index=False, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)

        correct_a = group["correct_a"].to_numpy(dtype=bool)
        correct_b = group["correct_b"].to_numpy(dtype=bool)
        wins_b = int((~correct_a & correct_b).sum())
        wins_a = int((correct_a & ~correct_b).sum())
        n_paired = int(len(group))
        pvalue_accuracy = mcnemar_exact_pvalue(wins_a=wins_a, wins_b=wins_b)

        dataset_name = str(group["dataset"].iloc[0])
        labels = labels_by_dataset.get(dataset_name, [])
        label_to_index = label_to_index_by_dataset.get(dataset_name, {})
        unknown_index = len(labels)
        if labels and dataset_name != "nejm" and macro_f1_permutations > 0:
            y_true_idx = encode_labels(group["label_norm_a"].to_numpy(dtype=object), label_to_index, unknown_index)
            y_pred_a_idx = encode_labels(group["prediction_norm_a"].to_numpy(dtype=object), label_to_index, unknown_index)
            y_pred_b_idx = encode_labels(group["prediction_norm_b"].to_numpy(dtype=object), label_to_index, unknown_index)
            pvalue_macro_f1 = paired_permutation_macro_f1_pvalue(
                y_true_idx=y_true_idx,
                y_pred_a_idx=y_pred_a_idx,
                y_pred_b_idx=y_pred_b_idx,
                n_labels=len(labels),
                permutations=macro_f1_permutations,
                rng=rng,
            )
        else:
            pvalue_macro_f1 = float("nan")

        row = dict(zip(AGG_KEYS, key_values))
        row.update(
            {
                "n_paired_qwen_placebo": n_paired,
                "wins_qwen_scored": wins_a,
                "wins_placebo": wins_b,
                "pvalue_qwen_placebo_accuracy": pvalue_accuracy,
                "significant_accuracy": bool(n_paired >= min_paired and pvalue_accuracy < alpha),
                "pvalue_qwen_placebo_macro_f1": pvalue_macro_f1,
                "significant_macro_f1": bool(
                    n_paired >= min_paired and pd.notna(pvalue_macro_f1) and float(pvalue_macro_f1) < alpha
                ),
                "model": model,
            }
        )
        summary_rows.append(row)

    return pd.DataFrame.from_records(summary_rows), notes


def compute_source_vs_baseline_for_model(
    model: str,
    sources: list[str],
    datasets: list[str],
    significance_alpha: float,
    significance_min_paired: int,
    macro_f1_permutations: int,
    significance_seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    model_root = Path("../out") / model
    if not model_root.exists():
        notes.append(f"[WARN] Missing model root: {model_root}")
        return pd.DataFrame(), notes

    baseline_label_sets = collect_baseline_label_sets(model_root=model_root, datasets=datasets)
    baseline_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        labels = sorted(baseline_label_sets.get(dataset, set()))
        baseline_dir = model_root / dataset
        if not has_prediction_files(baseline_dir):
            notes.append(f"[WARN] Missing baseline dataset directory: {baseline_dir}")
            continue
        frame = collect_multimodal_noisy_metrics_for_dir(dataset=dataset, dataset_dir=baseline_dir, labels=labels)
        if frame.empty:
            notes.append(f"[WARN] No baseline multimodal perturbation rows for {model}/{dataset}")
            continue
        baseline_frames.append(frame)

    if not baseline_frames:
        notes.append(f"[WARN] No baseline records collected for {model}")
        return pd.DataFrame(), notes

    baseline_records = pd.concat(baseline_frames, ignore_index=True)
    baseline_agg = (
        baseline_records.groupby(AGG_KEYS, as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_baseline", "macro_f1": "macro_f1_baseline"})
    )

    source_records_all: list[pd.DataFrame] = []
    for source in sources:
        source_root = model_root / source
        if not source_root.exists():
            notes.append(f"[WARN] Missing source root: {source_root}")
            continue

        source_frames: list[pd.DataFrame] = []
        for dataset in datasets:
            baseline_labels = baseline_label_sets.get(dataset, set())
            resolved_dir, resolve_note = resolve_source_dataset_dir(
                source_root=source_root,
                dataset=dataset,
                baseline_labels=baseline_labels,
            )
            if resolve_note:
                notes.append(f"[INFO] {model}/{source}/{dataset}: {resolve_note}")
            if resolved_dir is None:
                continue

            labels = sorted(collect_label_set_for_dir(resolved_dir) | baseline_labels)
            frame = collect_multimodal_noisy_metrics_for_dir(dataset=dataset, dataset_dir=resolved_dir, labels=labels)
            if frame.empty:
                notes.append(f"[WARN] No multimodal perturbation rows for {model}/{source}/{dataset}")
                continue
            source_frames.append(frame)

        if not source_frames:
            notes.append(f"[WARN] No records for source {model}/{source}")
            continue

        source_records = pd.concat(source_frames, ignore_index=True)
        source_agg = (
            source_records.groupby(AGG_KEYS, as_index=False)[["accuracy", "macro_f1"]]
            .mean()
            .rename(columns={"accuracy": "accuracy_source", "macro_f1": "macro_f1_source"})
        )

        merged = baseline_agg.merge(source_agg, on=AGG_KEYS, how="inner")
        if merged.empty:
            notes.append(f"[WARN] No overlapping perturbation groups for {model}/{source}")
            continue

        merged["delta_accuracy"] = merged["accuracy_source"] - merged["accuracy_baseline"]
        merged["delta_macro_f1"] = merged["macro_f1_source"] - merged["macro_f1_baseline"]
        merged["model"] = model
        merged["source"] = source
        source_records_all.append(merged)

    if not source_records_all:
        return pd.DataFrame(), notes

    merged_records = pd.concat(source_records_all, ignore_index=True)
    significance_source_a: str | None = None
    source_set = set(sources)
    if "placebo" in source_set:
        for candidate in ("qwen_scored_rescaled", "qwen_scored"):
            if candidate in source_set:
                significance_source_a = candidate
                break
        if significance_source_a is None:
            for source in sources:
                if source != "placebo":
                    significance_source_a = source
                    break

    if significance_source_a is not None:
        significance_records, significance_notes = collect_qwen_placebo_significance_for_model(
            model_root=model_root,
            model=model,
            datasets=datasets,
            baseline_label_sets=baseline_label_sets,
            alpha=significance_alpha,
            min_paired=significance_min_paired,
            macro_f1_permutations=macro_f1_permutations,
            significance_seed=significance_seed,
            source_a=significance_source_a,
            source_b="placebo",
        )
        if significance_source_a != "qwen_scored":
            notes.append(
                f"[INFO] {model}: significance markers compare {significance_source_a} vs placebo "
                "(stored in legacy qwen/placebo significance columns)."
            )
        notes.extend(significance_notes)
        if not significance_records.empty:
            merge_cols = ["model", *AGG_KEYS]
            merged_records = merged_records.merge(
                significance_records[
                    [
                        *merge_cols,
                        "n_paired_qwen_placebo",
                        "wins_qwen_scored",
                        "wins_placebo",
                        "pvalue_qwen_placebo_accuracy",
                        "significant_accuracy",
                        "pvalue_qwen_placebo_macro_f1",
                        "significant_macro_f1",
                    ]
                ],
                on=merge_cols,
                how="left",
            )

    if "n_paired_qwen_placebo" not in merged_records.columns:
        merged_records["n_paired_qwen_placebo"] = 0
    if "wins_qwen_scored" not in merged_records.columns:
        merged_records["wins_qwen_scored"] = 0
    if "wins_placebo" not in merged_records.columns:
        merged_records["wins_placebo"] = 0
    if "pvalue_qwen_placebo_accuracy" not in merged_records.columns:
        merged_records["pvalue_qwen_placebo_accuracy"] = np.nan
    if "significant_accuracy" not in merged_records.columns:
        merged_records["significant_accuracy"] = False
    if "pvalue_qwen_placebo_macro_f1" not in merged_records.columns:
        merged_records["pvalue_qwen_placebo_macro_f1"] = np.nan
    if "significant_macro_f1" not in merged_records.columns:
        merged_records["significant_macro_f1"] = False

    merged_records["n_paired_qwen_placebo"] = merged_records["n_paired_qwen_placebo"].fillna(0).astype(int)
    merged_records["wins_qwen_scored"] = merged_records["wins_qwen_scored"].fillna(0).astype(int)
    merged_records["wins_placebo"] = merged_records["wins_placebo"].fillna(0).astype(int)
    merged_records["significant_accuracy"] = merged_records["significant_accuracy"].astype("boolean").fillna(False).astype(bool)
    merged_records["significant_macro_f1"] = (
        merged_records["significant_macro_f1"].astype("boolean").fillna(False).astype(bool)
    )
    return merged_records, notes


def ordered_methods(records: pd.DataFrame, modality: str) -> list[str]:
    methods = sorted(records.loc[records["target"] == modality, "method"].dropna().astype(str).unique().tolist())
    hint = METHOD_ORDER_HINTS.get(modality, [])
    in_hint = [method for method in hint if method in methods]
    extra = [method for method in methods if method not in in_hint]
    return in_hint + extra


def format_method_tick_label(method: str) -> str:
    label = str(method).replace("_", " ")
    if label == "gaussian noise":
        return "uniform noise"
    if label == "occlusion":
        return "static occlusion"
    return label


def format_column_header(model: str, source: str) -> str:
    model_label = str(model).replace("_", " ")
    source_label = source_display_label(source)
    return f"{model_label}\n{source_label}"


def severity_name(severity: int) -> int:
    severity_map = {3: 1, 5: 2}
    return int(severity_map.get(int(severity), int(severity)))


def plot_metric_heatmap(
    records: pd.DataFrame,
    models: list[str],
    sources: list[str],
    datasets: list[str],
    severity: int,
    metric_col: str,
    output_path: Path,
    dpi: int,
) -> None:
    if records.empty:
        return

    available_targets = [str(value) for value in records["target"].dropna().astype(str).unique().tolist()]
    row_modalities = [modality for modality in MODALITY_ORDER if modality in set(available_targets)]
    row_modalities.extend(
        sorted([modality for modality in available_targets if modality not in set(MODALITY_ORDER)])
    )
    if not row_modalities:
        return
    column_specs = [(model, source) for model in models for source in sources]
    if not column_specs:
        return

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
        figsize=(3.5 * len(column_specs) + 2.8, 5.1 * len(row_modalities)),
        sharey="row",
        constrained_layout=False,
    )

    if len(column_specs) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, modality in enumerate(row_modalities):
        methods = ordered_methods(records[records["severity"] == severity], modality)
        for col_idx, (model, source) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            subset = records[
                (records["model"] == model)
                & (records["source"] == source)
                & (records["target"] == modality)
                & (records["severity"] == severity)
            ]

            matrix = (
                subset.pivot_table(index="dataset", columns="method", values=metric_col, aggfunc="mean")
                .reindex(index=datasets, columns=methods)
            )
            if matrix.empty:
                matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)

            if len(datasets) == 0 or len(methods) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
                if row_idx == 0:
                    ax.set_title("")
                continue

            values = matrix.to_numpy(dtype=float)
            ax.imshow(values.T, cmap=cmap, norm=norm, aspect="auto")

            for y in range(len(methods) + 1):
                ax.axhline(y - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)
            for x in range(len(datasets) + 1):
                ax.axvline(x - 0.5, color="#d9d9d9", linewidth=0.4, zorder=2)

            annotations = matrix.map(lambda value: "" if pd.isna(value) else f"{value:+.1f}")
            significance_col = "significant_accuracy" if metric_col == "delta_accuracy" else "significant_macro_f1"
            significance_matrix = pd.DataFrame(index=datasets, columns=methods, dtype=object)
            if significance_col in subset.columns:
                significance_matrix = (
                    subset.pivot_table(index="dataset", columns="method", values=significance_col, aggfunc="max")
                    .reindex(index=datasets, columns=methods)
                )
            for y_idx, method in enumerate(methods):
                for x_idx, dataset_name in enumerate(datasets):
                    text = annotations.at[dataset_name, method]
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
                            fontsize=11,
                            color="black",
                            fontweight="bold" if (source == "placebo" and is_significant) else "normal",
                        )

            if row_idx == 0:
                ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(MODALITY_LABEL.get(modality, modality.upper()), fontsize=14)
                ax.yaxis.set_label_coords(-0.78, 0.5)
            else:
                ax.set_ylabel("")

            dataset_tick_labels = [DATASET_LABEL.get(dataset, dataset) for dataset in datasets]
            ax.set_xticks(np.arange(len(datasets)))
            ax.set_xticklabels(dataset_tick_labels, rotation=26, ha="right", fontsize=10.5)

            method_tick_labels = [format_method_tick_label(method) for method in methods]
            ax.set_yticks(np.arange(len(methods)))
            if col_idx == 0:
                ax.set_yticklabels(method_tick_labels, rotation=0, fontsize=10.5)
                ax.tick_params(axis="y", labelleft=True, left=True)
            else:
                ax.tick_params(axis="y", labelleft=False, left=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.914, 0.16, 0.012, 0.71])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Source - Baseline (pp)", fontsize=12)
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.ax.tick_params(labelsize=10)

    metric_label = "Accuracy" if metric_col == "delta_accuracy" else "Macro F1"
    datasets_title = ", ".join(DATASET_LABEL.get(dataset, dataset.upper()) for dataset in datasets)
    severity_display = severity_name(severity)
    source_title = "/".join(str(source) for source in sources)
    fig.suptitle("")
    fig.supxlabel("Dataset", fontsize=13, y=0.045)
    fig.subplots_adjust(left=0.18, right=0.89, top=0.90, bottom=0.18, wspace=0.12, hspace=0.44)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_csv(records: pd.DataFrame, csv_output: Path) -> Path:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "model",
        "source",
        "dataset",
        "modalities",
        "noise_modalities",
        "target",
        "method",
        "severity",
        "accuracy_baseline",
        "accuracy_source",
        "delta_accuracy",
        "macro_f1_baseline",
        "macro_f1_source",
        "delta_macro_f1",
        "n_paired_qwen_placebo",
        "wins_qwen_scored",
        "wins_placebo",
        "pvalue_qwen_placebo_accuracy",
        "significant_accuracy",
        "pvalue_qwen_placebo_macro_f1",
        "significant_macro_f1",
    ]
    available_cols = [col for col in export_cols if col in records.columns]
    export = records[available_cols].copy()
    export["dataset_label"] = export["dataset"].map(lambda value: DATASET_LABEL.get(value, value))
    export["target_label"] = export["target"].map(lambda value: MODALITY_LABEL.get(value, value))
    export["source_label"] = export["source"].map(source_display_label)
    export["method_label"] = export["method"].map(format_method_tick_label)
    export.to_csv(csv_output, index=False)
    return csv_output


def main() -> None:
    args = parse_args()
    datasets = [str(dataset).strip().lower() for dataset in args.datasets]
    datasets = [dataset for dataset in datasets if dataset]
    if not datasets:
        raise RuntimeError("No datasets provided.")

    all_records: list[pd.DataFrame] = []
    all_notes: list[str] = []
    for model in args.models:
        model_records, notes = compute_source_vs_baseline_for_model(
            model=model,
            sources=args.sources,
            datasets=datasets,
            significance_alpha=args.significance_alpha,
            significance_min_paired=args.significance_min_paired,
            macro_f1_permutations=args.macro_f1_permutations,
            significance_seed=args.significance_seed,
        )
        all_notes.extend(notes)
        if model_records.empty:
            all_notes.append(f"[WARN] No merged source-vs-baseline records for {model}")
            continue
        all_records.append(model_records)

    for note in all_notes:
        print(note)

    if not all_records:
        raise RuntimeError("No source-vs-baseline records found for requested models/sources/datasets.")

    records = pd.concat(all_records, ignore_index=True)
    records = records[records["severity"] == int(args.severity)].copy()
    if records.empty:
        raise RuntimeError(f"No records found at severity {args.severity}.")

    available_models = [model for model in args.models if model in set(records["model"].astype(str).unique())]
    available_sources = [source for source in args.sources if source in set(records["source"].astype(str).unique())]

    csv_path = write_csv(records=records, csv_output=Path(args.csv_output))
    output_dir = Path(args.output_dir)
    dataset_slug = "_".join(datasets)
    severity_display = severity_name(int(args.severity))
    source_slug = "_".join(available_sources) if available_sources else "sources"
    accuracy_path = output_dir / f"{source_slug}_vs_baseline_{dataset_slug}_s{severity_display}_accuracy.png"
    macro_path = output_dir / f"{source_slug}_vs_baseline_{dataset_slug}_s{severity_display}_macro_f1.png"

    plot_metric_heatmap(
        records=records,
        models=available_models,
        sources=available_sources,
        datasets=datasets,
        severity=int(args.severity),
        metric_col="delta_accuracy",
        output_path=accuracy_path,
        dpi=args.dpi,
    )
    plot_metric_heatmap(
        records=records,
        models=available_models,
        sources=available_sources,
        datasets=datasets,
        severity=int(args.severity),
        metric_col="delta_macro_f1",
        output_path=macro_path,
        dpi=args.dpi,
    )

    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote heatmap (Accuracy) to {accuracy_path}")
    print(f"Wrote heatmap (Macro-F1) to {macro_path}")


if __name__ == "__main__":
    main()
