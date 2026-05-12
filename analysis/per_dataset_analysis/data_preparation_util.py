import pandas as pd
import os
from typing import List
import itertools
import unicodedata
import argparse

LOW_CLASS_THRESHOLD = 10
HIGH_CLASS_THRESHOLD = 150

DATASET_MODALITIES = {
    "emotion": {"t", "a", "v"},
    "sentiment": {"t", "a", "v"},
    "homeprice": {"t", "i"},
    "imdb": {"t", "i"},
    "voxceleb": {"a", "v"},
    "nejm": {"t", "i"},
    "marine": {"a", "i"},
}


def _join_with_optional_model(base: str, model: str | None, task: str) -> str:
    if model:
        return os.path.join(base, model, task)
    return os.path.join(base, task)


def normalize_class_value(value: object) -> str:
    """
    Normalize class strings for robust comparisons (case/spacing/unicode).
    """
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return normalized.strip().casefold()


def calculate_cm_metrics(split_data: dict[str, pd.DataFrame], classes: List[str]) -> pd.DataFrame:
    """
    Calculate confusion matrix metrics (TP, FP, FN, TN) for each class in each data split.
    """
    metrics = pd.DataFrame(columns=['split', 'class', 'TP', 'FP', 'FN', 'TN'])
    normalized_classes = []
    seen = set()
    for cl in classes:
        normalized = normalize_class_value(cl)
        if normalized == "" or normalized in seen:
            continue
        seen.add(normalized)
        normalized_classes.append(normalized)

    for split, data in split_data.items():
        pred_norm = data['prediction'].map(normalize_class_value)
        label_norm = data['label'].map(normalize_class_value)
        for cl in normalized_classes:
            tp = ((pred_norm == cl) & (label_norm == cl)).sum()
            fp = ((pred_norm == cl) & (label_norm != cl)).sum()
            fn = ((pred_norm != cl) & (label_norm == cl)).sum()
            tn = ((pred_norm != cl) & (label_norm != cl)).sum()
            metrics.loc[len(metrics)] = [split, cl, int(tp), int(fp), int(fn), int(tn)]

    return metrics


def calculate_accuracy_metrics(split_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate total/correct predictions and accuracy for each split.
    """
    metrics = pd.DataFrame(columns=['split', 'total', 'correct', 'accuracy'])

    for split, data in split_data.items():
        total = len(data)
        pred_norm = data['prediction'].map(normalize_class_value)
        label_norm = data['label'].map(normalize_class_value)
        correct = int((pred_norm == label_norm).sum())
        accuracy = correct / total if total > 0 else 0.0
        metrics.loc[len(metrics)] = [split, total, correct, accuracy]

    return metrics


def split_data_by_modification(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = sorted(predictions['split'].dropna().unique().tolist())
    split_data = {}

    for split in splits:
        split_data[split] = predictions[predictions['split'] == split]

    return split_data


def prepare_data_for_analysis(modalities: List[str], noise: List[str], task: str, model: str | None = None) -> None:
    """
    Prepare data for analysis by combining modified and unmodified datasets, calculating metrics, and saving the results.
    """
    pred_folder = _join_with_optional_model("out", model, task)
    analysis_task = os.path.join(model, task) if model else task
    mod_str = "".join(sorted(modalities))
    noise_str = "".join(sorted(noise)) if noise else ""

    no_noise_str = "".join([m for m in modalities if m not in noise])

    # Build paths for expected files
    path_all_unmodified = os.path.join(pred_folder, f"prediction_{mod_str}_noise_.csv")
    path_predictions = os.path.join(pred_folder, f"prediction_{mod_str}_noise_{noise_str}.csv")
    path_only_unmodified = os.path.join(pred_folder, f"prediction_{no_noise_str}_noise_.csv")

    # Read files robustly: predictions is required, others are optional
    if not os.path.exists(path_predictions):
        print(f"Required predictions file not found: {path_predictions}")
        return

    # Read the required predictions file
    predictions = pd.read_csv(path_predictions)

    # Conditionally read optional files; if missing, use empty DataFrame
    if os.path.exists(path_all_unmodified):
        all_unmodified = pd.read_csv(path_all_unmodified)
        all_unmodified.replace('unmodified', f"unmodified/{mod_str}", inplace=True)
        all_unmodified.replace('all', f"all/{mod_str}", inplace=True)
    else:
        all_unmodified = pd.DataFrame()

    if mod_str != noise_str:
        if os.path.exists(path_only_unmodified):
            only_unmodified = pd.read_csv(path_only_unmodified)
            only_unmodified.replace('unmodified', f"unmodified/{no_noise_str}", inplace=True)
            only_unmodified.replace('all', f"all/{no_noise_str}", inplace=True)
        else:
            only_unmodified = pd.DataFrame()
    else:
        only_unmodified = pd.DataFrame()

    # If we have noise specified, concatenate available optional frames
    if noise_str != "":
        frames = [df for df in [predictions, only_unmodified, all_unmodified] if not df.empty]
        if frames:
            predictions = pd.concat(frames, ignore_index=True)

    predictions.dropna(inplace=True)
    predictions = predictions.copy()
    predictions['label'] = predictions['label'].map(normalize_class_value)
    predictions['prediction'] = predictions['prediction'].map(normalize_class_value)
    predictions['split'] = predictions['split'].astype(str)
    predictions = predictions[predictions['label'] != 'unknown']
    predictions = predictions[predictions['label'] != '']

    class_count = predictions['label'].nunique()
    true_label_counts = predictions['label'].value_counts().reset_index()
    true_label_counts.columns = ['class', 'true_label_count']

    split_data = split_data_by_modification(predictions)

    os.makedirs(os.path.join("analysis", "../out", analysis_task, "prepared_data"), exist_ok=True)

    true_label_count_out = os.path.join(
        "analysis",
        "../out",
        analysis_task,
        "prepared_data",
        f"true_label_count_{mod_str}_noise_{noise_str}.csv",
    )
    true_label_counts.to_csv(true_label_count_out, index=False)

    accuracy_metrics = calculate_accuracy_metrics(split_data)
    accuracy_metrics['class_count'] = class_count
    accuracy_out_path = os.path.join(
        "analysis",
        "../out",
        analysis_task,
        "prepared_data",
        f"accuracy_{mod_str}_noise_{noise_str}.csv",
    )
    accuracy_metrics.to_csv(accuracy_out_path, index=False)

    if class_count > HIGH_CLASS_THRESHOLD:
        metrics = accuracy_metrics.copy()
        metrics['evaluation_mode'] = 'accuracy_only'
    else:
        metrics = calculate_cm_metrics(split_data, classes=predictions['label'].unique())
        metrics['class_count'] = class_count
        if class_count >= LOW_CLASS_THRESHOLD:
            metrics['evaluation_mode'] = 'top9_plus_others'
        else:
            metrics['evaluation_mode'] = 'full_per_class'

    out_path = os.path.join(
        "analysis",
        "../out",
        analysis_task,
        "prepared_data",
        f"prepared_{mod_str}_noise_{noise_str}.csv",
    )
    metrics.to_csv(out_path, index=False)


def create_subset(l: List[str]) -> List[List[str]]:
    subsets = [list(s) for r in range(len(l) + 1) for s in itertools.combinations(l, r)]
    return subsets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'IMDB')")
    parser.add_argument("--state", type=str, default=None, help="Whether scored, calibrated, unscored")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen_7B",
        help="Optional model folder under out/. New layout: out/<model>/<dataset or state/dataset>.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset.lower()
    all_modalities = list(DATASET_MODALITIES[dataset])

    subsets_modalities = create_subset(l=all_modalities)
    subsets_modalities.remove([])
    subsets_noise = create_subset(l=all_modalities)

    task = f"{state}/{dataset}" if (state := args.state) else dataset

    for modalities in subsets_modalities:
        for noise in subsets_noise:
            prepare_data_for_analysis(modalities=modalities, noise=noise, task=task, model=args.model)
