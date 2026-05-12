import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from enum import Enum
import re
from matplotlib.lines import Line2D

LOW_CLASS_THRESHOLD = 10
HIGH_CLASS_THRESHOLD = 150
TOP_CLASS_COUNT = 9

DATASET_MODALITIES = {
    "emotion": {"t", "a", "v"},
    "sentiment": {"t", "a", "v"},
    "homeprice": {"t", "i"},
    "imdb": {"t", "i"},
    "voxceleb": {"a", "v"},
    "nejm": {"t", "i"},
    "marine": {"a", "i"},
}

COLORS_MODALITIES = sns.color_palette('bright', 10)


class ModalityPlotting(Enum):
    AUDIO = ("Audio", COLORS_MODALITIES[0])
    VIDEO = ("Video", COLORS_MODALITIES[1])
    TEXT = ("Text", COLORS_MODALITIES[2])
    IMAGE = ("Image", COLORS_MODALITIES[3])
    AUDIO_VIDEO = ("Audio, Video", COLORS_MODALITIES[4])
    AUDIO_TEXT = ("Audio, Text", COLORS_MODALITIES[5])
    AUDIO_IMAGE = ("Audio, Image", COLORS_MODALITIES[6])
    IMAGE_TEXT = ("Image, Text", COLORS_MODALITIES[7])
    VIDEO_TEXT = ("Video, Text", COLORS_MODALITIES[8])
    ALL = ("Audio, Video, Text", COLORS_MODALITIES[9])
    UNKNOWN = ("Unknown", "#000000")

    def __init__(self, label, color):
        self.label = label
        self.color = color

    @staticmethod
    def _extract_modalities(s: str) -> set[str]:
        text = str(s or "").lower().strip()
        if not text:
            return set()

        # Prefer explicit split markers such as A=..., V=..., T=..., I=...
        explicit_markers = {marker.lower() for marker in re.findall(r"(?<![a-z0-9])([atvi])=", text)}
        if explicit_markers:
            return explicit_markers

        modalities = set()
        for token in re.split(r"[^a-z]+", text):
            if not token:
                continue
            if token in {"a", "audio"}:
                modalities.add("a")
                continue
            if token in {"v", "video"}:
                modalities.add("v")
                continue
            if token in {"t", "text"}:
                modalities.add("t")
                continue
            if token in {"i", "image"}:
                modalities.add("i")
                continue
            # Compact modality tokens like av, tv, atv, ai, it
            if all(char in {"a", "v", "t", "i"} for char in token):
                modalities.update(token)

        return modalities

    @staticmethod
    def from_string(s: str):
        modalities = ModalityPlotting._extract_modalities(s)

        has_audio = "a" in modalities
        has_video = "v" in modalities
        has_text = "t" in modalities
        has_image = "i" in modalities

        key = (has_audio, has_video, has_text, has_image)

        mapping = {
            (True, False, False, False): ModalityPlotting.AUDIO,
            (False, True, False, False): ModalityPlotting.VIDEO,
            (False, False, True, False): ModalityPlotting.TEXT,
            (True, True, False, False): ModalityPlotting.AUDIO_VIDEO,
            (True, False, True, False): ModalityPlotting.AUDIO_TEXT,
            (False, True, True, False): ModalityPlotting.VIDEO_TEXT,
            (True, True, True, False): ModalityPlotting.ALL,
            (False, False, False, True): ModalityPlotting.IMAGE,
            (True, False, False, True): ModalityPlotting.AUDIO_IMAGE,
            (False, False, True, True): ModalityPlotting.IMAGE_TEXT
        }

        return mapping.get(key, ModalityPlotting.UNKNOWN)


def get_modality_and_noise_string(modalities: List[str], noise: List[str]) -> tuple[str, str]:
    mod_str = "".join(sorted(modalities))
    noise_str = "".join(sorted(noise)) if noise else ""
    return mod_str, noise_str


def is_reference_split(split: str) -> bool:
    split = str(split or "").strip().lower()
    return ('unmodified' in split) or ('all/' in split) or (split in {'all', 'test_all', 'dev'})


def get_modality_for_split(split: str, fallback_modalities: str = "") -> ModalityPlotting:
    split = split.lower()
    if "/" in split:
        split = split.split("/", 1)[1]
    detected = ModalityPlotting.from_string(split)
    if detected != ModalityPlotting.UNKNOWN:
        return detected
    if fallback_modalities:
        fallback_detected = ModalityPlotting.from_string(fallback_modalities)
        if fallback_detected != ModalityPlotting.UNKNOWN:
            return fallback_detected
    return ModalityPlotting.UNKNOWN


def get_class_count(data: pd.DataFrame) -> int:
    if 'class_count' in data.columns and not data['class_count'].dropna().empty:
        return int(float(data['class_count'].dropna().iloc[0]))
    if 'class' in data.columns:
        return int(data['class'].astype(str).nunique())
    return 0


def calculate_score(metric: str, tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0) -> float:
    """
    Calculate the specified metric based on confusion matrix values.
    """
    match metric:
        case 'precision':
            return calculate_precision(tp, fp)
        case 'recall':
            return calculate_recall(tp, fn)
        case 'f1':
            return calculate_f1(tp, fp, fn)
        case 'mcc':
            return calculate_mcc(tp, tn, fp, fn)
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def calculate_precision(tp: int, fp: int) -> float:
    """
    Calculate precision metric.
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp: int, fn: int) -> float:
    """
    Calculate recall metric.
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    """
    Calculate F1 score metric.
    """
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate Matthews correlation coefficient (MCC) metric.
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_accuracy_data(modalities: List[str], noise: List[str], task: str) -> pd.DataFrame:
    mod_str, noise_str = get_modality_and_noise_string(modalities=modalities, noise=noise)

    accuracy_path = os.path.join("analysis", "../out", task, "prepared_data", f"accuracy_{mod_str}_noise_{noise_str}.csv")
    if os.path.exists(accuracy_path):
        data = pd.read_csv(accuracy_path)
        return data

    prepared_path = os.path.join("analysis", "../out", task, "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv")
    if not os.path.exists(prepared_path):
        return pd.DataFrame()

    prepared_data = pd.read_csv(prepared_path)
    required_columns = {'split', 'TP', 'FN'}
    if not required_columns.issubset(prepared_data.columns):
        return pd.DataFrame()

    prepared_data['TP'] = pd.to_numeric(prepared_data['TP'], errors='coerce').fillna(0)
    prepared_data['FN'] = pd.to_numeric(prepared_data['FN'], errors='coerce').fillna(0)
    prepared_data['total_per_class'] = prepared_data['TP'] + prepared_data['FN']

    accuracy_data = prepared_data.groupby('split', as_index=False).agg(
        correct=('TP', 'sum'),
        total=('total_per_class', 'sum')
    )
    accuracy_data['accuracy'] = accuracy_data.apply(
        lambda row: row['correct'] / row['total'] if row['total'] > 0 else 0.0,
        axis=1
    )
    return accuracy_data[['split', 'total', 'correct', 'accuracy']]


def get_top_classes(modalities: List[str], noise: List[str], task: str, prepared_data: pd.DataFrame) -> List[str]:
    mod_str, noise_str = get_modality_and_noise_string(modalities=modalities, noise=noise)
    true_label_count_path = os.path.join(
        "analysis",
        "../out",
        task,
        "prepared_data",
        f"true_label_count_{mod_str}_noise_{noise_str}.csv"
    )

    if os.path.exists(true_label_count_path):
        true_label_counts = pd.read_csv(true_label_count_path)
        true_label_counts['class'] = true_label_counts['class'].astype(str)
        true_label_counts['true_label_count'] = pd.to_numeric(true_label_counts['true_label_count'], errors='coerce').fillna(0)
        sorted_classes = true_label_counts.sort_values(by='true_label_count', ascending=False)['class'].tolist()
        return sorted_classes[:TOP_CLASS_COUNT]

    fallback_data = prepared_data.copy()
    fallback_data['TP'] = pd.to_numeric(fallback_data['TP'], errors='coerce').fillna(0)
    fallback_data['FN'] = pd.to_numeric(fallback_data['FN'], errors='coerce').fillna(0)
    fallback_data['support'] = fallback_data['TP'] + fallback_data['FN']
    class_counts = fallback_data.groupby('class', as_index=False)['support'].sum()
    class_counts['class'] = class_counts['class'].astype(str)
    sorted_classes = class_counts.sort_values(by='support', ascending=False)['class'].tolist()
    return sorted_classes[:TOP_CLASS_COUNT]


def calculate_weighted_group_score(group_data: pd.DataFrame, metric: str) -> float:
    if group_data.empty:
        return 0.0

    group_data = group_data.copy()
    group_data['TP'] = pd.to_numeric(group_data['TP'], errors='coerce').fillna(0)
    group_data['FP'] = pd.to_numeric(group_data['FP'], errors='coerce').fillna(0)
    group_data['TN'] = pd.to_numeric(group_data['TN'], errors='coerce').fillna(0)
    group_data['FN'] = pd.to_numeric(group_data['FN'], errors='coerce').fillna(0)
    group_data['support'] = group_data['TP'] + group_data['FN']

    scores = group_data.apply(
        lambda row: calculate_score(
            metric=metric,
            tp=row['TP'],
            fp=row['FP'],
            tn=row['TN'],
            fn=row['FN']
        ),
        axis=1
    )
    total_support = group_data['support'].sum()
    if total_support == 0:
        return float(scores.mean()) if not scores.empty else 0.0
    return float((scores * group_data['support']).sum() / total_support)


def plot_accuracy_over_dataset(modalities: List[str], noise: List[str], task: str) -> None:
    mod_str, noise_str = get_modality_and_noise_string(modalities=modalities, noise=noise)
    accuracy_data = load_accuracy_data(modalities=modalities, noise=noise, task=task)

    if accuracy_data.empty:
        print(f"Accuracy data file not found for {mod_str} noise={noise_str}")
        return

    accuracy_data = accuracy_data.copy()
    accuracy_data['split'] = accuracy_data['split'].astype(str)
    accuracy_data['accuracy'] = pd.to_numeric(accuracy_data['accuracy'], errors='coerce').fillna(0.0)
    accuracy_data = accuracy_data.sort_values(by='split')
    force_bar_mode = accuracy_data['split'].nunique() <= 1

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(accuracy_data)), 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("")

    bar_scores = []
    bar_splits = []
    line_handles = []
    line_labels = []

    for _, row in accuracy_data.iterrows():
        split_name = row['split']
        score = float(row['accuracy'])

        if force_bar_mode or not is_reference_split(split_name):
            bar_scores.append(score)
            bar_splits.append(split_name)
        else:
            mod = get_modality_for_split(split_name, fallback_modalities=mod_str)
            ax.axhline(y=score, color=mod.color, linestyle='--')
            if mod.label not in line_labels:
                line_handles.append(Line2D([0], [0], color=mod.color, linestyle='--'))
                line_labels.append(mod.label)

    colors = sns.color_palette("ocean_r", len(bar_splits))
    ax.bar(bar_splits, bar_scores, color=colors)
    ax.set_xticks(ticks=range(len(bar_splits)), labels=bar_splits, rotation=45, ha='right')
    ax.grid(axis='y')

    if line_handles:
        fig.subplots_adjust(right=0.85)
        fig.legend(
            handles=line_handles,
            labels=line_labels,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.95),
            title="Unmodified Input"
        )

    out_path = os.path.join("analysis", "../out", task, "accuracy", f"accuracy_{mod_str}_noise_{noise_str}")
    os.makedirs(os.path.join("analysis", "../out", task, "accuracy"), exist_ok=True)
    plt.savefig(os.path.join(f"{out_path}.svg"), bbox_inches='tight')
    plt.close(fig)


def plot_metric_per_class(modalities: List[str], noise: List[str], metric: str, task: str) -> None:
    """
    Plot class-wise metrics with class-count aware behavior.
    """
    mod_str, noise_str = get_modality_and_noise_string(modalities=modalities, noise=noise)

    try:
        prepared_data = pd.read_csv(os.path.join("analysis", "../out", task, "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv"))
    except FileNotFoundError as e:
        print(f"Prepared data file not found: {e}")
        return

    required_columns = {'split', 'class', 'TP', 'FP', 'TN', 'FN'}
    if not required_columns.issubset(prepared_data.columns):
        print(f"Skipping {metric} plot for {mod_str} noise={noise_str}: no TP/FP/TN/FN data available.")
        return

    prepared_data = prepared_data.copy()
    prepared_data['split'] = prepared_data['split'].astype(str)
    prepared_data['class'] = prepared_data['class'].astype(str)
    for col in ['TP', 'FP', 'TN', 'FN']:
        prepared_data[col] = pd.to_numeric(prepared_data[col], errors='coerce').fillna(0)

    class_count = get_class_count(prepared_data)
    if class_count > HIGH_CLASS_THRESHOLD:
        print(f"Skipping {metric} plot for {mod_str} noise={noise_str}: class_count={class_count} > {HIGH_CLASS_THRESHOLD}.")
        return

    all_classes = sorted(prepared_data['class'].unique().tolist())
    classes_to_plot = all_classes
    other_classes = []

    if LOW_CLASS_THRESHOLD <= class_count <= HIGH_CLASS_THRESHOLD:
        top_classes = get_top_classes(modalities=modalities, noise=noise, task=task, prepared_data=prepared_data)
        top_classes = [cl for cl in top_classes if cl in all_classes]

        if len(top_classes) < TOP_CLASS_COUNT:
            for cl in all_classes:
                if cl not in top_classes:
                    top_classes.append(cl)
                if len(top_classes) == TOP_CLASS_COUNT:
                    break

        other_classes = [cl for cl in all_classes if cl not in top_classes]
        classes_to_plot = top_classes.copy()
        if other_classes:
            classes_to_plot.append("others")

    if not classes_to_plot:
        print(f"Skipping {metric} plot for {mod_str} noise={noise_str}: no classes available.")
        return

    fig, axs = plt.subplots(1, len(classes_to_plot), sharey=True, figsize=(4 * len(classes_to_plot), 5))
    if hasattr(axs, "ravel"):
        axs = axs.ravel().tolist()
    else:
        axs = [axs]

    if metric == 'mcc':
        for ax in axs:
            ax.set_ylim(-1, 1)
    else:
        for ax in axs:
            ax.set_ylim(0, 1)
    axs[0].set_ylabel(metric)

    line_handles = []
    line_labels = []
    split_names = sorted(prepared_data['split'].dropna().unique().tolist())
    force_bar_mode = len(split_names) <= 1

    for i, cl in enumerate(classes_to_plot):
        bar_scores = []
        bar_splits = []
        ax = axs[i]

        for split_name in split_names:
            split_data = prepared_data[prepared_data['split'] == split_name]

            if cl == "others":
                class_data = split_data[split_data['class'].isin(other_classes)]
                score = calculate_weighted_group_score(class_data, metric=metric)
            else:
                class_data = split_data[split_data['class'] == cl]
                if class_data.empty:
                    continue
                row = class_data.iloc[0]
                score = calculate_score(metric, tp=row['TP'], fp=row['FP'], tn=row['TN'], fn=row['FN'])

            if force_bar_mode or not is_reference_split(split_name):
                bar_scores.append(score)
                bar_splits.append(split_name)
            else:
                mod = get_modality_for_split(split_name, fallback_modalities=mod_str)
                ax.axhline(y=score, color=mod.color, linestyle='--')
                if mod.label not in line_labels:
                    line_handles.append(Line2D([0], [0], color=mod.color, linestyle='--'))
                    line_labels.append(mod.label)

        colors = sns.color_palette("ocean_r", len(bar_splits))
        ax.bar(bar_splits, bar_scores, color=colors)
        ax.set_xticks(ticks=range(len(bar_splits)), labels=bar_splits, rotation=45, ha='right')
        ax.set_title("")
        ax.grid(axis='y')

    if line_handles:
        fig.subplots_adjust(right=0.85)
        fig.legend(handles=line_handles, labels=line_labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), title="Unmodified Input")

    out_path = os.path.join("analysis", "../out", task, metric, f"{metric}_{mod_str}_noise_{noise_str}")
    os.makedirs(os.path.join("analysis", "../out", task, metric), exist_ok=True)
    plt.savefig(os.path.join(f"{out_path}.svg"), bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'IMDB')")
    parser.add_argument("--state", type=str, default=None, help="Whether scored, calibrated, unscored")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model folder under analysis/out/. New layout: analysis/out/<model>/...",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = ['mcc', 'f1', 'recall', 'precision']
    dataset_name = args.dataset.lower()
    state = args.state

    if "/" in dataset_name:
        dataset_name = dataset_name.split("/")[-1]
    modalities = list(DATASET_MODALITIES[dataset_name])

    task = f"{state}/{dataset_name}" if state else dataset_name
    if args.model:
        task = os.path.join(args.model, task)

    plot_accuracy_over_dataset(modalities=modalities, noise=[], task=task)
    for metric in metrics:
        plot_metric_per_class(metric=metric, modalities=modalities, noise=[], task=task)

    for mod in modalities:
        plot_accuracy_over_dataset(modalities=modalities, noise=[mod], task=task)
        plot_accuracy_over_dataset(modalities=[mod], noise=[], task=task)
        plot_accuracy_over_dataset(modalities=[mod], noise=[mod], task=task)

        for metric in metrics:
            plot_metric_per_class(metric=metric, modalities=modalities, noise=[mod], task=task)
            plot_metric_per_class(metric=metric, modalities=[mod], noise=[], task=task)
            plot_metric_per_class(metric=metric, modalities=[mod], noise=[mod], task=task)
