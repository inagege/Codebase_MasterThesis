import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from enum import Enum
import re
from matplotlib.lines import Line2D

DATASET_MODALITIES = {
    "meld": {"t", "a", "v"},
    "homeprice": {"t", "i"},
    "IMDB": {"t", "i"},
    "voxceleb": {"a", "v"},
    "nejm": {"t", "i"},
    "marine": {"a", "i"},
}

COLORS_MODALITIES = sns.color_palette('bright', 10)

class ModalityPlotting(Enum):
    AUDIO = ("Audio", COLORS_MODALITIES[0])
    VIDEO = ("Video", COLORS_MODALITIES[1])
    TEXT  = ("Text", COLORS_MODALITIES[2])
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
    def from_string(s: str):
        s = s.lower()

        has_audio = bool(re.search(r"a", s))
        has_video = bool(re.search(r"v", s))
        has_text  = bool(re.search(r"t", s))
        has_image = bool(re.search(r"i", s))


        key = (has_audio, has_video, has_text, has_image)

        mapping = {
            (True,  False, False, False): ModalityPlotting.AUDIO,
            (False, True,  False, False): ModalityPlotting.VIDEO,
            (False, False, True, False):  ModalityPlotting.TEXT,
            (True,  True,  False, False): ModalityPlotting.AUDIO_VIDEO,
            (True,  False, True, False):  ModalityPlotting.AUDIO_TEXT,
            (False, True,  True, False):  ModalityPlotting.VIDEO_TEXT,
            (True,  True,  True, False):  ModalityPlotting.ALL,
            (False, False, False, True): ModalityPlotting.IMAGE,
            (True,  False, False, True):  ModalityPlotting.AUDIO_IMAGE,
            (False, False, True, True):  ModalityPlotting.IMAGE_TEXT
        }

        return mapping.get(key, ModalityPlotting.UNKNOWN)



def extract_classes(data: pd.DataFrame) -> dict[object, pd.DataFrame]:
    """
    Extract data for each class from the metrics DataFrame.
    """
    classes = [cl for cl in data['class'].unique().tolist() if pd.notna(cl)]
    classes = sorted(classes)
    class_data = {}

    for cl in classes:
        class_data[cl] = data[data['class'] == cl]

    return class_data


def calculate_score(metric: str, tp:int=0, fp:int=0, tn:int=0, fn:int=0) -> float:
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


# python
def plot_metric_per_class(modalities: List[str], noise: List[str], metric: str, task: str) -> None:
    """
    Plot the specified metric for each class across different data splits.
    """
    mod_str = "".join(sorted(modalities))
    noise_str = "".join(sorted(noise)) if noise else ""

    try:
        prepared_data = pd.read_csv(os.path.join("analysis", "out", task, "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv"))
    except FileNotFoundError as e:
        print(f"Prepared data file not found: {e}")
        return

    class_data = extract_classes(prepared_data)

    fig, axs = plt.subplots(1, len(class_data.keys()), sharey=True, figsize=(4*len(class_data.keys()), 5))
    axs = axs.ravel()  # ensure indexing works even if returned as 1d array
    if metric == 'mcc':
        axs[0].set_ylim(-1, 1)
    else:
        axs[0].set_ylim(0, 1)
    axs[0].set_ylabel(metric)

    # prepare legend handles for unmodified lines
    line_handles = []
    line_labels = []

    i = 0
    for cl, data in class_data.items():
        metrics = []
        split = []
        for ind, row in data.iterrows():
            score = calculate_score(metric, tp=row['TP'], fp=row['FP'], tn=row['TN'], fn=row['FN'])
            if ('unmodified' not in row['split']) and ('all/' not in row['split']):
                metrics.append(score)
                split.append(row['split'])
            else:
                if 'all/' in row['split']:
                    mod = ModalityPlotting.from_string(row['split'].split('all/')[1])
                else:
                    mod = ModalityPlotting.from_string(row['split'])
                axs[i].axhline(y=score, color=mod.color, linestyle='--')
                if mod.label not in line_labels:
                    # create a legend handle for this unmodified split (one per unique split)
                    line_handles.append(Line2D([0], [0], color=mod.color, linestyle='--'))
                    line_labels.append(mod.label)

        colors = sns.color_palette("ocean_r", len(split))
        axs[i].bar(split, metrics, color=colors)
        axs[i].set_xticks(ticks=range(len(split)), labels=split, rotation=45, ha='right')
        axs[i].set_title(cl, fontsize=10)
        axs[i].grid(axis='y')
        i += 1

    # place a single shared legend for the unmodified lines
    if line_handles:
        fig.subplots_adjust(right=0.85)  # make room on the right for the legend
        fig.legend(handles=line_handles, labels=line_labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), title="Unmodified Input")

    out_path = os.path.join("analysis", "out", task, metric, f"{metric}_{mod_str}_noise_{noise_str}")
    os.makedirs(os.path.join("analysis", "out", task, metric), exist_ok=True)
    plt.savefig(os.path.join(f"{out_path}.svg"), bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'IMDB')")
    parser.add_argument("--scored", type=bool, default=False, help="Whether scored or unscored")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = ['mcc', 'f1', 'recall', 'precision']
    dataset = args.dataset
    if '/' in dataset:
        dataset = dataset.split('/')[1]
    modalities = list(DATASET_MODALITIES[dataset])
    if args.scored:
        dataset = f"test/{dataset}"

    for metric in metrics:
        plot_metric_per_class(metric=metric, modalities=modalities, noise=[], task=dataset)
        for mod in modalities:
            plot_metric_per_class(metric=metric, modalities=modalities, noise=[mod], task=dataset)
            plot_metric_per_class(metric=metric, modalities=[mod], noise=[], task=dataset)

