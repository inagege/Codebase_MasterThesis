import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from enum import Enum
import re
from matplotlib.lines import Line2D

COLORS_MODALITIES = sns.color_palette('bright', 7)

class ModalityPlotting(Enum):
    AUDIO = ("Audio", COLORS_MODALITIES[0])
    VIDEO = ("Video", COLORS_MODALITIES[1])
    TEXT  = ("Text", COLORS_MODALITIES[2])
    AUDIO_VIDEO = ("Audio, Video", COLORS_MODALITIES[3])
    AUDIO_TEXT = ("Audio, Text", COLORS_MODALITIES[4])
    VIDEO_TEXT = ("Video, Text", COLORS_MODALITIES[5])
    ALL = ("Audio, Video, Text", COLORS_MODALITIES[6])
    UNKNOWN = ("Unknown", "#000000")

    def __init__(self, label, color):
        self.label = label
        self.color = color

    @staticmethod
    def from_string(s: str):
        s = s.lower()

        has_audio = bool(re.search(r"audio", s)) or bool(re.search(r"a", s))
        has_video = bool(re.search(r"video", s)) or bool(re.search(r"v", s))
        has_text  = bool(re.search(r"text",  s)) or bool(re.search(r"t", s))

        key = (has_audio, has_video, has_text)

        mapping = {
            (True,  False, False): ModalityPlotting.AUDIO,
            (False, True,  False): ModalityPlotting.VIDEO,
            (False, False, True):  ModalityPlotting.TEXT,
            (True,  True,  False): ModalityPlotting.AUDIO_VIDEO,
            (True,  False, True):  ModalityPlotting.AUDIO_TEXT,
            (False, True,  True):  ModalityPlotting.VIDEO_TEXT,
            (True,  True,  True):  ModalityPlotting.ALL,
        }

        return mapping.get(key, ModalityPlotting.UNKNOWN)



def extract_classes(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Extract data for each class from the metrics DataFrame.
    """
    classes = data['class'].unique()
    classes.sort()
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
        prepared_data = pd.read_csv(os.path.join("out", task, "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv"))
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
            if 'unmodified' not in row['split']:
                metrics.append(score)
                split.append(row['split'])
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

    out_path = os.path.join("out", task, metric, f"{metric}_{mod_str}_noise_{noise_str}")
    plt.savefig(os.path.join(f"{out_path}.svg"), bbox_inches='tight')
    plt.show()