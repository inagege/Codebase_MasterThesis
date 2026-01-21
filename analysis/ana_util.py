import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def extract_classes(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    classes = data['class'].unique()
    class_data = {}

    for cl in classes:
        class_data[cl] = data[data['class'] == cl]

    return class_data

def calculate_score(metric: str, tp:int=0, fp:int=0, tn:int=0, fn:int=0) -> float:
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
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def calculate_recall(tp: int, fn: int) -> float:
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def calculate_f1(tp: int, fp: int, fn: int) -> float:
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator

def plot_metric_per_class(class_data: dict[str, pd.DataFrame], metric: str, out_path: str, modality: str) -> None:
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(4, 20))
    fig.set_size_inches(10, 5)
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel(metric)

    i = 0

    for cl, data in class_data.items():
        metrics = []
        split = []
        for ind, row in data.iterrows():
            score = calculate_score(metric, tp=row['TP'], fp=row['FP'], tn=row['TN'], fn=row['FN'])
            if row['split'].split('/')[0] != 'unmodified':
                metrics.append(score)
                split.append(row['split'].split('/')[0])
            else:
                axs[i].axhline(y=score, color='r', linestyle='--')

        colors = sns.color_palette("Paired", len(split))
        axs[i].bar(split, metrics, color=colors)
        axs[i].set_xticks(ticks=range(len(split)), labels=split, rotation=45, ha='right')
        axs[i].set_title(cl, fontsize=10)
        i = i + 1

    fig.legend(['unmodified'])
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(f"{out_path}", f"{metric}_per_class_{modality}.png"), bbox_inches='tight')
    plt.show()