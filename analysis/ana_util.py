import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_cm_metrics(split_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate confusion matrix metrics (TP, FP, FN, TN) for each class in each data split.
    """
    classes = ['positive', 'negative', 'neutral']
    metrics = pd.DataFrame(columns=['split', 'class', 'TP', 'FP', 'FN', 'TN'])

    for split, data in split_data.items():
        for cl in classes:
            tp = data[(data['prediction'] == cl) & (data['label'] == cl)]
            fp = data[(data['prediction'] == cl) & (data['label'] != cl)]
            fn = data[(data['prediction'] != cl) & (data['label'] == cl)]
            tn = data[(data['prediction'] != cl) & (data['label'] != cl)]
            metrics.loc[len(metrics)] = [split, cl, len(tp), len(fp), len(fn), len(tn)]

    return metrics

def prepare_data_for_analysis_one_modality(unmodified_data_file: str, modified_data_file: str, out_file: str) -> None:
    """
    Prepare data for analysis by combining modified and unmodified datasets, calculating metrics, and saving the results.
    """
    path = os.path.join("..", "out", "one_modality", modified_data_file)
    predictions = pd.read_csv(path)

    unmodified_path = os.path.join("..", "out", "one_modality", unmodified_data_file)
    unmodified_data = pd.read_csv(unmodified_path)

    out_path = os.path.join("out", "one_modality", out_file)

    predictions = pd.concat([predictions, unmodified_data], ignore_index=True)

    splits = predictions['split'].unique()
    split_data = {}

    for split in splits:
        split_data[split] = predictions[predictions['split'] == split]

    metrics = calculate_cm_metrics(split_data)

    metrics.to_csv(out_path, index=False)


def prepare_data_for_analysis_two_modalities(modified_modality2_file: str, modified_modality1_file: str, unmodified_data_file: str, out_file: str) -> None:
    """
    Prepare data for analysis by combining two modified datasets and an unmodified dataset, calculating metrics, and saving the results.
    """
    path1 = os.path.join("..", "out", "two_modalities", modified_modality1_file)
    predictions1 = pd.read_csv(path1)

    path2 = os.path.join("..", "out", "two_modalities", modified_modality2_file)
    predictions2 = pd.read_csv(path2)

    unmodified_path = os.path.join("..", "out", "two_modalities", unmodified_data_file)
    unmodified_data = pd.read_csv(unmodified_path)

    out_path = os.path.join("out", "two_modalities", out_file)

    predictions = pd.concat([predictions1, predictions2, unmodified_data], ignore_index=True)

    splits = predictions['split'].unique()
    split_data = {}

    for split in splits:
        split_data[split] = predictions[predictions['split'] == split]

    metrics = calculate_cm_metrics(split_data)

    metrics.to_csv(out_path, index=False)


def extract_classes(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Extract data for each class from the metrics DataFrame.
    """
    classes = data['class'].unique()
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


def plot_metric_per_class(class_data: dict[str, pd.DataFrame], metric: str, out_path: str, modality: str) -> None:
    """
    Plot the specified metric for each class across different data splits.
    """
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