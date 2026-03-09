import pandas as pd
import os
from typing import List
import itertools

def calculate_cm_metrics(split_data: dict[str, pd.DataFrame], classes: List[str]) -> pd.DataFrame:
    """
    Calculate confusion matrix metrics (TP, FP, FN, TN) for each class in each data split.
    """
    metrics = pd.DataFrame(columns=['split', 'class', 'TP', 'FP', 'FN', 'TN'])
    #convert classes to string to avoid issues with NaN values
    classes = [str(cl) for cl in classes]

    for split, data in split_data.items():
        for cl in classes:
            tp = data[(data['prediction'] == cl) & (data['label'] == cl)]
            fp = data[(data['prediction'] == cl) & (data['label'] != cl)]
            fn = data[(data['prediction'] != cl) & (data['label'] == cl)]
            tn = data[(data['prediction'] != cl) & (data['label'] != cl)]
            metrics.loc[len(metrics)] = [split, cl, len(tp), len(fp), len(fn), len(tn)]

    return metrics


def split_data_by_modification(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = sorted(predictions['split'].dropna().unique().tolist())
    split_data = {}

    for split in splits:
        split_data[split] = predictions[predictions['split'] == split]

    return split_data


def prepare_data_for_analysis(modalities: List[str], noise: List[str], task: str) -> None:
    """
    Prepare data for analysis by combining modified and unmodified datasets, calculating metrics, and saving the results.
    """
    pred_folder = os.path.join("out", task)
    mod_str = "".join(sorted(modalities))
    noise_str = "".join(sorted(noise)) if noise else ""

    no_noise_str = "".join([m for m in modalities if m not in noise])

    try:
        all_unmodified = pd.read_csv(os.path.join(pred_folder, f"prediction_{mod_str}_noise_.csv"))
        all_unmodified.replace('unmodified', f"unmodified/{mod_str}", inplace=True)
        all_unmodified.replace('all', f"all/{mod_str}", inplace=True)

        predictions = pd.read_csv(os.path.join(pred_folder, f"prediction_{mod_str}_noise_{noise_str}.csv"))

        if mod_str != noise_str:
            only_unmodified = pd.read_csv(os.path.join(pred_folder, f"prediction_{no_noise_str}_noise_.csv"))
            only_unmodified.replace('unmodified', f"unmodified/{no_noise_str}", inplace=True)
            only_unmodified.replace('all', f"all/{no_noise_str}", inplace=True)
        else:
            only_unmodified = pd.DataFrame()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    if not noise_str == "":
        predictions = pd.concat([predictions, only_unmodified, all_unmodified], ignore_index=True)

    predictions = predictions[predictions['label']!='unknown']
    predictions.dropna(inplace=True)

    split_data = split_data_by_modification(predictions)
    #convert all columns to string to avoid issues with NaN values
    split_data = {split: data.astype(str) for split, data in split_data.items()}

    metrics = calculate_cm_metrics(split_data, classes=predictions['label'].unique())

    os.makedirs(os.path.join("analysis", "out", task, "prepared_data"), exist_ok=True)
    out_path = os.path.join("analysis", "out", task, "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv")
    metrics.to_csv(out_path, index=False)

def create_subset(l: List[str]) -> List[List[str]]:
    subsets = [list(s) for r in range(len(l) + 1) for s in itertools.combinations(l, r)]
    return subsets


if __name__ == "__main__":

    all_modalities = ['i', 't']
    all_noise = ['i', 't']

    subsets_modalities = create_subset(l=all_modalities)
    subsets_modalities.remove([])
    subsets_noise = create_subset(l=all_noise)

    for modalities in subsets_modalities:
        for noise in subsets_noise:

            prepare_data_for_analysis(modalities=modalities, noise=noise, task='test/IMDB')
