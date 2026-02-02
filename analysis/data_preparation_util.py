from typing import List

import pandas as pd
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


def split_data_by_modification(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = predictions['split'].unique()
    splits.sort()
    split_data = {}

    for split in splits:
        split_data[split] = predictions[predictions['split'] == split]

    return split_data


def prepare_data_for_analysis(modalities: List[str], noise: List[str]) -> None:
    """
    Prepare data for analysis by combining modified and unmodified datasets, calculating metrics, and saving the results.
    """
    pred_folder = os.path.join("..", "out")
    mod_str = "".join(sorted(modalities))
    noise_str = "".join(sorted(noise)) if noise else ""

    no_noise_str = "".join([m for m in modalities if m not in noise])

    try:
        all_unmodified = pd.read_csv(os.path.join(pred_folder, f"prediction_{mod_str}_noise_.csv"))

        predictions = pd.read_csv(os.path.join(pred_folder, f"prediction_{mod_str}_noise_{noise_str}.csv"))

        if mod_str != noise_str:
            only_unmodified = pd.read_csv(os.path.join(pred_folder, f"prediction_{no_noise_str}_noise_.csv"))
        else:
            only_unmodified = pd.DataFrame()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    if not noise_str == "":
        predictions = pd.concat([predictions, only_unmodified, all_unmodified], ignore_index=True)

    split_data = split_data_by_modification(predictions)

    metrics = calculate_cm_metrics(split_data)

    out_path = os.path.join("out", "prepared_data", f"prepared_{mod_str}_noise_{noise_str}.csv")
    metrics.to_csv(out_path, index=False)


if __name__ == "__main__":
    # Example usage
    prepare_data_for_analysis(modalities=['a'], noise=[''])