"""Utility to load multiple CSVs, add a label column indicating their origin, and merge them.

Usage (example):
python merge_predictions.py \
  --files preds1.csv preds2.csv preds3.csv \
  --labels modelA modelB modelC \
  --out merged.csv
"""
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import os


def load_and_label(path: Path, label: str, label_col: str = "source") -> pd.DataFrame:
    """Load a CSV into a DataFrame and add a column with a given label.

    Args:
        path: path to the CSV file
        label: string to add to all rows in this DataFrame
        label_col: column name for the label

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(path)
    df[label_col] = label
    return df


def merge_csvs(paths_labels: List[Tuple[str, str]], label_col: str = "source") -> pd.DataFrame:
    """Load multiple CSVs, add labels, and concatenate them.

    Args:
        paths_labels: list of (path, label) tuples
        label_col: name of the added label column

    Returns:
        concatenated DataFrame
    """
    dfs = []
    for p, lbl in paths_labels:
        pth = Path(p)
        if not pth.exists():
            raise FileNotFoundError(f"File not found: {pth}")
        dfs.append(load_and_label(pth, lbl, label_col))
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    return merged


train_pred_file = os.path.join("out", "train_predictions.csv")
dev_pred_file = os.path.join("out", "dev_predictions.csv")
test_pred_file = os.path.join("out", "test_predictions.csv")

train_pred = pd.read_csv(train_pred_file)
dev_pred = pd.read_csv(dev_pred_file)
test_pred = pd.read_csv(test_pred_file)

for df, split_name in [(train_pred, "train_split"), (dev_pred, "dev_splits_complete"), (test_pred, "output_repeated_splits_test")]:
    df["split"] = split_name

merged = pd.concat([train_pred, dev_pred, test_pred], ignore_index=True, sort=False)
merged.to_csv(os.path.join("out","all_predictions.csv"), index=False)
