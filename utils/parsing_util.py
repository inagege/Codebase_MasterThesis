# python
import os
import re
from typing import Tuple
import pandas as pd

def get_ids_for_file(filename: str) -> Tuple[int, int]:
    """
    Extract (dialog_id, utterance_id) from filenames like 'dia0_utt1.mp4'.
    Returns (-1, -1) if not found or not integer.
    """
    base = os.path.basename(filename)
    m = re.search(r"\bdia(\d+)_utt(\d+)\b", base, flags=re.IGNORECASE)
    if not m:
        return -1, -1
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return -1, -1

def get_utterance_text_for_file(filename: str, meta_data: pd.DataFrame) -> str:
    """
    Return the utterance text from `meta_data` for the given filename.
    Safely handles missing IDs, missing columns, and empty results.
    """
    dia_id, utt_id = get_ids_for_file(filename)
    if dia_id < 0 or utt_id < 0:
        return ""
    # Ensure columns exist
    if "Dialogue_ID" not in meta_data.columns or "Utterance_ID" not in meta_data.columns or "Utterance" not in meta_data.columns:
        return ""
    # Compare as ints where possible
    try:
        mask = (meta_data["Dialogue_ID"].astype(int) == dia_id) & (meta_data["Utterance_ID"].astype(int) == utt_id)
    except Exception:
        # If casting fails, try equality as-is (safe fallback)
        mask = (meta_data["Dialogue_ID"] == dia_id) & (meta_data["Utterance_ID"] == utt_id)
    matches = meta_data.loc[mask, "Utterance"]
    if matches.empty:
        return ""
    return str(matches.iloc[0]).strip()

def extract_assistant_reply(full_text: str) -> str:
    """
    Extract the assistant reply block. If multiple 'assistant' sections exist,
    returns the last one. Falls back to the last non-empty line if no label found.
    """
    if not full_text:
        return ""
    pattern = re.compile(r"\bassistant\b\s*[:\-]?\s*([\s\S]+)$", flags=re.IGNORECASE)
    matches = list(pattern.finditer(full_text))
    if matches:
        return matches[-1].group(1).strip()
    # fallback: last non-empty line
    for line in reversed(full_text.splitlines()):
        if line.strip():
            return line.strip()
    return full_text.strip()



def get_label_for_file(filename: str, meta_data: pd.DataFrame) -> str:
    """Get the sentiment label for a given filename from the metadata DataFrame.

    Raises:
        ValueError: if filename format is unexpected or meta_data is invalid.
        LabelNotFoundError: if no matching sentiment is found for the extracted IDs.
    """
    if meta_data is None or not isinstance(meta_data, pd.DataFrame):
        raise ValueError("meta_data must be a pandas DataFrame")

    base = os.path.basename(filename)
    m = re.search(r"dia(\d+)_utt(\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename {filename!r} does not match expected pattern 'dia<id>_utt<id>'")

    dia_id = int(m.group(1))
    utt_id = int(m.group(2))

    # Try to compare numeric values where possible
    if "Dialogue_ID" in meta_data.columns and "Utterance_ID" in meta_data.columns:
        try:
            dialogue = pd.to_numeric(meta_data["Dialogue_ID"], errors="coerce")
            utterance = pd.to_numeric(meta_data["Utterance_ID"], errors="coerce")
            mask = (dialogue == dia_id) & (utterance == utt_id)
        except Exception:
            mask = (meta_data["Dialogue_ID"] == dia_id) & (meta_data["Utterance_ID"] == utt_id)
    else:
        raise ValueError("meta_data must contain 'Dialogue_ID' and 'Utterance_ID' columns")

    matches = meta_data.loc[mask, "Sentiment"]
    if matches.empty:
        return "unknown"

    # normalize and return as string
    return str(matches.iloc[0])
