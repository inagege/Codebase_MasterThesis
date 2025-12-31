import re
import pandas as pd
import os

# helper to get utterance text for a given filename
def get_utterance_text_for_file(filename: str, meta_data: pd.DataFrame) -> str:
    base = os.path.basename(filename)
    dia_id = base[base.find("dia") + 3:base.find("_utt")]
    utt_id = base[base.find("_utt") + 4:base.find(".mp4", base.find("_utt"))]
    matches = meta_data[(meta_data["Dialogue_ID"] == int(dia_id)) & (meta_data["Utterance_ID"] == int(utt_id))]["Utterance"]
    return matches.iat[0] if len(matches) > 0 else ""

def extract_assistant_reply(full_text: str) -> str:
    m = re.search(r"(?:\\bassistant\\b[:\\n\\s]*)([\\s\\S]*)$", full_text, flags=re.IGNORECASE)
    if m:
        reply = m.group(1).strip()
        return reply
    for line in reversed(full_text.splitlines()):
        if line.strip():
            return line.strip()
    return full_text.strip()

def get_ids_for_file(filename: str) -> tuple[int, int]:
    """Extract (dialog_id, utterance_id) from a filename like 'dia0_utt1.mp4'. Returns ints."""
    base = os.path.basename(filename)
    dia_str = base[base.find("dia") + 3:base.find("_utt")]
    utt_str = base[base.find("_utt") + 4:base.find(".mp4", base.find("_utt"))]
    try:
        dia_id = int(dia_str)
    except ValueError:
        dia_id = -1
    try:
        utt_id = int(utt_str)
    except ValueError:
        utt_id = -1
    return dia_id, utt_id

def get_label_for_file(filename: str, meta_data: pd.DataFrame) -> str:
    """Get the sentiment label for a given filename from the metadata DataFrame."""
    base = os.path.basename(filename)
    dia_id = base[base.find("dia") + 3:base.find("_utt")]
    utt_id = base[base.find("_utt") + 4:base.find(".mp4", base.find("_utt"))]
    matches = meta_data[(meta_data["Dialogue_ID"] == int(dia_id)) & (meta_data["Utterance_ID"] == int(utt_id))]["Sentiment"]
    return matches.iat[0] if len(matches) > 0 else "unknown"