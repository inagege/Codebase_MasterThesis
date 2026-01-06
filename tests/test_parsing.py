#iterrate over files in data/output_repeated_splits_test
from pathlib import Path
import sys

# ensure repo root is on sys.path so imports work under pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from utils.parsing_util import get_label_for_file, get_ids_for_file


@pytest.fixture(scope="module")
def meta_data():
    repo_root = Path(__file__).resolve().parent.parent
    meta_csv = repo_root / "data" / "test_sent_emo.csv"
    assert meta_csv.exists(), f"Metadata CSV not found: {meta_csv}"
    return pd.read_csv(meta_csv)


@pytest.fixture(scope="module")
def top_level_files():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "output_repeated_splits_test"
    assert data_dir.exists(), f"Target directory does not exist: {data_dir}"
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == '.mp4'])


def test_metadata_columns_and_no_duplicates(meta_data):
    """Metadata must contain required ID columns and not contain duplicate ID pairs."""
    assert "Dialogue_ID" in meta_data.columns and "Utterance_ID" in meta_data.columns, \
        "Metadata CSV must contain 'Dialogue_ID' and 'Utterance_ID' columns"

    dialogue_nums = pd.to_numeric(meta_data["Dialogue_ID"], errors="coerce")
    utterance_nums = pd.to_numeric(meta_data["Utterance_ID"], errors="coerce")

    pairs = [ (int(d), int(u)) for d, u in zip(dialogue_nums, utterance_nums) if not pd.isna(d) and not pd.isna(u) ]
    if not pairs:
        pytest.skip("No valid (Dialogue_ID,Utterance_ID) pairs found in metadata")

    from collections import Counter
    counts = Counter(pairs)
    duplicates = {pair:cnt for pair,cnt in counts.items() if cnt > 1}
    if duplicates:
        msgs = [f"{len(duplicates)} duplicate (Dialogue_ID,Utterance_ID) pairs found in metadata:"]
        for (d,u),cnt in list(duplicates.items())[:200]:
            msgs.append(f"  dia{d}_utt{u}.mp4 appears {cnt} times in metadata")
        pytest.fail("\n".join(msgs))


# New focused duplicate check: fails only on duplicate ID pairs
def test_no_duplicate_meta_pairs(meta_data):
    """Fail if the metadata contains duplicate (Dialogue_ID,Utterance_ID) pairs.

    This is a focused, single-purpose test so failures are easy to read.
    """
    assert "Dialogue_ID" in meta_data.columns and "Utterance_ID" in meta_data.columns, \
        "Metadata CSV must contain 'Dialogue_ID' and 'Utterance_ID' columns"

    dialogue_nums = pd.to_numeric(meta_data["Dialogue_ID"], errors="coerce")
    utterance_nums = pd.to_numeric(meta_data["Utterance_ID"], errors="coerce")
    pairs = [ (int(d), int(u)) for d, u in zip(dialogue_nums, utterance_nums) if not pd.isna(d) and not pd.isna(u) ]

    from collections import Counter
    counts = Counter(pairs)
    duplicates = {pair:cnt for pair,cnt in counts.items() if cnt > 1}

    if duplicates:
        msgs = [f"{len(duplicates)} duplicate (Dialogue_ID,Utterance_ID) pairs found in metadata:"]
        for (d,u),cnt in list(duplicates.items())[:200]:
            msgs.append(f"  dia{d}_utt{u}.mp4 appears {cnt} times in metadata")
        pytest.fail("\n".join(msgs))


def test_file_id_parsing(top_level_files):
    """Ensure get_ids_for_file parses each top-level filename into valid integer IDs."""
    errors = []
    for p in top_level_files:
        name = p.name
        try:
            di, ui = get_ids_for_file(name)
        except Exception as e:
            errors.append((name, f"exception: {e!r}"))
            continue
        if not (isinstance(di, int) and isinstance(ui, int)):
            errors.append((name, f"invalid types: {di!r}, {ui!r}"))
        elif di < 0 or ui < 0:
            errors.append((name, f"invalid ids: {di}, {ui}"))

    if errors:
        msgs = [f"{len(errors)} files failed id parsing:"]
        msgs += [f"  {fn}: {err}" for fn, err in errors[:200]]
        pytest.fail("\n".join(msgs))


def test_metadata_vs_files_pairs(meta_data, top_level_files):
    """Compare sets of (Dialogue_ID,Utterance_ID) in metadata and in filenames.
    Fail if any pair exists only on one side.
    """
    # build file pairs
    file_pairs = set()
    parse_errors = []
    for p in top_level_files:
        try:
            di, ui = get_ids_for_file(p.name)
        except Exception as e:
            parse_errors.append((p.name, repr(e)))
            continue
        if isinstance(di, int) and isinstance(ui, int) and di >= 0 and ui >= 0:
            file_pairs.add((di, ui))
        else:
            parse_errors.append((p.name, f"invalid ids: {(di, ui)!r}"))

    # build meta pairs
    dialogue_nums = pd.to_numeric(meta_data["Dialogue_ID"], errors="coerce")
    utterance_nums = pd.to_numeric(meta_data["Utterance_ID"], errors="coerce")
    meta_pairs = set((int(d), int(u)) for d, u in zip(dialogue_nums, utterance_nums) if not pd.isna(d) and not pd.isna(u))

    meta_only = sorted(list(meta_pairs - file_pairs))
    files_only = sorted(list(file_pairs - meta_pairs))

    msgs = []
    if meta_only:
        msgs.append(f"{len(meta_only)} (dialogue,utterance) pairs are present in metadata but missing as files:")
        msgs += [f"  dia{d}_utt{u}.mp4" for d, u in meta_only[:200]]
    if files_only:
        msgs.append(f"{len(files_only)} (dialogue,utterance) pairs are present as files but missing in metadata:")
        msgs += [f"  dia{d}_utt{u}.mp4" for d, u in files_only[:200]]
    if parse_errors:
        msgs.append("Some files couldn't be parsed into (dialogue,utterance) pairs:")
        msgs += [f"  {name}: {err}" for name, err in parse_errors[:200]]

    if msgs:
        pytest.fail("\n".join(msgs))


def test_metadata_pairs_have_corresponding_files(meta_data, top_level_files):
    """Fail if any (Dialogue_ID,Utterance_ID) in metadata has no corresponding top-level file.

    This test focuses only on metadata->file coverage and gives a concise failure message
    when metadata expects files that are missing on disk.
    """
    # build set of file pairs
    file_pairs = set()
    for p in top_level_files:
        try:
            di, ui = get_ids_for_file(p.name)
        except Exception:
            continue
        if isinstance(di, int) and isinstance(ui, int) and di >= 0 and ui >= 0:
            file_pairs.add((di, ui))

    # build set of meta pairs
    dialogue_nums = pd.to_numeric(meta_data["Dialogue_ID"], errors="coerce")
    utterance_nums = pd.to_numeric(meta_data["Utterance_ID"], errors="coerce")
    meta_pairs = set((int(d), int(u)) for d, u in zip(dialogue_nums, utterance_nums) if not pd.isna(d) and not pd.isna(u))

    # compute metadata pairs missing as files
    missing_meta_pairs = sorted(meta_pairs - file_pairs)
    if missing_meta_pairs:
        msgs = [f"{len(missing_meta_pairs)} metadata (Dialogue_ID,Utterance_ID) pairs have no corresponding top-level file:"]
        for d, u in missing_meta_pairs[:200]:
            msgs.append(f"  dia{d}_utt{u}.mp4")
        pytest.fail("\n".join(msgs))


def test_get_label_for_files(meta_data, top_level_files, tmp_path):
    """Call get_label_for_file for each file and ensure labels are valid.

    Writes a CSV to tmp_path with columns: filename, dialog_id, utterance_id, split, label
    """
    allowed_labels = {"neutral", "positive", "negative"}
    rows = []
    errors = []
    invalid = []

    for p in top_level_files:
        name = p.name
        try:
            dia_id, utt_id = get_ids_for_file(name)
        except Exception as e:
            errors.append((name, 'get_ids_for_file', repr(e)))
            dia_id, utt_id = -1, -1

        try:
            label = get_label_for_file(name, meta_data)
        except Exception as e:
            errors.append((name, 'get_label_for_file', repr(e)))
            label = f"ERROR:{repr(e)}"

        rows.append({
            "filename": name,
            "dialog_id": int(dia_id) if isinstance(dia_id, int) else dia_id,
            "utterance_id": int(utt_id) if isinstance(utt_id, int) else utt_id,
            "split": "output_repeated_splits_test",
            "label": str(label),
        })

        if not str(label).startswith("ERROR:") and str(label) not in allowed_labels:
            invalid.append((name, label))

    out_csv = tmp_path / "output_labels.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    msgs = []
    if errors:
        msgs.append(f"{len(errors)} files raised exceptions:")
        msgs += [f"  {fn}: {fnc} -> {err}" for fn, fnc, err in errors[:200]]
    if invalid:
        msgs.append(f"{len(invalid)} files have invalid labels (allowed: {allowed_labels}):")
        msgs += [f"  {fn}: {lab}" for fn, lab in invalid[:200]]

    if msgs:
        pytest.fail("\n".join(msgs))

    # successful run writes CSV for manual inspection
    print(f"Labels CSV written to: {out_csv}")
