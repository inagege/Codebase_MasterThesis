#!/usr/bin/env python3
import re
from collections import Counter
from pathlib import Path
import pandas as pd

repo = Path('.')
meta_csv = repo / 'data' / 'test_sent_emo.csv'
files_dir = repo / 'data' / 'output_repeated_splits_test'
meta = pd.read_csv(meta_csv)

# parse metadata pairs
if 'Dialogue_ID' not in meta.columns or 'Utterance_ID' not in meta.columns:
    print('Metadata missing required columns: Dialogue_ID, Utterance_ID')
    raise SystemExit(1)

dialogue = pd.to_numeric(meta['Dialogue_ID'], errors='coerce')
utter = pd.to_numeric(meta['Utterance_ID'], errors='coerce')
meta_pairs_list = [(int(d), int(u)) for d, u in zip(dialogue, utter) if not pd.isna(d) and not pd.isna(u)]
meta_pairs = set(meta_pairs_list)

# duplicate checking
meta_counts = Counter(meta_pairs_list)
duplicates = {pair:cnt for pair,cnt in meta_counts.items() if cnt > 1}

# collect top-level files only (no subdirs)
files = sorted([p for p in files_dir.iterdir() if p.is_file() and p.suffix.lower()=='.mp4'])

def parse_fname(name):
    m = re.search(r'dia(\d+)_utt(\d+)', name, flags=re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None

file_pairs_list = [parse_fname(p.name) for p in files]
file_pairs = {p for p in file_pairs_list if p is not None}
file_parse_errors = [p.name for p,ids in zip(files,file_pairs_list) if ids is None]
file_names_for_pairs = {parse_fname(p.name): p.name for p in files if parse_fname(p.name) is not None}

# set differences
meta_only = sorted(list(meta_pairs - file_pairs))
files_only = sorted(list(file_pairs - meta_pairs))

print('total_files =', len(files))
print('meta_rows =', len(meta))
print('unique_meta_pairs =', len(meta_pairs))
print('unique_file_pairs =', len(file_pairs))
print('duplicate_meta_pairs_count =', len(duplicates))
print('meta_only_count =', len(meta_only))
print('files_only_count =', len(files_only))
if duplicates:
    print('\nduplicate examples (pair -> count):')
    for (d,u),c in list(duplicates.items())[:50]:
        print(f'  dia{d}_utt{u} -> {c}')
if meta_only:
    print('\nmeta_only examples (in metadata but not as files):')
    for d,u in meta_only[:50]:
        print(f'  dia{d}_utt{u}.mp4')
if files_only:
    print('\nfiles_only examples (files with no matching metadata pair):')
    for d,u in files_only[:200]:
        print(f'  dia{d}_utt{u}.mp4 -> filename: {file_names_for_pairs.get((d,u))}')
if file_parse_errors:
    print('\nfile parse errors (filenames not matching pattern):')
    for n in file_parse_errors[:50]:
        print(' ', n)
