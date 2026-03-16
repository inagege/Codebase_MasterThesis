from __future__ import annotations

import argparse
import csv
from pathlib import Path

from quality_calibration import (
    SUPPORTED_MODALITIES,
    build_percentile_calibration_payload,
    save_percentile_calibration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build percentile-based modality calibration JSON from score CSV files."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help=(
            "Input CSV path(s). Can be provided multiple times. "
            "Expected score columns: <modality>_raw_quality (preferred), "
            "then <modality>_quality, then <modality>."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output calibration JSON path.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="text,audio,image,video",
        help="Comma-separated modalities to include.",
    )
    return parser.parse_args()


def _parse_modalities(modalities_arg: str) -> list[str]:
    modalities = [token.strip().lower() for token in modalities_arg.split(",") if token.strip()]
    bad = sorted(set(modalities) - set(SUPPORTED_MODALITIES))
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Supported: {list(SUPPORTED_MODALITIES)}")
    if not modalities:
        raise ValueError("No modalities selected for calibration build.")
    return modalities


def _extract_score(row: dict[str, str], modality: str):
    candidates = [
        f"{modality}_raw_quality",
        f"{modality}_quality",
        modality,
    ]
    for column in candidates:
        raw_value = row.get(column)
        if raw_value is None:
            continue
        value = raw_value.strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _read_scores_from_csv(csv_path: Path, modalities: list[str]) -> dict[str, list[float]]:
    scores = {modality: [] for modality in modalities}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for modality in modalities:
                value = _extract_score(row, modality)
                if value is not None:
                    scores[modality].append(value)
    return scores


def main():
    args = parse_args()
    modalities = _parse_modalities(args.modalities)

    all_scores = {modality: [] for modality in modalities}
    input_paths = [Path(path) for path in args.input_csv]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        file_scores = _read_scores_from_csv(path, modalities)
        for modality in modalities:
            all_scores[modality].extend(file_scores[modality])

    payload = build_percentile_calibration_payload(
        all_scores,
        source_paths=[str(path) for path in input_paths],
    )
    save_percentile_calibration(args.out_path, payload)

    print(f"[INFO] Saved calibration file: {args.out_path}")
    for modality in modalities:
        print(f"[INFO] {modality}: {len(all_scores[modality])} raw scores", flush=True)


if __name__ == "__main__":
    main()
