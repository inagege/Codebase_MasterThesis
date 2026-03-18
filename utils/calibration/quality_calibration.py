from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_MODALITIES = ("text", "audio", "image", "video")


@dataclass(frozen=True)
class PercentileCalibrator:
    sorted_scores: np.ndarray
    higher_is_better: bool = True

    def calibrate(self, raw_score: float) -> float:
        if self.sorted_scores.size < 1:
            raise ValueError(
                "PercentileCalibrator has no reference scores; cannot calibrate raw score."
            )

        rank_right = int(np.searchsorted(self.sorted_scores, raw_score, side="right"))
        percentile = (rank_right + 0.5) / (self.sorted_scores.size + 1.0)
        if not self.higher_is_better:
            percentile = 1.0 - percentile
        return float(np.clip(percentile, 0.0, 1.0))


def _sort_scores(scores) -> np.ndarray:
    return np.sort(scores)


def build_percentile_calibration_payload(
    modality_scores: dict[str, list[float]],
    *,
    source_paths: list[str] | None = None,
    higher_is_better_by_modality: dict[str, bool] | None = None,
) -> dict:
    modalities_payload = {}
    for modality, scores in sorted(modality_scores.items()):
        sorted_scores = _sort_scores(scores)
        if sorted_scores.size < 1:
            continue
        modalities_payload[modality] = {
            "sorted_scores": sorted_scores.tolist(),
            "higher_is_better": (
                bool(higher_is_better_by_modality.get(modality, True))
                if higher_is_better_by_modality is not None
                else True
            ),
            "num_scores": int(sorted_scores.size),
        }

    payload = {
        "method": "percentile_ecdf_v1",
        "modalities": modalities_payload,
    }
    if source_paths:
        payload["source_paths"] = [str(path) for path in source_paths]
    return payload


def save_percentile_calibration(path: str | Path, payload: dict):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_percentile_calibration(path: str | Path) -> dict[str, PercentileCalibrator]:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    method = payload.get("method")
    if method not in (None, "percentile_ecdf_v1"):
        raise ValueError(f"Unsupported calibration method {method!r} in {json_path}")

    modalities_payload = payload.get("modalities", payload)
    if not isinstance(modalities_payload, dict):
        raise ValueError(f"Invalid percentile calibration payload in {json_path}: missing modalities mapping.")

    calibrators: dict[str, PercentileCalibrator] = {}
    for modality, modality_payload in modalities_payload.items():
        if isinstance(modality_payload, dict):
            sorted_scores = modality_payload.get("sorted_scores")
            if sorted_scores is None:
                sorted_scores = modality_payload.get("scores")
            higher_is_better = bool(modality_payload.get("higher_is_better", True))
        else:
            sorted_scores = modality_payload
            higher_is_better = True

        cleaned_scores = _sort_scores(sorted_scores if sorted_scores is not None else [])
        if cleaned_scores.size < 1:
            continue
        calibrators[modality] = PercentileCalibrator(
            sorted_scores=cleaned_scores,
            higher_is_better=higher_is_better,
        )

    if not calibrators:
        raise ValueError(f"No valid modality calibrators found in {json_path}")

    return calibrators


def apply_percentile_calibration_to_batch(
    modality_scores_per_entry: list[dict[str, float]],
    calibrators: dict[str, PercentileCalibrator] | None,
) -> list[dict[str, float]]:
    if not calibrators:
        raise ValueError("Expected non-empty calibrators mapping, but none was provided.")

    calibrated = []
    calibrator_modalities = set(calibrators)

    for sample_idx, sample_scores in enumerate(modality_scores_per_entry):
        missing_modalities = sorted(set(sample_scores) - calibrator_modalities)
        if missing_modalities:
            raise ValueError(
                "Missing modality calibrators for sample index "
                f"{sample_idx}: {missing_modalities}. "
                f"Available calibrators: {sorted(calibrator_modalities)}"
            )

        calibrated_scores = {}
        for modality, raw_score in sample_scores.items():
            calibrator = calibrators[modality]
            calibrated_scores[modality] = calibrator.calibrate(raw_score)
        calibrated.append(calibrated_scores)

    return calibrated
