from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.benchmark_data_loading import filter_samples_by_sample_id, select_stratified_samples


def _build_samples(class_counts: dict[str, int]):
    samples = []
    for label, count in class_counts.items():
        for idx in range(count):
            sample_id = f"{label}_{idx:03d}"
            samples.append(
                {
                    "dataset": "imdb",
                    "split": "all",
                    "sample_id": sample_id,
                    "file": f"{sample_id}.jpg",
                    "label": label,
                }
            )
    return samples


def test_select_stratified_samples_is_deterministic():
    samples = _build_samples({"action": 60, "comedy": 30, "romance": 10})
    first = select_stratified_samples(samples, 25)
    second = select_stratified_samples(samples, 25)

    assert [sample["sample_id"] for sample in first] == [sample["sample_id"] for sample in second]


def test_select_stratified_samples_respects_distribution():
    samples = _build_samples({"action": 60, "comedy": 30, "romance": 10})
    selected = select_stratified_samples(samples, 20)
    label_counts = Counter(sample["label"] for sample in selected)

    assert len(selected) == 20
    assert label_counts == Counter({"action": 12, "comedy": 6, "romance": 2})


def test_select_stratified_samples_rejects_non_positive_limits():
    samples = _build_samples({"action": 5, "comedy": 5})
    with pytest.raises(ValueError):
        select_stratified_samples(samples, 0)


def test_select_stratified_samples_returns_all_when_limit_exceeds_count():
    samples = _build_samples({"action": 4, "comedy": 3})
    selected = select_stratified_samples(samples, 99)
    assert selected == samples


def test_filter_samples_by_sample_id_keeps_all_variants_for_selected_base_ids():
    base_samples = [
        {"sample_id": "a", "label": "x"},
        {"sample_id": "b", "label": "x"},
        {"sample_id": "c", "label": "y"},
        {"sample_id": "d", "label": "y"},
    ]
    selected_base = select_stratified_samples(base_samples, 2)
    selected_ids = {sample["sample_id"] for sample in selected_base}

    noisy_samples = [
        {"sample_id": "a", "split": "all_noise1"},
        {"sample_id": "b", "split": "all_noise1"},
        {"sample_id": "c", "split": "all_noise1"},
        {"sample_id": "d", "split": "all_noise1"},
        {"sample_id": "a", "split": "all_noise2"},
        {"sample_id": "b", "split": "all_noise2"},
        {"sample_id": "c", "split": "all_noise2"},
        {"sample_id": "d", "split": "all_noise2"},
    ]
    filtered = filter_samples_by_sample_id(noisy_samples, selected_ids)

    assert {sample["sample_id"] for sample in filtered} == selected_ids
    assert len(filtered) == len(selected_ids) * 2
