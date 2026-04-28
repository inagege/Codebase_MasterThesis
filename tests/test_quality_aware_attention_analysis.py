from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("transformers")

from utils.qaa.quality_aware_attention import (
    QAA_NORMALIZATION_EXCLUDE_UNSCALED,
    QAA_NORMALIZATION_GLOBAL,
    compute_quality_adjusted_attention_weights,
)


def test_compute_quality_adjusted_attention_weights_global():
    attention_weights = torch.tensor([[[[0.2, 0.3, 0.5]]]], dtype=torch.float32)
    quality_scores = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)

    adjusted = compute_quality_adjusted_attention_weights(
        attention_weights,
        quality_scores,
        quality_normalization_mode=QAA_NORMALIZATION_GLOBAL,
    )

    expected = torch.tensor([[[[0.1904762, 0.5714286, 0.2380952]]]], dtype=torch.float32)
    assert torch.allclose(adjusted, expected, atol=1e-6)
    assert torch.allclose(adjusted.sum(dim=-1), torch.ones_like(adjusted.sum(dim=-1)), atol=1e-6)


def test_compute_quality_adjusted_attention_weights_exclude_unscaled():
    attention_weights = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=torch.float32)
    quality_scores = torch.tensor([1.0, 0.5, 2.0, 1.0], dtype=torch.float32)
    scaled_mask = torch.tensor([False, True, True, False])

    adjusted = compute_quality_adjusted_attention_weights(
        attention_weights,
        quality_scores,
        quality_scaled_token_mask=scaled_mask,
        quality_normalization_mode=QAA_NORMALIZATION_EXCLUDE_UNSCALED,
    )

    expected = torch.tensor([[[[0.1, 0.07142857, 0.42857143, 0.4]]]], dtype=torch.float32)
    assert torch.allclose(adjusted, expected, atol=1e-6)

    unscaled_indices = torch.tensor([0, 3])
    scaled_indices = torch.tensor([1, 2])
    assert torch.allclose(
        adjusted[..., unscaled_indices],
        attention_weights[..., unscaled_indices],
        atol=1e-6,
    )
    assert torch.allclose(
        adjusted[..., scaled_indices].sum(dim=-1),
        attention_weights[..., scaled_indices].sum(dim=-1),
        atol=1e-6,
    )
    assert torch.allclose(adjusted.sum(dim=-1), torch.ones_like(adjusted.sum(dim=-1)), atol=1e-6)


def test_compute_quality_adjusted_attention_weights_exclude_unscaled_requires_mask():
    attention_weights = torch.tensor([[[[0.2, 0.8]]]], dtype=torch.float32)
    quality_scores = torch.tensor([0.1, 0.9], dtype=torch.float32)

    with pytest.raises(ValueError):
        compute_quality_adjusted_attention_weights(
            attention_weights,
            quality_scores,
            quality_normalization_mode=QAA_NORMALIZATION_EXCLUDE_UNSCALED,
        )
