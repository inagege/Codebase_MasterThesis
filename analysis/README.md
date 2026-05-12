# Analysis Scripts Overview

This folder contains analysis and plotting scripts grouped by topic.

## General Notes

- Run scripts from the repository root so relative paths like `out/...` and `analysis/out/...` resolve correctly.
- Most scripts expose options via `--help`.
- Generated artifacts are typically written under `analysis/out/...`.

## `heatmap_analysis/`

Core heatmap generation scripts for perturbation and method comparisons.

- `plot_qaa_vs_baseline_multimodal_delta_heatmaps.py`  
  Compares `qwen_scored` vs baseline on multimodal settings and plots delta heatmaps (accuracy / macro-F1).
- `plot_single_modality_perturbation_heatmaps.py`  
  Builds heatmaps for single-modality perturbations across models and severities.
- `plot_two_modality_one_modified_heatmaps.py`  
  Builds heatmaps for two-modality settings where one modality is perturbed; also exports a LaTeX table.
- `plot_qwen_scored_placebo_vs_baseline_sentiment_emotion_heatmaps.py`  
  Compares `qwen_scored` and `placebo` against baseline for sentiment/emotion at a selected severity.
- `plot_forced_quality_delta_by_dataset_models.py`  
  Plots forced-quality deltas per dataset and model, with CSV export.

## `quality_score_analysis/`

Scripts focused on quality-score distributions, consistency, and correlation with performance.

- `analyze_quality_score_distributions_by_noise.py`  
  Analyzes quality-score distributions by perturbation/noise.
- `analyze_quality_score_distributions_calibration.py`  
  Analyzes calibrated quality-score distributions and percentile behavior.
- `compare_quality_scores_across_methods.py`  
  Compares score distributions across multiple scoring methods.
- `check_unmodified_quality_consistency.py`  
  Checks whether unmodified-quality scores remain consistent across files/settings.
- `plot_two_modality_one_modified_quality_delta_heatmap.py`  
  Heatmap view of quality-score deltas in two-modality one-modified setups.
- `correlate_quality_vs_degradation.py`  
  Correlates quality scores with downstream performance degradation.

## `per_dataset_analysis/`

Dataset-level preparation and metric plotting utilities.

- `data_preparation_util.py`  
  Prepares per-split/per-class metric tables from prediction CSVs.
- `plotting_util.py`  
  Produces per-dataset plots (accuracy and classwise metrics).
- `compare_classification_methods.py`  
  Paired comparison of baseline vs candidate predictions, with summary tables and optional plots.
- `data_explotation.ipynb`  
  Notebook for exploratory analysis and ad-hoc inspection.

## `plots_for_slides/`

Slide-oriented plotting scripts (curated outputs for presentations).

- `plot_qwen7b_qaa_delta_barplot.py`  
  QAA vs baseline delta bar plot (Qwen 7B).
- `plot_qwen7b_worst_degradation_barplot.py`  
  Worst perturbation degradation summary bar plot (Qwen 7B).
- `plot_qwen7b_forced_quality_delta_heatmaps.py`  
  Forced-quality delta heatmaps for selected tasks/modalities (Qwen 7B).
- `plot_placebo_rescaled_seed_vs_baseline_sentiment_emotion_heatmaps.py`  
  Seeded placebo-rescaled vs baseline heatmaps for sentiment/emotion.

## `attention_mass_analysis/`

Attention-focused analysis scripts for QAA behavior.

- `analyze_first_layer_qaa_attention_shift.py`  
  Computes first-layer attention shifts under QAA scaling.
- `analyze_audio_video_attention_mass.py`  
  Summarizes attention mass allocated to audio vs video tokens.
- `analyze_qaa_score_sensitivity.py`  
  Consolidated sensitivity checks for prediction changes and attention-level effects.

## Shared Helper

- `result_plot_parsing.py`  
  Shared parsing helpers for prediction filenames and split metadata used by multiple scripts.

