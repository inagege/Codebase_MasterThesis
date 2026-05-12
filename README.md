# Project Setup

This project uses [Pixi](https://pixi.sh) to manage environments and dependencies.

## 1. Install Pixi (if not installed yet)

Check whether Pixi is already available:

```bash
pixi --version
```

If the command is not found, install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your terminal (or reload your shell config) so `pixi` is on `PATH`.

## 2. Enter the Pixi environment

From the project root (where `pixi.toml` is located), run:

```bash
pixi shell
```

Pixi will create/sync the environment and open a shell with all project dependencies.

## 3. Run commands inside Pixi

Examples:

```bash
pixi run python --version
pixi run python benchmark_modalities.py
```

## 5. Set Up PAM for Audio Quality Assessment

For experiments with calibrated quality scores, the audio quality scorer expects a local `PAM` repository at:

```bash
./PAM
```

From the project root, run:

```bash
git clone https://github.com/soham97/PAM.git PAM
pixi run pip install -r PAM/requirements.txt
```

```bash
pixi run python -c "from PAM import PAM; print('PAM import OK')"
```

# Run Experiments

## Benchmark Unmodified Model

Use `benchmark_modalities.py` to run the base Qwen model without Quality Aware Attention.

Example:

```bash
pixi run python benchmark_modalities.py \
  --dataset meld \
  --classification-task emotion \
  --split test \
  --modalities text,audio,video \
  --batch-size 4
```

Useful options:
- `--noisy-modalities audio,image,text,video` to load noisy variants for selected modalities
- `--noise-severity <S>` to filter noisy variants to a specific severity
- `--stratified-samples <N>`, `--total-samples <N>`, `--start-at-sample <idx>` to control evaluation size
- `--qwen-model-id <hf-model-id>` to switch checkpoints (default: `Qwen/Qwen2.5-Omni-7B`)

Outputs are written to `out/...` (predictions and error rows), unless overridden with `--out-path` and `--out-error-path`.

## Benchmark Model with Quality Aware Attention

Use `benchmark_scored_modalities.py` to run the model with Quality Aware Attention (QAA), where modality quality scores are used to scale first-layer attention.

Example (calibrated quality scores):

```bash
pixi run python benchmark_scored_modalities.py \
  --dataset imdb \
  --split test \
  --modalities text,audio,video \
  --quality-calibration \
  --batch-size 4
```

Quality scoring modes:
- Add `--qwen-quality` to estimate quality scores with Qwen (cannot be combined with `--quality-calibration`)
- Add `--quality-calibration` to use ecdf calibrated quality scores (cannot be combined with `--qwen-quality`)

Extra QAA controls:
- `--qaa-normalization-mode global|exclude_unscaled` depending on whether quality scores are normalized across all samples or only among the scaled modalities for each sample
- `--force-quality-scores-one` or `--force-modality-quality-scores text=0.2,audio=0.9`
- `--quality-placebo-random --quality-placebo-random-seed <seed>` for placebo runs

This script writes:
- prediction CSV (`--out-path`)
- error CSV (`--out-error-path`)
- per-sample quality score CSV (`--quality-score-out-path`)
