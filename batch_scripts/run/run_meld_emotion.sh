#!/bin/bash
#SBATCH --job-name=meld                   # Job name
#SBATCH --output=logs/%x_%j.out           # Stdout log
#SBATCH --error=logs/%x_%j.err            # Stderr log
#SBATCH --time=72:00:00                   # Max runtime (hh:mm:ss)
#SBATCH --gres=gpu:full:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8                 # CPU cores
#SBATCH --mem=10G                         # RAM
#SBATCH --partition=normal                # Partition name

module load devel/cuda/12.9

# Go to your project directory
cd /hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis || exit 1

# (Optional but recommended) create logs directory if it doesn't exist
mkdir -p logs

# Run your Python script
pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities video \
  --noisy-modalities video \
  --classification-task sentiment \
  --noise-severity 3 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio \
  --noisy-modalities audio \
  --classification-task sentiment \
  --noise-severity 3 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio,video \
  --noisy-modalities audio \
  --classification-task sentiment \
  --noise-severity 3 \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio,video \
  --noisy-modalities video \
  --classification-task sentiment \
  --noise-severity 3 \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B


pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities video \
  --noisy-modalities video \
  --classification-task sentiment \
  --noise-severity 5 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio \
  --noisy-modalities audio \
  --classification-task sentiment \
  --noise-severity 5 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio,video \
  --noisy-modalities audio \
  --classification-task sentiment \
  --noise-severity 5 \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio,video \
  --noisy-modalities video \
  --classification-task sentiment \
  --noise-severity 5 \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B


pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities video \
  --classification-task sentiment \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio,video \
  --classification-task sentiment \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B

pixi run python benchmark_modalities.py \
  --dataset meld \
  --modalities audio \
  --classification-task sentiment \
  --stratified-samples 5000 \
  --batch-size 8 \
  --qwen-model-id Qwen/Qwen2.5-Omni-3B