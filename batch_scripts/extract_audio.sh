#!/bin/bash
#SBATCH --job-name=extract_audio            # Job name
#SBATCH --output=logs/%x_%j.out           # Stdout log
#SBATCH --error=logs/%x_%j.err            # Stderr log
#SBATCH --time=2:00:00                   # Max runtime (hh:mm:ss)
#SBATCH --gres=gpu:full:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8                 # CPU cores
#SBATCH --mem=100G                        # RAM
#SBATCH --partition=normal                # Partition name

# Go to your project directory
cd /hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis || exit 1

# (Optional but recommended) create logs directory if it doesn't exist
mkdir -p logs

# Run your Python script
pixi run python utils/extract_audio_only.py \
  --input-dir data/MELD.Raw/output_repeated_splits_test/unmodified \
  --sr 16000 --channels 1 --overwrite
