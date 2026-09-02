#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=lq_small
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D
.venv/bin/python scripts/run_egolongqa.py   --gold $D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl   --frames $D/cache/egolongqa896   --backend local --model $D/models/internvl3_5-1b   --n-frames 16 --batch 1   --out $D/out/egolongqa_small_1b.jsonl
