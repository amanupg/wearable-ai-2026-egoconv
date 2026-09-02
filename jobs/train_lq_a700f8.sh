#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=11:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=lq_a700f8
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/depts/edu/users/au2327/wearable/logs_lq_a700f8.txt
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D
$D/cuda_guard.sh $D/.venv/bin/python scripts/train_longqa_lora.py   --model $D/models/internvl3_5-1b   --gold $D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl   --frames $D/cache/egolongqa896   --out $D/runs/lq_small_a700f8   --rank 32 --epochs 1 --n-frames 8 --no-balance --class-weight --save-every 200 --holdout 0
