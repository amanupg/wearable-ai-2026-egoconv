#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=11:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=egoconv_sft
#SBATCH --exclude=ins082
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D
.venv/bin/python scripts/train_egoconv_lora.py   --model $D/models/internvl3_5-1b   --gold $D/scripts/wearable_ai_2026_egoconv_val_700.jsonl   --frames $D/cache/egoconv   --out $D/runs/egoconv_small_r32e1   --rank 32 --epochs 1 --batch 1 --accum 8 --recent 6 --history 2 --workers 4 --save-every 300
