#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=11:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_r32e3
#SBATCH --output=/insomnia001/depts/edu/users/au2327/wearable/logs_eval_r32e3.txt
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D
$D/cuda_guard.sh $D/.venv/bin/python scripts/run_egoconv_local.py   --gold $D/scripts/wearable_ai_2026_egoconv_val_700.jsonl   --frames $D/cache/egoconv --model $D/models/internvl3_5-1b   --adapter $D/runs/egoconv_small_lora_r32e3   --out $D/out/egoconv_small_r32e3.jsonl   --style commit --recent 6 --history 2 --batch 6
