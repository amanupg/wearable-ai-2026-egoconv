#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=11:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=egoconv_small
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$D"
.venv/bin/python scripts/run_egoconv_local.py \
  --gold scripts/wearable_ai_2026_egoconv_val_700.jsonl \
  --frames "$D/cache/egoconv" --model "$D/models/internvl3_5-1b" \
  --out "$D/out/egoconv_small_s${SHARD}.jsonl" \
  --style commit --recent 6 --history 2 --batch 6 \
  --shard "${SHARD}" --num-shards 2
