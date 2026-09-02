#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=siglip_direct
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf
cd $D
$D/cuda_guard.sh $D/.venv/bin/python scripts/longqa_siglip_direct.py   --gold $D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl   --frames $D/cache/egolongqa_dense --topk 3   --out $D/out/lq_siglip_direct.jsonl
