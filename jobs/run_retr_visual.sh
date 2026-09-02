#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=retr_visual
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf
cd $D
.venv/bin/python scripts/retrieve_frames.py   --gold $D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl   --frames $D/cache/egolongqa_dense   --query-mode visual --queries $D/out/visual_queries.json   --topk 16 --min-gap 3 --out $D/out/retrieved_visual.json
