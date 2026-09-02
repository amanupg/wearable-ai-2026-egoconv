#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=proactive
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf
cd $D
$D/cuda_guard.sh $D/.venv/bin/python scripts/proactive_causal.py --embed --train   --gold $D/data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl   --frames $D/cache/egoproactive   --cache $D/out/proactive_siglip.pkl --out $D/out/proactive_clf.pkl
