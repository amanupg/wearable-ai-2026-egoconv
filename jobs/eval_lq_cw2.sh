#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ins082
D=/insomnia001/depts/edu/users/au2327/wearable
export HF_HOME=$D/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D
$D/cuda_guard.sh $D/.venv/bin/python scripts/run_egolongqa.py   --gold $D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl   --frames $D/cache/egolongqa896 --backend local   --model $D/models/internvl3_5-1b --adapter $D/runs/$RUNDIR   --n-frames ${NFRAMES:-8} --batch 4 --out $D/out/$OUTNAME
