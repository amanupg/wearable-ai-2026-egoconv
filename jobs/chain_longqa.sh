#!/usr/bin/env bash
# Wait for the egolongqa download to finish, then extract frames for all 700.
# Extraction is resumable, so videos already done are skipped.
D=/insomnia001/depts/edu/users/au2327/wearable
cd "$D"
while pgrep -u "$USER" -f "hf download facebook/wearable-ai" >/dev/null; do sleep 60; done
echo "download finished at $(date)"
du -sh "$D/data/egolongqa"
ls "$D/data/egolongqa/val"/*.mp4 | wc -l
.venv/bin/python scripts/extract_longqa_frames.py \
  --videos "$D/data/egolongqa/val" --out "$D/cache/egolongqa" \
  --gold "$D/data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
  --frames 32 --height 448 --jobs 48
echo "extraction finished at $(date)"
