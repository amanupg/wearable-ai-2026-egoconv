#!/usr/bin/env python
"""EgoConv inference with a local <=2B VLM on a single GPU (Insomnia).

This is the small-division counterpart to run_egoconv.py. The frame-selection
strategy and prompt are deliberately identical to the 32B OpenRouter run so the
two divisions differ only in the model -- that keeps the comparison clean and
means any prompt improvement found on one transfers to the other.

Division cap is on TOTAL params and is strict (0 < total <= 2e9). Note that every
model marketed as "2B" actually exceeds it: Qwen3-VL-2B is 2.128B, Qwen2-VL-2B is
2.209B, InternVL3.5-2B is 2.348B, SmolVLM2-2.2B is 2.247B. InternVL3.5-1B is
1.061B and is what this defaults to.

THROUGHPUT
----------
Turns within a video are strictly sequential (turn i conditions on the model's own
answer to turn i-1), so the only axis to batch on is across videos. We run B videos
in lockstep: batch all their turn-0 prompts, then all their turn-1 prompts, and so
on, with videos dropping out as they run out of turns. On a 48 GB A6000 a 1B model
leaves plenty of room for B=16.

Resumable: completed videos are appended as they finish and skipped on restart.

Usage:
    python run_egoconv_local.py --limit 4 --batch 4     # smoke test
    python run_egoconv_local.py --batch 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "egoconv" / "wearable_ai_2026_egoconv_val_700.jsonl"
FRAMES = ROOT / "cache" / "frames" / "egoconv"

SYSTEM = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see, in chronological order. Answer their question directly and "
    "conversationally, as a helpful assistant speaking aloud.\n"
    "Be specific and concrete -- name what you actually see. Aim for one to two "
    "sentences (about 25-30 words). Do not describe the images as images, do not "
    "mention frames or timestamps, and never say you cannot see. If the question "
    "does not need the video, just answer it directly."
)


# A 1B model ignores negative instructions. The v1 prompt said "never say you
# cannot see" and the model opened 63% of its answers with exactly that, scoring
# 0.0 on every one and 1.0 on none. Measured: generic hedging scores 0.003 with
# this judge, so a refusal is worth nothing while a confident wrong guess at least
# has a chance at partial credit.
#
# So: no prohibitions, only a demonstrated pattern. Small models imitate far
# better than they comply.
SYSTEM_COMMIT = (
    "You are a wearable AI assistant. The images show what the user sees through "
    "smart glasses, in time order. Answer their question in one or two short "
    "sentences, as if speaking to them.\n"
    "Always give a definite answer. Make your best guess from what is visible and "
    "state it plainly, even when you are unsure. Name specific objects.\n\n"
    "Examples of the required style:\n"
    "Q: What is the name of this exhibit?\n"
    "A: This exhibit is called Treasures.\n"
    "Q: Does this look shaggy?\n"
    "A: Yes, the dough looks shaggy and uneven, so it is ready for kneading.\n"
    "Q: How long should I let this rise?\n"
    "A: Let it rise about 1 to 2 hours, until it has roughly doubled in size."
)


# Exact copy of the fine-tuning system prompt. Train/inference prompt drift
# cost 20% empty answers on the 8B run.
SYSTEM_SFT = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see, in chronological order. Answer their question directly and "
    "conversationally, as a helpful assistant speaking aloud."
)

STYLES = {"v1": SYSTEM, "commit": SYSTEM_COMMIT, "sft": SYSTEM_SFT}


def pick_frames(frame_dir: Path, intervals, turn: int, n_recent: int, n_history: int):
    """Dense frames from the current interval + sparse frames from earlier ones.

    Cache is 1 fps on real container time, so second S -> frame index S.
    `duration_in_sec` in the annotations is wrong for 688/700 videos and is
    never consulted; intervals index the true container length.
    """
    # Ignore macOS AppleDouble sidecars ("._000001.jpg"). They are created by
    # tar on macOS during transfer, and because they sort alongside the real
    # frames they would silently shift the second->frame index mapping.
    avail = sorted(p for p in frame_dir.glob("*.jpg") if not p.name.startswith("."))
    if not avail:
        return []

    def at(sec):
        i = int(round(sec))
        return avail[i] if 0 <= i < len(avail) else None

    def span(lo, hi, k):
        if k <= 0 or hi <= lo:
            return []
        step = (hi - lo) / k
        out = []
        for i in range(k):
            f = at(lo + step * (i + 0.5))
            if f is not None and f not in out:
                out.append(f)
        return out

    start, end = intervals[turn]
    recent = span(start, end, n_recent)
    history = []
    if turn > 0:
        history = [f for f in span(intervals[0][0], intervals[turn - 1][1], n_history)
                   if f not in recent]
    return sorted(set(history + recent), key=lambda p: p.name)


def build_chat(row, turn, prior, frames, system=SYSTEM):
    content = [{"type": "image", "image": str(f)} for f in frames]
    convo = ""
    for i in range(turn):
        convo += f"User: {row['questions'][i]}\nYou: {prior[i]}\n"
    convo += f"User: {row['questions'][turn]}"
    content.append({"type": "text", "text": convo})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "internvl3_5-1b"))
    ap.add_argument("--recent", type=int, default=10)
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=90)
    ap.add_argument("--batch", type=int, default=16, help="videos in lockstep")
    ap.add_argument("--adapter", default="",
                    help="LoRA adapter dir; merged into the base for inference")
    ap.add_argument("--style", default="v1", choices=list(STYLES))
    ap.add_argument("--limit", type=int, default=0)
    # Shard across GPUs: each job takes every Nth video, so the two halves finish
    # together even though per-video cost varies with turn count.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "out" / "egoconv_small.jsonl"))
    # The cluster copy has a flatter layout than the laptop, so both inputs are
    # overridable rather than hardcoded to this repo's tree.
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--frames", default=str(FRAMES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    frames_root = Path(args.frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in gold_path.open() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["video_path"])
            except Exception:
                pass
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    todo = [r for r in rows if r["video_path"] not in done]
    print(f"{len(todo)} videos to do ({len(done)} cached) | batch={args.batch} | style={args.style} | shard {args.shard}/{args.num_shards}", flush=True)
    if not todo:
        return

    print(f"loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        # Merge so inference runs at base speed and the param count reported below
        # is the real deployed total (what gets declared to the leaderboard).
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter from {args.adapter} (merged)", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {n_params:,} params ({n_params/1e9:.3f}B) "
          f"-> division '{'small' if n_params <= 2e9 else 'LARGE (OVER CAP!)'}'", flush=True)

    t0 = time.perf_counter()
    turns_done = 0

    for bstart in range(0, len(todo), args.batch):
        chunk = todo[bstart : bstart + args.batch]
        answers = {r["video_path"]: [] for r in chunk}
        max_turns = max(len(r["questions"]) for r in chunk)

        for turn in range(max_turns):
            active = [r for r in chunk if turn < len(r["questions"])]
            if not active:
                break
            batch_msgs, batch_imgs = [], []
            for r in active:
                frames = pick_frames(
                    frames_root / r["video_path"][:-4], r["video_intervals"],
                    turn, args.recent, args.history,
                )
                batch_msgs.append(build_chat(r, turn, answers[r["video_path"]], frames,
                                             STYLES[args.style]))
                batch_imgs.append([Image.open(f).convert("RGB") for f in frames])

            inputs = processor.apply_chat_template(
                batch_msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                )
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            texts = processor.batch_decode(trimmed, skip_special_tokens=True)

            for r, txt in zip(active, texts):
                answers[r["video_path"]].append(txt.strip())
            turns_done += len(active)

        with out_path.open("a") as fh:
            for r in chunk:
                fh.write(json.dumps({
                    "video_path": r["video_path"],
                    "answers": answers[r["video_path"]],
                }) + "\n")

        el = time.perf_counter() - t0
        vids = bstart + len(chunk)
        rate = vids / el
        print(f"  {vids}/{len(todo)} vids | {turns_done} turns | "
              f"{el/60:.1f} min | ETA {(len(todo)-vids)/rate/60:.0f} min", flush=True)

    print(f"\ndone: {turns_done} turns in {(time.perf_counter()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
