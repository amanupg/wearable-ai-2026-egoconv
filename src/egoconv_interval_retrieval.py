#!/usr/bin/env python
"""Question-conditioned frame retrieval WITHIN each EgoConv turn interval.

WHY THIS IS THE REMAINING LEVER
-------------------------------
Everything else on EgoConv large is measured out: capacity does nothing (235B
0.5100 vs 32B 0.5088, and a 122B-A10B was far worse), prompting is closed after
five variants, resolution peaks at 896p, and frame count peaks around 24.

But retrieval was worth +0.127 of headroom on EgoLongQA, and EgoConv has never
had it applied. Currently frames are sampled UNIFORMLY inside each turn's
interval. Those intervals run 25-35s, so at 1 fps there are ~25-35 candidate
frames and we take ~20 of them blind. That is a weaker version of exactly the
problem retrieval solved on LongQA -- less severe, because the interval is
already question-aligned by the annotation, but the same shape.

The measured EgoConv error profile says this is the right target: turns tagged
Multimodal_relevant score 0.391 while Unimodal_relevant score 0.588. The gap is
visual grounding, and showing better frames is the only untried way to attack it.

Expected gain is smaller than LongQA's because the search space is ~30 frames
rather than ~600. If it is flat, that closes EgoConv structurally and the honest
conclusion is that the remaining gap is not addressable by frame selection.

Output: {video_path: {turn_index: [frame paths]}} consumed by run_egoconv.py.

Usage:
    python egoconv_interval_retrieval.py --gold ... --frames ... --out retr.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image


def as_tensor(o):
    if torch.is_tensor(o):
        return o
    for a in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
        v = getattr(o, a, None)
        if torch.is_tensor(v):
            return v.mean(1) if v.dim() == 3 else v
    raise TypeError(f"cannot extract tensor from {type(o)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--recent", type=int, default=20, help="frames from current interval")
    ap.add_argument("--history", type=int, default=4, help="frames from earlier intervals")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - refusing to run SigLIP on CPU")
    from transformers import AutoModel, AutoProcessor

    rows = [json.loads(l) for l in open(args.gold) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    print(f"siglip {sum(p.numel() for p in model.parameters())/1e9:.3f}B on "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    root = Path(args.frames)
    out: dict[str, dict[str, list[str]]] = {}
    t0 = time.perf_counter()

    with torch.inference_mode():
        for n, r in enumerate(rows, 1):
            d = root / r["video_path"][:-4]
            avail = sorted(p for p in d.glob("*.jpg") if not p.name.startswith("."))
            if not avail:
                continue

            # Embed every frame this video needs, once. Intervals overlap across
            # turns, so per-turn embedding would redo most of the work.
            ivs = r["video_intervals"]
            need = set()
            for lo, hi in ivs:
                for s in range(max(0, int(lo)), min(len(avail), int(hi) + 1)):
                    need.add(s)
            need = sorted(need)
            if not need:
                continue
            feats = []
            for i in range(0, len(need), 64):
                imgs = [Image.open(avail[j]).convert("RGB") for j in need[i:i + 64]]
                px = proc(images=imgs, return_tensors="pt").to("cuda")
                f = as_tensor(model.get_image_features(
                    **{k: (v.to(model.dtype) if v.is_floating_point() else v)
                       for k, v in px.items()}))
                feats.append(torch.nn.functional.normalize(f.float(), dim=-1))
            emb = torch.cat(feats)
            pos = {s: i for i, s in enumerate(need)}

            qs = [q[:300] for q in r["questions"]]
            tk = proc(text=qs, padding="max_length", truncation=True,
                      return_tensors="pt").to("cuda")
            qemb = torch.nn.functional.normalize(
                as_tensor(model.get_text_features(**tk)).float(), dim=-1)

            per_turn: dict[str, list[str]] = {}
            for t in range(len(qs)):
                lo, hi = ivs[t]
                cur = [s for s in range(max(0, int(lo)), min(len(avail), int(hi) + 1)) if s in pos]
                hist = [s for s in range(max(0, int(ivs[0][0])),
                                         min(len(avail), int(ivs[t - 1][1]) + 1)) if s in pos] if t > 0 else []
                sel = []
                for pool, k in ((cur, args.recent), (hist, args.history)):
                    if not pool or k <= 0:
                        continue
                    idx = torch.tensor([pos[s] for s in pool], device="cuda")
                    sims = (emb[idx] @ qemb[t])
                    take = min(k, len(pool))
                    top = sims.topk(take).indices.tolist()
                    sel.extend(pool[i] for i in top)
                # Chronological -- the model reads frames as a timeline.
                per_turn[str(t)] = [str(avail[s]) for s in sorted(set(sel))]
            out[r["video_path"]] = per_turn

            if n % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {n}/{len(rows)} | {el/60:.1f} min | "
                      f"ETA {(len(rows)-n)/(n/el)/60:.0f} min", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"retriever": args.model, "recent": args.recent,
               "history": args.history, "selected": out}, open(args.out, "w"))
    print(f"\nretrieved per-turn frames for {len(out)} videos -> {args.out}")


if __name__ == "__main__":
    main()
