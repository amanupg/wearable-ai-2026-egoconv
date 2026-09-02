#!/usr/bin/env python
"""LoRA supervised fine-tuning for EgoConv (small division).

WHY
---
Zero-shot InternVL3.5-1B scores 0.2402 on val; the small-division leader is at
0.39. On the public board, fufu's same-model pair differs only by fine-tuning --
EgoAssist-Conv-0.9B 0.26 -> EgoAssist-Conv-0.9B-SFT 0.34. That +0.08 is a training
gap, not a prompting gap, and no amount of frame/prompt search closes it (eight
levers measured; the configuration space is exhausted).

DATA
----
The only labelled data anywhere is the released val split, so that is what we
train on -- as does every competitor, since test is held out. To keep an honest
read on generalisation, the split is BY VIDEO (never by turn): all turns of a
video land on the same side, otherwise turn i of a conversation leaks the context
of turn i+1 and holdout numbers become fiction.

Each training example is one turn: the frames for that turn, the conversation so
far (using GOLD previous answers, i.e. teacher forcing), and the gold answer as
the target. Only the answer tokens carry loss.

WHAT IS TRAINED
---------------
LoRA on the language model's attention and MLP projections; the vision tower is
frozen. At 1B with rank 16 that is ~0.5% of parameters, fits comfortably on one
A6000, and leaves the base weights untouched so the released artifact is
base + adapter.

Param-count note: LoRA adds parameters. rank-16 adapters on a 1.061B model add
roughly 9M, so the merged model stays ~1.07B -- still inside the 2B cap. The
script prints the merged total so the declaration can't drift.

Usage:
    python train_egoconv_lora.py --epochs 2 --rank 16 --out runs/egoconv_lora
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def pick_frames(frame_dir: Path, intervals, turn: int, n_recent: int, n_history: int):
    """Same hierarchical selection as inference -- train/test mismatch here would
    silently cost more than the fine-tuning gains."""
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


SYSTEM = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see, in chronological order. Answer their question directly and "
    "conversationally, as a helpful assistant speaking aloud."
)


class TurnDataset(Dataset):
    """One item = one conversational turn."""

    def __init__(self, rows, frames_root: Path, n_recent: int, n_history: int):
        self.items = []
        for r in rows:
            for t in range(len(r["questions"])):
                self.items.append((r, t))
        self.frames_root = frames_root
        self.n_recent = n_recent
        self.n_history = n_history

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, turn = self.items[i]
        frames = pick_frames(self.frames_root / r["video_path"][:-4],
                             r["video_intervals"], turn, self.n_recent, self.n_history)
        convo = ""
        for k in range(turn):
            # Teacher forcing: condition on gold history, not on model output.
            convo += f"User: {r['questions'][k]}\nYou: {r['answers'][k]}\n"
        convo += f"User: {r['questions'][turn]}"
        return {"frames": [str(f) for f in frames], "prompt": convo,
                "target": r["answers"][turn]}


def build_collate(processor):
    """Tokenise exactly the way inference does.

    The first version hand-rolled `processor(text=..., images=...)` with
    truncation, which silently chopped image placeholder tokens and blew up with
    "Mismatch in image token count" (ids=4045 vs text=26624). Two lessons baked in
    here: (1) drive the processor through apply_chat_template, the same call the
    inference path uses, so train/test tokenisation cannot diverge; (2) never
    truncate a multimodal sequence -- dropping image tokens corrupts alignment
    rather than just shortening context.

    Prompt tokens are masked by tokenising the same conversation WITHOUT the
    assistant turn and masking up to that length, so loss lands only on the answer.
    """
    def collate(batch):
        full_msgs, prompt_msgs = [], []
        for b in batch:
            content = [{"type": "image", "image": f} for f in b["frames"]]
            content.append({"type": "text", "text": b["prompt"]})
            base = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content},
            ]
            prompt_msgs.append(base)
            full_msgs.append(base + [{"role": "assistant",
                                      "content": [{"type": "text", "text": b["target"]}]}])

        enc = processor.apply_chat_template(
            full_msgs, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True)

        # Length of the prompt half, per item, to know where the answer starts.
        prompt_lens = []
        for m in prompt_msgs:
            pe = processor.apply_chat_template(
                [m], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            prompt_lens.append(pe["input_ids"].shape[1])

        labels = enc["input_ids"].clone()
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :min(plen, labels.shape[1])] = -100
        enc["labels"] = labels
        return enc
    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--recent", type=int, default=6)
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=140, help="videos held out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qlora", action="store_true",
                    help="load base in 4-bit (nf4). Required for 32B: bf16 is 64GB "
                         "and leaves no room for activations on an 80GB card.")
    ap.add_argument("--pissa", action="store_true",
                    help="initialise adapters from principal singular vectors "
                         "instead of randomly. Chosen because our binding "
                         "constraint is that only ~1 epoch fits, so convergence "
                         "speed matters more than asymptotic quality.")
    ap.add_argument("--save-every", type=int, default=400,
                    help="checkpoint every N steps so a wall-clock kill costs "
                         "minutes of work, not the whole run")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = [json.loads(l) for l in open(args.gold) if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    hold, train = rows[: args.holdout], rows[args.holdout :]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "holdout_ids.json", "w") as fh:
        json.dump([r["video_path"] for r in hold], fh)
    print(f"train videos {len(train)} | holdout videos {len(hold)} (split by video)",
          flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    load_kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if args.qlora:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
    base_params = sum(p.numel() for p in model.parameters())

    # Freeze the vision tower: with 4.4k examples there is not enough signal to
    # retrain perception, and it is the bulk of the memory cost.
    for name, p in model.named_parameters():
        if "vision" in name.lower():
            p.requires_grad = False

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", use_rslora=True,
        init_lora_weights=("pissa_niter_16" if args.pissa else True),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)

    # Activation memory, not weights, is what OOMs here: a 4-frame 448p turn is a
    # ~13k-token sequence and backprop keeps every intermediate. Checkpointing
    # recomputes them instead, trading ~30% step time for a large memory cut.
    # use_reentrant=False is required for it to cooperate with frozen params.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    merged = base_params + trainable
    print(f"base {base_params:,} | LoRA trainable {trainable:,} "
          f"| merged ~{merged:,} ({merged/1e9:.3f}B) -> "
          f"{'small OK' if merged <= 2e9 else 'OVER 2B CAP'}", flush=True)

    ds = TurnDataset(train, Path(args.frames), args.recent, args.history)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=build_collate(processor),
                    pin_memory=True, drop_last=True)
    print(f"train turns {len(ds)} | steps/epoch {len(dl)//args.accum}", flush=True)

    steps = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, steps // args.accum),
        pct_start=0.05)

    model.train()
    t0 = time.perf_counter()
    seen, running = 0, []
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels")

            # Only the answer tokens carry loss (~40 of a ~26k-token sequence), but
            # letting the model compute loss internally materialises logits for
            # EVERY position: 26k x 152k vocab in fp32 is a 15 GB allocation and an
            # instant OOM. Ask for logits over just the tail and score those.
            keep = int((labels != -100).sum(dim=1).max().item()) + 1
            keep = max(keep, 2)
            out = model(**batch, logits_to_keep=keep)
            logits = out.logits.float()

            # logits[:, j] predicts input_ids[:, -keep + j + 1]
            tgt = labels[:, -keep + 1:]
            pred = logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                pred.reshape(-1, pred.size(-1)), tgt.reshape(-1),
                ignore_index=-100)

            (loss / args.accum).backward()
            running.append(loss.item())
            seen += 1
            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if seen % 50 == 0:
                el = (time.perf_counter() - t0) / 60
                print(f"  step {seen}/{steps} | loss {sum(running[-50:])/len(running[-50:]):.4f} "
                      f"| {el:.1f} min | ETA {(steps-seen)/(seen/el):.0f} min", flush=True)
            if args.save_every and seen % args.save_every == 0:
                model.save_pretrained(args.out)
                print(f"  [checkpoint at step {seen}]", flush=True)
            if seen >= steps:
                done = True
                break

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"\nsaved adapter -> {args.out} ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
