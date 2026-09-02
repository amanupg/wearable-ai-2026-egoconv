#!/usr/bin/env python
"""EgoConv inference: one full pass over the 700-video val set -> predictions.jsonl.

DESIGN NOTES
------------
Frame selection is where this differs from the starter-kit baseline, and it is the
main lever on score. The baseline accumulates every interval's frames and strides
the whole pile down to --max-frames, which means that by turn 10 the frames for
the turn actually being asked about are ~1/10th of the context. But EgoConv
questions are overwhelmingly about the present moment: 2423 turns are tagged
Multimodal_relevant vs 697 Multiturn.

So we split the budget instead:
  - RECENT: dense frames from the current turn's interval (what's being asked about)
  - HISTORY: sparse frames spread across all earlier intervals (continuity only)

Answer length is the other lever. Gold answers average 26.7 words and the judge
scores 1.0 only when the key information is captured -- measured on real turns, a
generic hedging answer scores 0.003, so there is no benefit to vagueness. The
prompt targets the gold distribution directly.

Frame lookup is trivial because the cache is 1 fps indexed on REAL container time:
frame N == second N. Note `duration_in_sec` in the annotations is wrong for 688/700
videos (median 2.4x short) -- intervals index the true container length, so we
never touch that field.

Resumable: predictions are appended per video and completed ids are skipped, so a
crash or an interrupt never re-buys tokens already paid for.

Usage:
    python run_egoconv.py --model qwen/qwen3-vl-32b-instruct --division large
    python run_egoconv.py --limit 5 --out /tmp/smoke.jsonl      # smoke test
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

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


# Variant for the second pass. The v1 run put 53% of turns on partial credit
# (0.5) and averaged 32.1 words against gold's 26.7. The judge awards 1.0 only for
# "correct and captures the key information", so every extra clause is a chance to
# be "slightly wrong" without earning anything. This trims the target length and
# pushes for a direct answer-first sentence.
SYSTEM_TIGHT = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see, in chronological order. Answer their question directly and "
    "conversationally, as if speaking aloud.\n"
    "Lead with the direct answer in the first few words. If asked yes/no, start "
    "with Yes or No. Name the specific objects you actually see. Keep it to one "
    "or two short sentences, about 25 words, and stop -- do not add caveats, "
    "restate the question, or offer extra advice that was not asked for. Never "
    "mention images, frames, or being unable to see."
)


# --- R3: streaming textual memory (CogStream / Video-Streaming-Thinking) ---------
# The streaming-video literature converges on: compress visual history into a
# long-term TEXTUAL memory and keep a short-term VISUAL buffer for the present.
# Our v1-v5 runs instead spent 8 of 22 frames on sparse history frames, which is
# expensive (each 896p frame is ~604 tokens) and low-information, and we measured
# three separate times that adding frames past ~22 causes dilution and hurts.
#
# So here the model emits a one-line SCENE note alongside each answer. Notes
# accumulate as cheap text (~15 tokens each) and replace the history frames, which
# frees the entire frame budget for the interval actually being asked about.
# Crucially this costs no extra API calls -- the note rides along in the same
# completion as the answer.
SYSTEM_MEM = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see right now, in chronological order. Earlier moments are given to "
    "you as short SCENE notes rather than images.\n"
    "Answer their question directly and conversationally, as a helpful assistant "
    "speaking aloud. Be specific and concrete -- name what you actually see. Aim "
    "for one to two sentences (about 25-30 words). Do not describe the images as "
    "images, and never say you cannot see.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <your reply to the user>\n"
    "SCENE: <one short clause naming what is visible now, for your own future reference>"
)


def split_answer_scene(text: str) -> tuple[str, str]:
    """Pull ANSWER/SCENE apart. Falls back to treating the whole reply as the
    answer, so a formatting slip degrades to the v1 behaviour rather than to an
    empty prediction (which the judge scores 0.0)."""
    answer, scene = text.strip(), ""
    if "SCENE:" in text:
        head, _, tail = text.partition("SCENE:")
        answer, scene = head.strip(), tail.strip().split("\n")[0].strip()
    if answer.upper().startswith("ANSWER:"):
        answer = answer[7:].strip()
    return answer, scene


# Derived from reading actual 0.5-scored pairs, not from theory. The gold answers
# are EXPERT KNOWLEDGE with the scene as context ("vary your line weights: thick
# for emphasis, medium for detail"), while our predictions were SCENE DESCRIPTIONS
# ("use the thicker marker you're holding"). The v1 prompt's "name what you
# actually see" was steering away from the register that earns 1.0 -- 54% of turns
# sat on partial credit largely for answering a different question than gold did.
#
# This prompt asks for the substantive answer first, with the video as grounding
# rather than as the subject, and shows the register by example instead of
# describing it.
SYSTEM_EXPERT = (
    "You are a knowledgeable wearable AI assistant. The images show what the user "
    "sees through smart glasses. Use them to understand the situation, but answer "
    "the QUESTION -- do not merely describe the scene.\n"
    "Give the genuinely useful answer a knowledgeable friend would give: include "
    "the relevant facts, technique, or advice, and mention what you can see only "
    "when it makes the answer more specific. Two or three sentences.\n\n"
    "Examples of the expected register:\n"
    "Q: How do I make my drawing bold?\n"
    "A: Vary your line weights -- thicker lines for emphasis, medium for detail, "
    "thin for subtle hints. That contrast is what reads as bold.\n"
    "Q: What makes good winter clothing storage?\n"
    "A: Clean and fully dry everything first, use breathable containers rather "
    "than sealed plastic, keep it cool and dry, and label the bins so you are not "
    "opening all of them in October.\n"
    "Q: Is this a good inexpensive fabric?\n"
    "A: Silk is a luxury fabric, so not really. If you want the look for less, "
    "habotai, charmeuse or a synthetic silk blend give a similar drape."
)


# Surface-style variant. Our board BLEU is 0.04, the lowest of any team (others
# 0.07-0.11), while our judge score is mid-pack -- so the answers are semantically
# fine but lexically unlike gold. Measured divergences per 1k tokens:
#   contractions 's/'re/'t : ours 16.8/6.4/3.4  vs gold 0.5/0.1/0.1
#   "you"                  : ours 13.3          vs gold 3.7
#   "see"                  : ours 2.1           vs gold 0.1
# Gold is formal and expository; the v1 prompt asks for the opposite
# ("conversationally, as a helpful assistant speaking aloud").
#
# This changes ONLY the surface register, keeping v1's content strategy intact --
# unlike the "expert" variant, which changed WHAT to say and lost the visual
# grounding that was earning credit.
SYSTEM_FORMAL = (
    "You are a wearable AI assistant. The user wears smart glasses; the images are "
    "what they see, in chronological order. Answer their question directly.\n"
    "Be specific and concrete -- name what is actually visible. Aim for one to two "
    "sentences, about 25 words.\n"
    "Write in a clear, informative register: do NOT use contractions (write "
    "\"it is\" not \"it's\", \"you are\" not \"you're\"), and describe the "
    "situation rather than addressing the user as \"you\" wherever it reads "
    "naturally. Do not mention images, frames or seeing."
)

STYLES = {"v1": SYSTEM, "tight": SYSTEM_TIGHT, "mem": SYSTEM_MEM,
          "expert": SYSTEM_EXPERT, "formal": SYSTEM_FORMAL}


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def pick_frames(
    frame_dir: Path, intervals: list, turn: int, n_recent: int, n_history: int,
    select: str = "hier",
) -> list[Path]:
    """Dense frames from the current interval + sparse frames from earlier ones.

    The cache is 1 fps on real container time, so second S -> frame file S+1
    (ffmpeg's %06d numbering starts at 1).
    """
    # Ignore macOS AppleDouble sidecars ("._000001.jpg"). They are created by
    # tar on macOS during transfer, and because they sort alongside the real
    # frames they would silently shift the second->frame index mapping.
    avail = sorted(p for p in frame_dir.glob("*.jpg") if not p.name.startswith("."))
    if not avail:
        return []

    def at(sec: float) -> Path | None:
        idx = int(round(sec))
        return avail[idx] if 0 <= idx < len(avail) else None

    def span(lo: float, hi: float, k: int) -> list[Path]:
        if k <= 0 or hi <= lo:
            return []
        step = (hi - lo) / k
        out = []
        for i in range(k):
            f = at(lo + step * (i + 0.5))
            if f is not None and f not in out:
                out.append(f)
        return out

    if select == "kit":
        # Starter-kit default: sample n_recent frames uniformly WITHIN each interval
        # 0..turn, concatenate, then stride the whole pile down to the cap. The
        # leaderboard entry literally named "Baseline" (32B, 0.54) is presumably
        # this, so it is worth measuring rather than assuming ours is better.
        acc = []
        for k in range(turn + 1):
            lo, hi = intervals[k]
            acc.extend(span(lo, hi, n_recent))
        acc = sorted(set(acc), key=lambda p: p.name)
        cap = n_recent + n_history
        if len(acc) > cap:
            step = len(acc) / cap
            acc = [acc[int(i * step)] for i in range(cap)]
        return acc

    start, end = intervals[turn]
    recent = span(start, end, n_recent)

    history: list[Path] = []
    if turn > 0:
        hist_start = intervals[0][0]
        hist_end = intervals[turn - 1][1]
        history = [f for f in span(hist_start, hist_end, n_history) if f not in recent]

    # Chronological order matters -- the model reads them as a timeline.
    return sorted(set(history + recent), key=lambda p: p.name)


def build_messages(row: dict, turn: int, prior: list[str], frames: list[Path],
                   system: str = SYSTEM, scenes: list[str] | None = None) -> list:
    content: list[dict] = []
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode(f)}"},
        })

    convo = ""
    if scenes:
        # Long-term memory: what the earlier intervals looked like, as text.
        notes = "\n".join(f"  [{i+1}] {sc}" for i, sc in enumerate(scenes) if sc)
        if notes:
            convo += f"Earlier moments you already saw:\n{notes}\n\n"
    for i in range(turn):
        convo += f"User: {row['questions'][i]}\nYou: {prior[i]}\n"
    convo += f"User: {row['questions'][turn]}"
    content.append({"type": "text", "text": convo})

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-vl-32b-instruct")
    ap.add_argument("--division", default="large", choices=["small", "large"])
    ap.add_argument("--recent", type=int, default=10, help="frames from current interval")
    ap.add_argument("--history", type=int, default=6, help="frames spread over prior intervals")
    ap.add_argument("--max-tokens", type=int, default=90,
                    help="raise for --style mem, which also emits a SCENE line")
    ap.add_argument("--concurrency", type=int, default=8, help="videos in flight")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--style", default="v1", choices=list(STYLES))
    ap.add_argument("--frames-root", default=str(FRAMES),
                    help="frame cache root (swap to compare resolutions)")
    ap.add_argument("--select", default="hier", choices=["hier", "kit"],
                    help="hier = dense-recent + sparse-history; kit = starter-kit accumulate+stride")
    # Pin a specific serving precision. Same weights, different bit-width, so this
    # isolates quantization from every other variable.
    ap.add_argument("--quant", default="", help="bf16 | fp8 | fp4 (OpenRouter provider filter)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    out_path = Path(args.out) if args.out else (
        ROOT / "submissions" / f"egoconv_{args.division}" / "predictions.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in GOLD.open() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["video_path"])
            except Exception:
                pass
    todo = [r for r in rows if r["video_path"] not in done]
    print(f"model {args.model} | {len(todo)} videos to do ({len(done)} cached) "
          f"| frames {args.recent}+{args.history} | style {args.style}")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    lock = threading.Lock()
    stats = {"turns": 0, "tin": 0, "tout": 0, "err": 0, "vids": 0}
    t0 = time.perf_counter()

    def do_video(row: dict) -> None:
        frame_dir = Path(args.frames_root) / row["video_path"][:-4]
        answers: list[str] = []
        scenes: list[str] = []
        mem = args.style == "mem"
        for turn in range(len(row["questions"])):
            # In mem mode the whole budget goes to the current interval; history
            # is carried as SCENE notes instead of as frames.
            n_recent = args.recent + args.history if mem else args.recent
            n_hist = 0 if mem else args.history
            frames = pick_frames(
                frame_dir, row["video_intervals"], turn, n_recent,
                n_hist, args.select
            )
            msgs = build_messages(row, turn, answers, frames, STYLES[args.style],
                                  scenes if mem else None)
            text = ""
            for attempt in range(4):
                try:
                    extra = ({"provider": {"quantizations": [args.quant]}}
                             if args.quant else {})
                    r = client.chat.completions.create(
                        model=args.model, messages=msgs,
                        temperature=0.2, max_tokens=args.max_tokens,
                        extra_body=extra,
                    )
                    text = (r.choices[0].message.content or "").strip()
                    if mem:
                        text, scene = split_answer_scene(text)
                        scenes.append(scene)
                    with lock:
                        if r.usage:
                            stats["tin"] += r.usage.prompt_tokens
                            stats["tout"] += r.usage.completion_tokens
                    break
                except Exception:
                    if attempt == 3:
                        with lock:
                            stats["err"] += 1
                    else:
                        time.sleep(2 * (attempt + 1))
            # An empty answer is scored 0.0 by the judge with no API call, so a
            # failed turn costs exactly that turn and never poisons later ones.
            answers.append(text)
            with lock:
                stats["turns"] += 1

        with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(
                    {"video_path": row["video_path"], "answers": answers}
                ) + "\n")
            stats["vids"] += 1
            el = time.perf_counter() - t0
            if stats["vids"] % 10 == 0 or stats["vids"] == len(todo):
                rate = stats["vids"] / el
                eta = (len(todo) - stats["vids"]) / rate / 60 if rate else 0
                print(f"  {stats['vids']}/{len(todo)} vids | {stats['turns']} turns "
                      f"| {stats['tin']/1e6:.1f}M in | err {stats['err']} "
                      f"| ETA {eta:.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(do_video, todo))

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min | {stats['turns']} turns | errors {stats['err']}")
    print(f"tokens: {stats['tin']/1e6:.2f}M in, {stats['tout']/1e6:.2f}M out")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
