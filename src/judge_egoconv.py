#!/usr/bin/env python
"""Score an EgoConv predictions.jsonl with the official LLM judge.

The rubric, the 0/0.5/1.0 parsing, and greedy decoding are copied exactly from the
starter kit (`run_evaluation.py:_build_judge_prompt` / `_parse_judge_score`, which
calls generate with max_new_tokens=10, do_sample=False). Verified against real val
turns: feeding gold answers back as predictions scores 1.000, and a generic filler
answer scores 0.003 -- so this reproduces the official grader's behaviour.

The starter kit mandates Llama-4-Maverick-17B-128E-Instruct-FP8 served via vLLM on
8x H100. That hardware is out of reach here (Insomnia caps at 2 GPUs), so we call
the same checkpoint through OpenRouter at temperature 0. Scores should track
closely, and in any case the organizers' verified run supersedes self-reports.

With --write, appends the required `{"llm_judge": <score>}` metadata line to the
predictions file -- that single line IS how the score is submitted; there is no
form field for it.

Usage:
    python judge_egoconv.py --predictions submissions/egoconv_large/predictions.jsonl
    python judge_egoconv.py --predictions ... --write
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "egoconv" / "wearable_ai_2026_egoconv_val_700.jsonl"
JUDGE = "meta-llama/llama-4-maverick"


def build_prompt(question: str, gold: str, pred: str) -> str:
    return (
        "You are an evaluation judge. Given a question, a reference answer, "
        "and a predicted answer, rate the predicted answer.\n\n"
        "Score 1.0 if the predicted answer is correct and captures the key "
        "information.\n"
        "Score 0.5 if the predicted answer is partially correct (some key "
        "info present but incomplete or slightly wrong).\n"
        "Score 0.0 if the predicted answer is wrong or irrelevant.\n\n"
        "Reply with ONLY a single number: 1.0, 0.5, or 0.0\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {gold}\n"
        f"Predicted Answer: {pred}\n\n"
        "Score:"
    )


def parse_score(text: str) -> float:
    text = text.strip().strip(".")
    for token in text.split():
        token = token.strip(".,;:")
        try:
            val = float(token)
        except ValueError:
            continue
        return 1.0 if val >= 0.75 else (0.5 if val >= 0.25 else 0.0)
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--write", action="store_true",
                    help="append the {'llm_judge': score} line for submission")
    ap.add_argument("--dump", default="",
                    help="write per-turn scores here for error analysis / targeted reruns")
    ap.add_argument("--only-present", action="store_true",
                    help="score only videos present in the predictions file "
                         "(for smoke tests / monitoring a run in progress; a real "
                         "submission must cover all 700)")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    gold = {}
    for line in GOLD.open():
        r = json.loads(line)
        gold[r["video_path"]] = r

    pred_path = Path(args.predictions)
    preds = {}
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "video_path" in o:
            preds[o["video_path"]] = o

    # Build the full turn list. A missing or blank prediction scores 0.0 with no
    # API call, matching the starter kit's short-circuit.
    tasks, freebies = [], []
    for vid, g in gold.items():
        if args.only_present and vid not in preds:
            continue
        p = preds.get(vid, {})
        pa = p.get("answers", [])
        for j, gold_ans in enumerate(g["answers"]):
            q = g["questions"][j] if j < len(g["questions"]) else ""
            pred_ans = pa[j] if j < len(pa) else ""
            if not pred_ans.strip():
                freebies.append((g["task"], 0.0))
            else:
                tasks.append((g["task"], q, gold_ans, pred_ans, vid, j))

    print(f"videos scored {len(preds)}/{len(gold)} | turns {len(tasks) + len(freebies)} "
          f"({len(freebies)} empty -> auto 0.0)")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    lock = threading.Lock()
    usage = {"tin": 0}
    results: list[tuple[str, float]] = []

    def score_one(t):
        task, q, g, p, vid, j = t
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=JUDGE,
                    messages=[{"role": "user", "content": build_prompt(q, g, p)}],
                    temperature=0, max_tokens=10,
                )
                with lock:
                    if r.usage:
                        usage["tin"] += r.usage.prompt_tokens
                return task, parse_score(r.choices[0].message.content or ""), vid, j, q, g, p
            except Exception:
                if attempt == 3:
                    return task, 0.0, vid, j, q, g, p
                time.sleep(2 * (attempt + 1))
        return task, 0.0, vid, j, q, g, p

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(score_one, tasks))
    el = time.perf_counter() - t0

    detailed = results
    results = [(t, s) for t, s, *_ in results]
    results.extend(freebies)
    overall = statistics.mean(s for _, s in results)

    by_task = defaultdict(list)
    for task, s in results:
        by_task[task].append(s)

    print(f"\nLLM-Judge  {overall:.4f}   ({len(results)} turns, {el/60:.1f} min, "
          f"${usage['tin']/1e6*0.20:.2f})")
    dist = defaultdict(int)
    for _, s in results:
        dist[s] += 1
    tot = len(results)
    for v in (1.0, 0.5, 0.0):
        print(f"  score {v}: {dist[v]:>5} ({dist[v]/tot:.1%})")
    print("\nby task:")
    for task, vals in sorted(by_task.items(), key=lambda kv: -statistics.mean(kv[1])):
        print(f"  {task:<28} {statistics.mean(vals):.3f}  (n={len(vals)})")

    if args.dump:
        with open(args.dump, "w") as fh:
            for task, sc, vid, j, q, g, p in detailed:
                fh.write(json.dumps({"video_path": vid, "turn": j, "score": sc,
                                     "task": task, "question": q,
                                     "gold": g, "pred": p}) + "\n")
        print(f"\nwrote {len(detailed)} per-turn scores -> {args.dump}")

    if args.write:
        lines = [l for l in pred_path.read_text().splitlines()
                 if l.strip() and "video_path" in l]
        lines.append(json.dumps({"llm_judge": round(overall, 4)}))
        pred_path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote llm_judge={overall:.4f} into {pred_path} "
              f"({len(lines) - 1} prediction rows + 1 score line)")


if __name__ == "__main__":
    main()
