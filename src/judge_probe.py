#!/usr/bin/env python
"""Measure real throughput + cost of the ConvQA judge before committing to a provider.

The leaderboard mandates one judge model so self-reported scores are comparable:
    Llama-4-Maverick-17B-128E-Instruct-FP8   (config.py: CONVQA_JUDGE_MODEL)
...but not which endpoint serves it. Meta's own API is documented at ~10 RPM, which
is ~7-10h for a full 700-video run. This script times N calls against whichever
provider you point it at so you can pick on measured numbers, not guesses.

Usage:
    LLAMA_API_KEY=...   python judge_probe.py --provider meta
    OPENROUTER_API_KEY= python judge_probe.py --provider openrouter --concurrency 8
"""

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

PROVIDERS = {
    # (base_url, model_id, env var holding the key)
    "meta": (
        "https://api.llama.com/compat/v1/",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "LLAMA_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-4-maverick",
        "OPENROUTER_API_KEY",
    ),
    "fireworks": (
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "FIREWORKS_API_KEY",
    ),
}

# Shape-matched to the starter kit's rubric (0 / 0.5 / 1.0, temperature 0) so the
# timing and token counts reflect real judge traffic rather than a toy prompt.
JUDGE_PROMPT = """You are grading an assistant's answer to a question about an egocentric video.

Question: {q}
Reference answer: {ref}
Assistant answer: {hyp}

Score 1.0 if the assistant answer matches the reference in substance, 0.5 if
partially correct, 0.0 if wrong. Reply with only the number."""

SAMPLE = dict(
    q="What did I just put into the mixing bowl?",
    ref="Two eggs, cracked one at a time.",
    hyp="You added a couple of eggs to the bowl.",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="meta")
    ap.add_argument("--n", type=int, default=20, help="number of probe calls")
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    base_url, model, key_env = PROVIDERS[args.provider]
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"set {key_env} first")

    from openai import OpenAI  # OpenAI-compatible surface; all three expose one

    client = OpenAI(base_url=base_url, api_key=key)
    prompt = JUDGE_PROMPT.format(**SAMPLE)

    usage = []
    errors = []

    def one(_i: int) -> float:
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            usage.append((r.usage.prompt_tokens, r.usage.completion_tokens))
        except Exception as e:  # noqa: BLE001 - we want the message, not a trace
            errors.append(repr(e)[:200])
        return time.perf_counter() - t0

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        lat = list(pool.map(one, range(args.n)))
    wall = time.perf_counter() - wall0

    ok = len(usage)
    print(f"\nprovider     {args.provider}  ({model})")
    print(f"concurrency  {args.concurrency}")
    print(f"ok / total   {ok} / {args.n}")
    if errors:
        print(f"first error  {errors[0]}")
    if not ok:
        return

    print(f"latency      p50 {statistics.median(lat):.2f}s  max {max(lat):.2f}s")
    rpm = ok / wall * 60
    print(f"throughput   {rpm:.1f} req/min")

    pin = statistics.mean(p for p, _ in usage)
    pout = statistics.mean(c for _, c in usage)
    print(f"tokens/call  {pin:.0f} in, {pout:.0f} out")

    # A full val run is 700 videos x ~6 turns. Refine TURNS once the jsonl is local.
    turns = 700 * 6
    print(f"\n-- extrapolated to a full val run ({turns} judge calls) --")
    print(f"wall clock   {turns / rpm / 60:.1f} h")
    print(f"input tokens {turns * pin / 1e6:.1f} M")


if __name__ == "__main__":
    main()
