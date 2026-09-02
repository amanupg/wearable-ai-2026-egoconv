# EgoConv — ECCV 2026 Wearable-AI Grand Challenge (2nd place, Large division)

Runner-up entry for the **EgoConv Large (2B+)** sub-track of the
[Wearable-AI Grand Challenge](https://wearable-ai-workshop.github.io/) at ECCV 2026,
hosted by Meta.

**Team SoloLevelling** (solo entry) · Aman Upganlawar

| | |
|---|---|
| Sub-track | EgoConv, Large division (2B+) |
| Result | **2nd place — runner-up** |
| Model | Qwen3-VL-32B + QLoRA (r=32) supervised fine-tune |
| Total parameters | 33.36 B |
| LLM-Judge score (official, test split) | **0.4816** — 2nd of 7 |
| BLEU (official, test split) | **0.1256** — highest in the division |
| Self-measured judge score (our held-out split) | 0.5649 |
| Registration ID | `WAI-3292A8CA` |
| Submitted image | `sha256:6d22ab8fe301abe4abbd44f06495d4c23ecb8937d489ea7f99ff86934e06b6e003` |

## Task

EgoConv is streaming multi-turn conversational QA over egocentric video. For each turn the
model sees the video up to the current timestamp plus the dialogue so far, and generates a
free-form answer. Answers are scored by an LLM judge
(`Llama-4-Maverick-17B-128E-Instruct-FP8`, temperature 0) against the challenge rubric.
The no-future-leaking constraint means a turn may only condition on past and current video.

## Method

A QLoRA supervised fine-tune of Qwen3-VL-32B on the challenge training conversations.

```
base            Qwen3-VL-32B
quantisation    4-bit (QLoRA)
lora_r          32
lora_alpha      64
lora_dropout    0.05
target_modules  q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
epochs          1
```

One epoch, not two. A second epoch was trained and scored **0.4749** against this run's
**0.5649** — it overfit the conversational style of the training split. The interrupted
2-epoch adapter is published alongside the winner so the comparison is reproducible.

### What actually moved the score

Measured, not assumed. Full detail in [FINDINGS.md](FINDINGS.md).

| Lever | Measurement |
|---|---|
| Frame budget at serving time | 0.5745 (10 recent + 6 history) vs 0.4048 (6 + 2) — **+0.170** |
| QLoRA SFT vs best zero-shot prompting variant | 0.5649 vs 0.4802 — **+0.085** |
| 1 epoch vs 2 epochs | 0.5649 vs 0.4749 — **+0.090** |
| Input resolution | dominates model choice; see FINDINGS.md |

Scores in the first row come from a separate A/B on a shared eval subset and are not
directly comparable to the headline 0.5649, which is the full held-out split.

The frame-budget result is the one worth stealing: the served configuration mattered far
more than anything about the adapter. `WAI_CONV_RECENT=10` and `WAI_CONV_HISTORY=6` are
environment-overridable in [`src/wai_models.py`](src/wai_models.py) for exactly this reason.

## Layout

```
src/          model classes served in the container, training and eval code
container/    how the submitted image was assembled (crane layer surgery)
adapter/      adapter config; weights are on the Hugging Face Hub
jobs/         Slurm batch scripts for every training and eval run
logs/         raw training logs for the EgoConv runs
predictions/  the winning prediction file, as submitted
FINDINGS.md   full engineering write-up, including the negative results
```

## Weights

The trained adapter (1.07 GB) is too large for this repository and lives on the Hub:

```
amanupg/wearable-ai-2026-egoconv-qwen3vl32b-qlora-r32
```

## Reproducing

The challenge starter kit is Meta's and is licensed CC BY-NC 4.0, so it is **not** vendored
here. Obtain it from the organisers, then:

1. Copy [`src/wai_models.py`](src/wai_models.py) next to the kit's `model.py`.
2. Append the registration block from [`container/README.md`](container/README.md) to `model.py`.
3. Relax `run_evaluation.py`'s `--model-type` argument from `choices=["llama4","qwen"]` to `choices=None`.
4. Fetch the base model and adapter into `/models/qwen3vl32b` and the adapter path.
5. Run with `WAI_ACTIVE_MODEL=sl_convqa_large`.

Container assembly is documented in [`container/README.md`](container/README.md).

## Citation

```bibtex
@misc{upganlawar2026egoconv,
  title  = {EgoConv: A QLoRA Fine-Tune of Qwen3-VL-32B for Streaming Egocentric
            Conversational QA},
  author = {Upganlawar, Aman},
  year   = {2026},
  note   = {2nd place, EgoConv Large sub-track, Wearable AI Grand Challenge,
            Wearable AI Workshop at ECCV 2026}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Released under an OSI-approved permissive license as
required by the challenge prize terms. The challenge data and starter kit remain under
Meta's own licenses and are not redistributed here.
