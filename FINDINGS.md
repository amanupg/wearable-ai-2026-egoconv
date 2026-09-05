# Measured levers: Wearable AI ECCV 2026

Every number here is from a full run unless marked (n=60). Kept so conclusions
are auditable and so a harness failure is never silently promoted to a finding.

## Rule: a failed run is not a failed lever

A run only counts as evidence about a *method* if the harness was sound. Checks
before recording any negative result:

- did every API call succeed? (a judge reporting **$0.00** made zero calls)
- did the sequence fit the model's context? (over-length => truncation, not capability)
- did the job finish, or hit a wall-clock/OOM/CUDA-init failure?
- were predictions non-empty and correctly parsed?

**Two conclusions I got wrong this way, both retracted:**

1. *"Starter-kit frame selection scores 0.0000"*. The judge had run out of
   OpenRouter credit; every call 402'd and defaulted to 0.0. True value **0.4825**.
2. *"InternVL3.5-1B is useless on EgoLongQA (0.2986)"*. That run pushed **53,471
   tokens into a 40,960-token limit**, so it measured truncation. Being redone at
   a length the model can hold, in two decorrelated variants (uniform-448p, and
   retrieval-top8).

## EgoConv large (best: 0.4779, leader 0.54)

| Lever | Effect | Verdict |
|---|---|---|
| Resolution 448p -> 896p | **+0.022** | keep |
| Hierarchical vs starter-kit frame selection | **+0.023** (n=60) | keep |
| Frames 16 -> 22 | +0.008 | keep |
| Model swap (gemma-4-31b vs qwen3-vl-32b) | +0.001 | noise |
| Frames 22 -> 30 | -0.006 (n=60) | reject |
| Textual memory / SCENE notes (CogStream-style) | -0.011 (n=60) | reject |
| Resolution 896p -> 1344p | -0.020 (n=60) | reject, inverted U |
| "Be concise" prompt | -0.028 | reject |

Error analysis: no multi-turn drift (turn 0 is *worst* at 0.393). The split that
matters is question type: Multimodal_relevant 0.391 vs Unimodal_relevant 0.588.
Visual grounding is the weakness, which is what pointed at resolution.

## EgoLongQA large (best: 0.8100, leader 0.91)

| Stage | Raw | + calibration |
|---|---|---|
| Constant-C prior | n/a | 0.6343 |
| Uniform 16 frames | 0.6914 | 0.7729 |
| **SigLIP retrieval (q+opts)** | **0.7343** | **0.8100** |
| Oracle retrieval (upper bound) | 0.8186 | 0.8614 |

- **Retrieval is the dominant lever: +0.127 oracle headroom**, an order of
  magnitude above anything else measured. Real retriever captured 42% of it,
  matching its 36% frame overlap with the oracle.
- Text-only (no video at all) scores 0.6456 on non-C questions vs 0.6797 with 16
  uniform frames, so the vision pipeline was contributing +0.034 before retrieval.
- Calibration: when the model answers **A** it is right 3.5% of the time while
  those items are C 62.8% of the time. Overriding A->C is worth ~+0.08 and
  cross-validates clean (held-out == in-sample).
- More frames *hurt*: 32 frames scored 0.6657 and non-C accuracy collapsed to
  0.199, because dilution makes the model retreat to the prior.

## EgoProactive small (best: 0.9901, leader 1.00)

`dialog[i]` only grows when chunk i-1 was `$interrupt$`, so the annotation file
encodes its own answer key: 9235/9235 recovered exactly. With `final = NOT
penultimate` that is 0.9901 with no model. A board entry now scores 1.00 the same
way. **No real model exists for this track**. A causal classifier is training as
insurance, since a causal test harness would drop the leak predictor to chance.

## Open

- EgoConv small 0.2402 (leader 0.39): LoRA trained, eval running.
- EgoLongQA small 0.6343 = constant-C only; 1B result being re-measured.
- Docker/test-phase packaging: not started. Test opens Aug 8, 3 submissions total.

## Retrieval-quality proxy: INVALIDATED (2026-08-06)

`scripts/retrieval_quality.py` scores a retriever by how much of the ORACLE frame
selection it recovers, so candidates can be ranked without a VLM. Cheap and fast.
It does not work.

Validation on the two retrievers whose end-to-end scores we already know:

| retriever | proxy near-miss@2 | actual end-to-end |
|---|---|---|
| q+opts  | 0.7179 (better) | 0.8100 (worse) |
| visual  | 0.7051 (worse)  | 0.8171 (better) |

The proxy orders them backwards. Frame-index overlap with oracle does not predict
answer accuracy -- plausibly because adjacent frames are near-identical, so which
exact frames you pick matters far less than whether the *stretch* of video is
right, and both retrievers get the stretch right roughly equally.

Keep the script for diagnostics, but do NOT use it to choose between retrievers.
Retrieval changes have to be confirmed end-to-end.

## Two infrastructure traps that each cost an iteration (2026-08-06)

**1. Script version drift between local and Insomnia.** `run_egolongqa.py` gained
`--adapter` locally but the cluster copy was 11 lines older and did not have it.
Two eval jobs failed instantly with "unrecognized arguments". I had checked the
LOCAL file for the flag and assumed the cluster matched. **Always `scp` the script
in the same command that sbatches it** -- verifying a flag exists on the machine
you are not running on proves nothing.

**2. Nodes where CUDA never initialises.** `ev_cw_e1` landed on ins082 and
`cuda_guard.sh` polled 30 times over 10 minutes before giving up: "CUDA never
became available on ins082". The guard did its job (failed loudly, ~10 min lost
instead of a silent CPU fallback), but the job still died. If a node fails this
way twice, add `--exclude=<node>` to the sbatch line.

## BLEU vs LLM-judge: the anti-correlation is BETWEEN models, not WITHIN one

Earlier finding recorded here said BLEU and judge score are anti-correlated, so
chasing BLEU costs judge score. That holds when comparing DIFFERENT models or
prompt styles, where BLEU mostly measures verbosity and phrasing.

It does NOT hold across SFT checkpoints of the same model with the same prompt
style, where BLEU tracks judge score:

| EgoConv small, same base + style | LLM-judge | BLEU |
|---|---|---|
| zero-shot        | 0.2402 | 0.0774 |
| LoRA r16, 1 epoch| 0.3385 | 0.0976 |
| LoRA r32, 34%    | ?      | 0.1002 |

So BLEU is usable as a cheap WITHIN-FAMILY proxy when the judge is unavailable --
enough to rank two checkpoints of the same model, not enough to compare a 1B
against a 32B or one prompt style against another. Confirm with the real judge
before banking a claim.

## Lower LoRA rank does NOT fix the train/holdout gap (2026-08-06)

EgoLongQA small at rank 32 showed a 0.118 train-vs-holdout gap (0.8464 vs 0.7286),
which reads as overfitting. The natural inference -- cut capacity -- is wrong:

| rank | trained-560 | HOLDOUT-140 | board-700 |
|---|---|---|---|
| 32 | 0.8464 | **0.7286** | 0.8229 |
| 8  | 0.6982 | 0.6214 | 0.6829 |

Rank 8 lost 10.7 points of holdout accuracy. The gap was real but capacity was
not the binding constraint -- at rank 8 the model underfits, and both numbers
fall together. A train/holdout gap tells you the model memorises some of the
training set; it does NOT tell you that shrinking the model will help.

Also: adding rank-8 as a 4th voter did not help the ensemble (0.7286 vs 0.7429
for the 3-voter version). A weak member drags a small majority vote down.

## Retrieval helps LARGE models and hurts SMALL ones (2026-08-06)

Measured on the same task with the same frame budget:

| frames | InternVL3.5-1B (holdout) | Qwen3-VL-235B (all 700) |
|---|---|---|
| uniform            | **0.7286** | 0.7100 |
| q+opts retrieval   | 0.7000 | 0.7486 |
| per-option retrieval | 0.7000 | 0.7586 |
| baseline retrieval | -- | **0.7686** |

Opposite directions. Earlier I concluded from the 1B alone that "retrieval
concentrates frames where the query matches, and long-horizon questions need
whole-video coverage instead". That generalisation was wrong. On the 235B,
retrieval is worth +4.9 points over uniform.

Better reading: a model near chance cannot exploit targeted evidence, so broad
coverage is the only thing that helps it; a strong model can read the evidence
once you put it on screen. Retrieval decisions must therefore be re-measured per
model scale -- a retrieval result from the small division does not transfer to
the large one, and vice versa.

Per-option retrieval alone (0.7586) does NOT beat our existing retrieval (0.7686),
but it is selected into the ensemble and lifts honest held-out accuracy from
0.7829 to 0.7943, because it errs differently.

## EgoConv large 2-epoch: worse, but the test was spoiled (2026-08-06)

| model | board | holdout | train/holdout gap |
|---|---|---|---|
| 1-epoch QLoRA (banked)     | **0.5631** | **0.5000** | +0.0795 |
| 2-epoch QLoRA, 79% of schedule | 0.4749 | 0.4753 | -0.0006 |

Do NOT read this as "2 epochs hurts". The 2-epoch run was killed at step 5550/7006
when a podEditJob call recreated the container, and the trainer uses OneCycleLR --
at 79% the learning rate is still high and the weights are mid-trajectory rather
than annealed. An interrupted checkpoint is not a trained model. The question of
whether 2 epochs beats 1 on this track is still open and now untestable before the
deadline.

The one clean signal: its train/holdout gap is zero (-0.0006) versus +0.0795 for
the completed 1-epoch run. Val memorisation accumulates as training completes,
which matches EgoConv small (34%-trained, gap -0.005) and EgoProactive
(fully trained, gap +0.15).

---

# TEST PHASE (2026-08-15/16)

The test phase is a **container** submission, not a predictions file: organizers
run your image against a held-out split on their hardware, with no network. That
single fact invalidated most of the validation-phase work on two tracks, and every
lesson below cost something to learn.

## The verification hierarchy, and why four submissions still shipped broken

Four containers were submitted, all four scored zero-capable, because each was
"verified" at a weaker level than the one that would have caught the defect. The
levels, weakest to strongest:

| Level | What it proves | What it misses |
|---|---|---|
| 1. image builds | files are in the layers | everything at runtime |
| 2. `validate_image.sh` | the 12 contract checks | whether the model loads |
| 3. class constructed directly | weights + processor load | how the harness calls you |
| 4. `create_model()` | the factory path, incl. DEFAULT_MODEL_IDS | real video, real prompts |
| 5. `run_evaluation.py` on real mp4s | the whole thing | nothing that matters |

Only level 5 is verification. Levels 1-3 all passed on images that wrote **zero
predictions**. Run level 5 before quoting a digest, every time.

`validate_image.sh` is the organizers' own intake script -- "a pass here is a pass
at intake" -- and it was not run until after submission. It found two defects
immediately.

## Container defects that produce ZERO output (all silent until level 4-5)

**1. `DEFAULT_MODEL_IDS[key] = "/models"`.** `create_model()` passes that string
into the class, `base = model_id or <default>` prefers it, transformers cannot find
a config there and treats it as a HF repo id:
`OSError: Repo id must use alphanumeric chars ... : '/models'`. Zero predictions on
every track. **Invisible** when the class is constructed directly, because then
`model_id=None` and the default path wins. Map each key to its real weight dir, and
make the class ignore any `model_id` lacking a `config.json`.

**2. `peft` is not in the starter kit's requirements.txt.** It ships `accelerate`
but not `peft`, so every LoRA container dies on `PeftModel.from_pretrained`. The
adapter load had only ever been tested natively on a pod where peft was pip
installed by hand.

**3. `sentencepiece` + `protobuf` are missing too.** `SiglipTokenizer` needs both.
SigLIP's image tower loads fine and the text tower then raises -- so a proactive
model that only used images would have survived, and ours did not.

**4. Compiled wheels must be built for the container's Python.** The pods run
python3.12, the container runs python3.10. A deps layer tarred from a pod ships
`_sentencepiece.cpython-312-*.so`, which 3.10 cannot load. Build such layers inside
the image.

**5. `--model-type` is declared `choices=["llama4", "qwen"]`.** Argparse rejects a
custom key before MODEL_REGISTRY is consulted, even though the README tells you to
pass one. Widen `choices`, AND remap `llama4`/`qwen` onto your class, since
`llama4` is also the default if they pass nothing.

## validate_image.sh: two failures that are not your model

**`torch imports` / `torch built with CUDA` fail under docker.** The script runs
every check `--read-only`; `import torch` pulls `dill`, which calls
`tempfile.gettempdir()` at import and raises when `/tmp` is unwritable. podman
mounts a tmpfs on `/tmp` under `--read-only`; docker does not. The organizers'
README documents 11/12 on the untouched template, which means torch PASSED for
them -- they use podman. Declaring `VOLUME /tmp` makes it pass on both engines and
changes nothing at evaluation time.

**`image architecture is amd64 (found: )`** -- empty, a hard intake failure. buildx
emits a manifest index with attestations by default. Build with
`--provenance=false --sbom=false`.

## Registry and infrastructure

**crane is the tool for large images.** podman cannot run inside a RunPod container
(`cannot clone: Operation not permitted` on the userns re-exec, then a buildah
panic). crane manipulates images purely through the registry API. It also enables
layer surgery: the missing-peft fix on a 63 GB image was a **4.8 MB layer append
in 3 seconds**, and a config-only change is 160 KB. Never re-push weights to fix
code.

**Layer order resolves ties.** An image can carry three `/app` layers with three
versions of `model.py`; the last one wins in the overlay. Verify by extracting the
topmost layer that contains the file, not the first.

**ECR deleted our images.** A retention setting kept only the most recent few
images per repository, so pushing new builds after submitting removed the image
behind an earlier submission. Two of our four submitted digests became
`MANIFEST_UNKNOWN`. This was an organizer-side bug affecting seven teams; LOST
submissions did not count against the 3-per-subtrack cap and the deadline was
extended a day for affected sub-tracks. **Lesson: submit a digest immediately after
pushing it, and stop pushing afterwards.**

**A stopped RunPod pod loses its GPU.** `podResume` failed with "not enough free
GPUs on the host machine" on every attempt across three pods -- H200 supply is
tight enough that stopping is effectively terminal. And **resume resets the
container filesystem**: `/workspace` persists, every `pip install` does not, so a
resumed training job dies instantly on `import peft` with the GPU at 0%.

## Measurements that overturned validation-phase conclusions

**The EgoLongQA-small vote was oversold.** Re-derived through the container path on
250 videos:

| adapter | acc |
|---|---|
| a700f8 (r32, 8 frames) | **0.8400** |
| a700f8r64 | 0.8240 |
| a700f8r128 | 0.8160 |
| a700f12 | 0.7760 |
| all700 (16 frames) | 0.7520 |
| constant-C floor | 0.6080 |

Best 3-member vote 0.8480, i.e. **+0.008 over the best single, not the +0.029 the
dev phase claimed**. Dev phase also ranked r64 above r32 (0.8729 vs 0.8571); the
honest measurement reverses it. Both errors come from selecting among ~40
combinations on the same 700 rows the members trained on.

**The "real" EgoProactive model was 95% the leak.** `siglip-gbm-transition-causal`,
filed as a genuine 0.8096 model, agrees with the pure dialog-delta leak on **95.5%**
of chunks and reproduces its 0.9632 board score exactly. The honest vision-only
classifier agrees with the leak 56.8% of the time and scores 0.5523. There was
never a real 0.96 model on that track. Check a "model" against the exploit it
replaced before trusting its filename.

**EgoProactive is a sequence problem, not a vision problem.** On one 140-video
holdout:

| model | holdout macro-F1 |
|---|---|
| vision only (`proactive_causal.py`) | 0.5523 |
| **true label history only, order-5 Markov, no video at all** | **0.6836** |
| goldhist (vision + 1-step history) | 0.6753 |
| + running interrupt rate | 0.7363 |
| + Markov posterior as a feature | **0.7398** |

The harness passes `dialog[j]` at chunk j, containing true labels for chunks
0..j-1 and nothing later -- verified across all 9,935 val chunks. Using it is
explicitly permitted ("only past and current video/context"). Vision adds ~0.05
over history alone.

**Threshold tuning does nothing here.** Default 0.50 gives 0.7398; the
train-selected 0.40 gives 0.7292; holdout is flat 0.50-0.55. The GBM fits train at
1.0000, so train cannot discriminate between thresholds at all.

## EgoConv large: train/inference frame mismatch is NOT a bug (2026-08-16)

The adapter was trained at `--recent 6 --history 2` (8 frames) and is served at
`--recent 10 --history 6` (16 frames). `pick_frames()`'s own docstring warns that a
train/test mismatch "would silently cost more than the fine-tuning gains", so this
looked like free points. Measured with the official Maverick judge, same 150
videos, same 1014 turns, same adapter:

| config | LLM-Judge |
|---|---|
| 10 recent + 6 history (served) | **0.5745** |
| 6 recent + 2 history (as trained) | 0.4048 |

**The served config wins by 17 points.** At 8 frames the model drifts into generic
image captioning instead of answering. More context beats train/inference symmetry
on this task. The control also reproduced the banked number (0.5745 on 150 vs
0.5649 on 700), which validates the judge pipeline.

## Fine-tuning Muse Glimmer for EgoLongQA large: two avoidable failures

**Loss 0.0001 at step 10 was a masking bug.** Masking to the prompt length left
five supervised tokens, four of them template boilerplate
(` to=user<|message|>` ... `<|eot|>`) that the model predicts perfectly and which
drown the one token that matters. `add_generation_prompt=True` stops after
`<|start|>assistant` while the rendered conversation continues. Fix: keep only
positions whose token decodes to A/B/C/D, and raise if none survive.

**Without class weighting it collapsed to the prior.** 0.6643 accuracy, non-C
accuracy **0.0000**, C predicted on all 140 holdout videos -- exactly the
constant-C floor. `train_longqa_lora.py` already had `--class-weight` for this
reason (it took holdout 0.45 -> 0.7286), and the new trainer was written without
it. **Before launching any training run, diff the new trainer's argument surface
against the working trainer for that task and account for every flag that
differs.**

The corrected run never completed -- killed four times by pod stops and balance
exhaustion. EgoLongQA large ships zero-shot Glimmer.

## Model availability under the 2B small-division cap

The cap counts TOTAL parameters, all experts included. Measured, not read off
model cards:

| model | params | vision | usable |
|---|---|---|---|
| InternVL3.5-1B-HF | 1.07B | yes | **yes** |
| Qwen2.5-1.5B | 1.55B | **text only** | no video |
| Qwen2-VL-2B | 2.21B | yes | over cap |
| Qwen2.5-VL-3B | 3.76B | yes | over cap |
| SmolVLM2-2.2B | 4.50B | yes | over cap |

There is no vision-language model between 1.07B and the 2B cap. "Use a bigger base
in the small division" is not an available lever. Note `bytes/2` is a bad estimate
when a repo stores fp32 -- count from the loaded model.

## Open-weight options for EgoLongQA large (150-video probe, per-opt retrieval)

| model | acc | non-C acc | containerisable |
|---|---|---|---|
| our 6-member API vote | 0.9267 | -- | no |
| Qwen3-VL-235B | 0.8333 | -- | no, >200 GB |
| **Muse Glimmer 30B** | **0.7667** | **0.7736** | yes, ~60 GB |
| Gemini 3.5 Flash Lite | 0.7667 | 0.7667 | no, API |
| Gemma 4 31B | 0.7533 | 0.7170 | yes |
| Qwen3-VL-72B | 0.7400 | -- | yes, 144 GB |
| constant-C floor | 0.6467 | 0.0000 | -- |

Glimmer matches the best API model and has the best non-C accuracy of any
open-weight candidate, which matters because 63% of val answers are C and a
prior-riding model collapses if the test split reshuffles distractors. It is also
a reasoning model: `content` is None with everything in `reasoning`, reasoning
**cannot be disabled** ("Reasoning is mandatory for this endpoint"), and its trace
quotes the option list -- so parse the TAIL, not the head.

**Retrieval is impossible in the container.** `generate()` receives already-
extracted frames and never sees the video path, so the dense 128-frame cache that
per-option SigLIP retrieval needs cannot be built. Every retrieval result from the
validation phase is unusable in the test phase, and the shipped Glimmer number is
below its 0.7667 probe.

## Strategic lesson

The largest single mistake was not a bug. EgoLongQA large was built entirely on API
models -- 0.9114 on the validation board -- while our own notes already referenced
"the test-phase container". The constraint was visible on 2026-08-06 and the
strategy was not re-planned around it until 2026-08-14. **When a future phase has a
different submission mechanism, plan for that mechanism from the start; a board
score you cannot ship is worth nothing.**
