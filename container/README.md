# Container assembly

The submitted image is the official starter-kit base with four things added: the
Qwen3-VL-32B weights, the QLoRA adapter, `peft`, and one environment variable.

## Why `crane` and not `docker build`

The 32B weights made a conventional build impractical. The base image plus weights is
tens of gigabytes, and every rebuild re-pushed layers over a slow link. The final image was
assembled with [`crane`](https://github.com/google/go-containerregistry) instead, appending
layers registry-natively with no container runtime and no local pull:

```bash
R=<registry>/wearable-ai-2026/<team>

# append weights and adapter as their own layers
crane append --base "$R:base-peft" --new_layer qwen3vl32b.tar --new_tag "$R:convqa-large-raw"
crane append --base "$R:convqa-large-raw" --new_layer lora.tar --new_tag "$R:convqa-large-raw2"

# select which registered class serves every --model-type value
crane mutate "$R:convqa-large-raw2" \
  --env WAI_ACTIVE_MODEL=sl_convqa_large \
  --tag "$R:convqa-large-final"

crane digest "$R:convqa-large-final"
```

Keep each artifact in its own layer. A single oversized layer causes the registry push to
restart from zero.

## Base layers

`Containerfile.peftbase` and `Containerfile.deps` build on the official image in order:

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false \
  -f Containerfile.peftbase -t wai-base-peft:latest .
docker build --platform linux/amd64 --provenance=false --sbom=false \
  -f Containerfile.deps -t wai-base-deps:latest .
```

`--provenance=false --sbom=false` is not optional. Without them buildx produces a manifest
list with an empty architecture field and the organisers' intake rejects the image with
`image architecture is amd64 (found: )`.

Images that must run under `--read-only` also need `VOLUME /tmp`: importing torch calls
`tempfile.gettempdir()` through `dill`, which fails on a read-only filesystem.

## Registering the model classes

Append to the starter kit's `model.py`:

```python
from wai_models import WAI_REGISTRY as _WAI
MODEL_REGISTRY.update(_WAI)

# create_model() passes DEFAULT_MODEL_IDS[key] straight to from_pretrained(), so a
# bare "/models" raises `Repo id must use alphanumeric chars`. Point each key at the
# real directory instead.
_WAI_DIRS = {
    "sl_convqa_large":     "/models/qwen3vl32b",
    "sl_convqa_small":     "/models/internvl3_5-1b-hf",
    "sl_longqa_small":     "/models/internvl3_5-1b-hf",
    "sl_proactive_small":  "/models/siglip-so400m",
}
for _k in _WAI:
    DEFAULT_MODEL_IDS.setdefault(_k, _WAI_DIRS.get(_k, "/models"))
```

Also relax `run_evaluation.py`'s `--model-type` from `choices=["llama4","qwen"]` to
`choices=None`, or the new keys are rejected at argument parsing.

## Verify before submitting

Run the organisers' `validate_image.sh` against the image, and then actually run
`run_evaluation.py` on real video. A container that imports cleanly can still produce zero
predictions. That failure mode is invisible until you evaluate on real mp4s.
