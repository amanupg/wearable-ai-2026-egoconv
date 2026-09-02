# Winning adapter

QLoRA adapter for `Qwen3-VL-32B`, trained on the EgoConv training conversations.
1st place, EgoConv Large, ECCV 2026 Wearable-AI Grand Challenge.

The weights file is 1.07 GB and is hosted on the Hugging Face Hub rather than in git:

```
amanupg/wearable-ai-2026-egoconv-qwen3vl32b-qlora-r32
```

`adapter_config.json` in this directory is the exact config that produced the winning run
(896 LoRA tensors, rank 32, alpha 64).

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-32B-Instruct",
                                                   torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "amanupg/wearable-ai-2026-egoconv-qwen3vl32b-qlora-r32")
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
```

Serving defaults that matter — these were worth more than the adapter itself:

```
WAI_CONV_RECENT=10     # recent frames per turn
WAI_CONV_HISTORY=6     # history frames per turn
```
