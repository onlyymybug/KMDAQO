# Model Checkpoints

This directory is intentionally empty in the public artifact.

KMDAQO can run in mock mode without model checkpoints. For real LLM-assisted
experiments, place the fine-tuned base model and LoRA adapter here, or point to
external paths with either `configs/local.yaml` or environment variables:

```bash
export KMDAQO_LLM_PATH=/path/to/base-model
export KMDAQO_LLM_ADAPTER_PATH=/path/to/lora-adapter
```

The paper experiments used a local 8B SFT model plus a LoRA adapter. These
weights are not committed because they are large and may be distributed under
separate model licenses.
