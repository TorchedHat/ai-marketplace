# Better Benchmark Model Extraction

Claude Code skill for extracting representative fusion regions from Hugging
Face or other PyTorch models into better-benchmark. It guides block selection,
memory-conscious construction, compiler trace replay, and generated corpus
validation.

## Install and use

```bash
claude plugin marketplace add TorchedHat/ai-marketplace
claude plugin install better-benchmark
```

In Claude Code, invoke the skill with a model and workload:

```text
/better-benchmark:better-benchmark-model-extraction Plan a representative inference capture for the local model, preserving its production widths and input shapes.
```

Planning requests inspect the model and propose coverage. Extraction requests
capture and validate generated artifacts. Numerical comparisons, oracle work,
and benchmarking require their own request.

## Requirements

Use existing better-benchmark, PyTorch, and model-source checkouts and their
configured Python environment. Capture requires a compatible PyTorch/CUDA
environment, an available GPU with sufficient memory, and tlparse. Hugging
Face models also require transformers and access to the requested config.
The plugin provides instructions; follow the checkout's setup requirements.

## Files

```text
better-benchmark/
├── .claude-plugin/
│   └── plugin.json
├── README.md
└── skills/
    └── better-benchmark-model-extraction/
        ├── SKILL.md
        ├── coverage-selection.md
        ├── api-reference.md
        └── artifact-workflow.md
```

- [Model extraction skill](skills/better-benchmark-model-extraction/SKILL.md):
  concise workflow, examples, and validation gates.
- [Coverage selection](skills/better-benchmark-model-extraction/coverage-selection.md):
  representative blocks, coupled config fields, and coverage limitations.
- [API reference](skills/better-benchmark-model-extraction/api-reference.md):
  existing capture, merge, and validation entry points.
- [Artifact workflow](skills/better-benchmark-model-extraction/artifact-workflow.md):
  trace export, runnable replay, staging, consumer checks, and final integration.
