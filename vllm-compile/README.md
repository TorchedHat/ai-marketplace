# vLLM Compilation Expert Plugin

Expert guidance for understanding and working with vLLM's custom compilation system built on PyTorch Dynamo + Inductor.

## Overview

This plugin provides deep knowledge of vLLM's compilation internals across four core stages:

1. **@support_torch_compile Decorator** - Entry point and dynamic shapes specification
2. **vllmBackend & Inductor Passes** - Custom fusion passes and LLM optimizations
3. **PiecewiseBackend** - Graph splitting and piecewise compilation
4. **CudaGraphWrapper** - CUDA graph capture and replay

## When to Use

Activate when:
- Implementing or modifying `@support_torch_compile` decorated methods
- Adding new fusion passes to vllmBackend
- Debugging graph splitting in PiecewiseBackend
- Configuring CUDA graph capture
- Understanding compilation performance bottlenecks
- Investigating guard dropping behavior

## Installation

This plugin is part of the PyTorch DevContainers AI Marketplace.

## Skills Included

### vllm-compile

Main skill providing comprehensive guidance on vLLM's custom compiler, including:
- Decorator usage and dynamic dimensions
- vllmBackend custom fusion passes
- PiecewiseBackend graph splitting
- CudaGraphWrapper CUDA graph management
- Debugging and troubleshooting

### Supporting Documentation

- **ARCHITECTURE.md** - System architecture overview
- **COMPILATION-PIPELINE.md** - End-to-end pipeline walkthrough
- **QUICK-REFERENCE.md** - One-page reference guide

## Usage

The skill activates automatically when working with vLLM compilation topics. You can also invoke it directly:

```
/vllm-compile
```

## Key Files by Stage

| Stage | File | Purpose |
|-------|------|---------|
| 1 | `decorators.py` | `@support_torch_compile` decorator |
| 1 | `wrapper.py` | Guard dropping wrapper |
| 2 | `compiler_interface.py` | `InductorStandaloneAdaptor` entry point |
| 2 | `passes/pass_manager.py` | `PostGradPassManager` (fusion orchestrator) |
| 2 | `passes/fusion/*.py` | Individual fusion passes |
| 3 | `piecewise_backend.py` | Graph splitting logic |
| 4 | `piecewise_backend.py` | CUDA graph capture/replay |
| - | `backends.py` | `CompilerManager` orchestrator |

## Common Commands

```bash
# Disable all compilation
vllm serve model --enforce-eager

# Compilation with Dynamo only (no Inductor)
vllm serve model -cc.backend=eager

# Disable CUDA graphs only
vllm serve model -cc.cudagraph_mode=NONE

# Compile specific sizes only
vllm serve model -cc.compile_sizes='[1,8,16]'

# Force cache recompilation
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve model

# Enable debug logging
VLLM_LOGGING_LEVEL=DEBUG TORCH_LOGS="+dynamo,+inductor" vllm serve model
```

## License

BSD-3-Clause

## Author

PyTorch DevContainers
