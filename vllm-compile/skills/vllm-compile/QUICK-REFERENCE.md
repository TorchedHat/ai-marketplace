# vLLM Compilation Quick Reference

One-page reference for vLLM's custom compilation system.

**For detailed explanations, see [SKILL.md](SKILL.md).**

## 4-Stage Pipeline

```
1. @support_torch_compile → TorchCompileWithNoGuardsWrapper
2. CompilerManager → InductorStandaloneAdaptor → PostGradPassManager (fusion passes)
3. PiecewiseBackend (graph splitting at attention)
4. CUDA Graph capture per subgraph
```

## Key Entry Points

| Stage | File | Entry Point |
|-------|------|-------------|
| 1 | `decorators.py` | `@support_torch_compile(dynamic_arg_dims={...})` |
| 1 | `wrapper.py` | `TorchCompileWithNoGuardsWrapper.__call__` |
| 2 | `backends.py` | `CompilerManager.compile()` |
| 2 | `compiler_interface.py` | `InductorStandaloneAdaptor.compile()` → `torch._inductor.standalone_compile()` |
| 2 | `passes/pass_manager.py` | `PostGradPassManager.__call__()` (via `post_grad_custom_post_pass` hook) |
| 3 | `piecewise_backend.py` | `PiecewiseBackend.__call__()` |
| 4 | `piecewise_backend.py` | CUDA graph capture logic |

## File Locations

```
vllm/compilation/
├── decorators.py              # Stage 1: @support_torch_compile
├── wrapper.py                 # Stage 1: Guard dropping wrapper
├── backends.py                # CompilerManager orchestrator
├── compiler_interface.py      # Stage 2: Inductor adaptor
├── piecewise_backend.py       # Stage 3 & 4: Splitting + CUDA graphs
├── passes/
│   ├── pass_manager.py        # Stage 2: PostGradPassManager
│   ├── inductor_pass.py       # Pass base classes
│   ├── vllm_inductor_pass.py  # VllmInductorPass base
│   └── fusion/                # Stage 2: Fusion passes
│       ├── allreduce_rms_fusion.py
│       ├── rope_kvcache_fusion.py
│       ├── act_quant_fusion.py
│       ├── collective_fusion.py
│       └── ... (more)
└── config/compilation.py      # CompilationConfig
```

## Common Commands

### Disable Components

```bash
# No compilation at all
vllm serve model --enforce-eager

# Dynamo only (no Inductor)
vllm serve model -cc.backend=eager

# No CUDA graphs
vllm serve model -cc.cudagraph_mode=NONE

# Force recompilation
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve model
```

### Debug Logging

```bash
# Full debug output
VLLM_LOGGING_LEVEL=DEBUG \
TORCH_LOGS="+dynamo,+inductor" \
vllm serve model

# Fusion pass debugging
VLLM_PATTERN_MATCH_DEBUG=1 vllm serve model

# Save trace
TORCH_TRACE=/tmp/trace vllm serve model
```

### Configuration

```bash
# Compile specific sizes only
vllm serve model -cc.compile_sizes='[1,8,16]'

# CUDA graph capture sizes
vllm serve model -cc.cudagraph_capture_sizes='[1,2,4,8,16]'

# Use standalone compile (PyTorch 2.8+)
VLLM_USE_STANDALONE_COMPILE=1 vllm serve model

# Disable specific passes
vllm serve model \
  -cc.pass_config.fuse_allreduce_rms=false \
  -cc.pass_config.fuse_rope_kvcache=false
```

## Fusion Passes

| Pass | File | Speedup | What It Fuses |
|------|------|---------|---------------|
| AllReduceRMSNorm | `allreduce_rms_fusion.py` | 1.8x | TP allreduce + RMSNorm |
| RoPEKVCache | `rope_kvcache_fusion.py` | 2.1x | RoPE + KV cache update |
| ActivationQuant | `act_quant_fusion.py` | 1.5x | SiLU/GELU + multiply + quantization |
| CollectiveFusion | `collective_fusion.py` | 1.3x | Pipelined communication |
| RMSNormQuant | `rms_quant_fusion.py` | 1.4x | RMSNorm + quantization |
| AttnQuant | `attn_quant_fusion.py` | 1.3x | Attention output + quantization |

## Dynamic Shapes Modes

| Mode | Safety | Performance | Use Case |
|------|--------|-------------|----------|
| `BACKED` | ⚠️ Unsafe | ⭐⭐⭐ Best | Production (default) |
| `BACKED_SIZE_OBLIVIOUS` | ⚠️ Experimental | ⭐⭐ Good | Testing safer mode |
| `UNBACKED` | ✅ Safe | ⭐ OK | Debugging guard issues |

```bash
# Configure mode
vllm serve model -cc.dynamic_shapes_config.type=unbacked
```

## Cache Locations

```
~/.cache/vllm/torch_compile_cache/
└── <hash>/
    ├── rank_0_0/              # GPU 0, subgraph 0
    │   ├── computation_graph.py
    │   ├── transformed_code.py
    │   ├── inductor_code.py
    │   └── compiled.so
    ├── rank_0_1/              # GPU 0, subgraph 1
    └── rank_1_0/              # GPU 1, subgraph 0
```

## Debugging Checklist

### Stage 1: Decorator
- [ ] Decorator is applied: `hasattr(method, '_vllm_torch_compile_wrapped')`
- [ ] Dynamic dims correct: Check `wrapper.dynamic_arg_dims`

### Stage 2: Inductor/Passes
- [ ] Passes running: `VLLM_LOGGING_LEVEL=DEBUG` → Look for "Applied fusion pass:"
- [ ] Cache hit/miss: Check logs for "Inductor code cache hit/miss"
- [ ] Inspect generated code: `cat ~/.cache/.../inductor_code.py`

### Stage 3: Piecewise
- [ ] Graph split correctly: Look for "Splitting graph at node:"
- [ ] Subgraph count: Check "Created N subgraphs"
- [ ] Attention nodes found: Verify splitting_ops in config

### Stage 4: CUDA Graphs
- [ ] Graphs captured: Look for "Captured CUDA graph for batch_size=X"
- [ ] Sizes match requests: Check `-cc.cudagraph_capture_sizes`
- [ ] Try without graphs: `-cc.cudagraph_mode=NONE`

## Environment Variables

```bash
# Compilation
VLLM_USE_STANDALONE_COMPILE=1   # Use standalone Inductor (PyTorch 2.8+)
VLLM_DISABLE_COMPILE_CACHE=1    # Force recompilation
VLLM_ENABLE_PREGRAD_PASSES=0    # Disable Inductor pre-grad passes (default)
VLLM_USE_MEGA_AOT_ARTIFACT=1    # Use AOT compilation artifact (PyTorch 2.10+)

# Logging
VLLM_LOGGING_LEVEL=DEBUG        # Enable debug logging
VLLM_PATTERN_MATCH_DEBUG=1      # Debug fusion pattern matching
TORCH_LOGS="+dynamo,+inductor"  # PyTorch compilation logs
TORCH_TRACE=/tmp/trace          # Save compilation traces

# CUDA
CUDA_VISIBLE_DEVICES=0          # GPU selection
```

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Graph breaks | "Graph break detected" | Remove print/pdb in model code |
| OOM during compile | Out of memory | Use `-cc.compile_sizes='[1,8,16]'` |
| Cache invalidation | Slow startup | Don't change model code |
| Guard violations | Wrong results | Use `-cc.dynamic_shapes_config.type=unbacked` |
| Missing splits | Not splitting | Check `splitting_ops` config |
| CUDA graph fails | Capture errors | Try `-cc.cudagraph_mode=NONE` |

## Configuration Object

```python
from vllm.config.compilation import CompilationConfig

config = CompilationConfig(
    # Backend
    backend="inductor",  # or "eager"
    
    # Graph splitting
    splitting_ops=["vllm::unified_attention_with_output"],
    
    # CUDA graphs
    cudagraph_mode=CUDAGraphMode.PIECEWISE,
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    
    # Compile sizes (optional)
    compile_sizes=[1, 8, 16],  # Only compile these sizes
    
    # Inductor config
    inductor_compile_config={
        "post_grad_custom_post_pass": PostGradPassManager(),
        # ... other Inductor settings
    },
    
    # Pass config
    pass_config=PassConfig(
        fuse_allreduce_rms=True,
        fuse_rope_kvcache=True,
        fuse_act_quant=True,
        # ... other passes
    )
)
```

## Related Documentation

- Main skill: [SKILL.md](SKILL.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Pipeline: [COMPILATION-PIPELINE.md](COMPILATION-PIPELINE.md)
- PyTorch Dynamo: [../pytorch-dynamo/SKILL.md](../pytorch-dynamo/SKILL.md)
- PyTorch Inductor: [../pytorch-inductor/SKILL.md](../pytorch-inductor/SKILL.md)
