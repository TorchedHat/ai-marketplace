---
name: vllm-compile
description: Expert guidance for vLLM's custom compiler - focusing on @support_torch_compile decorator, vllmBackend/Inductor Passes, PiecewiseBackend, and CudaGraphWrapper.
---

# vLLM torch.compile Expert

Expert guidance for understanding and working with vLLM's custom compilation system, focusing on the four core stages:

1. **@support_torch_compile Decorator** - Entry point and dynamic shapes specification
2. **vllmBackend & Inductor Passes** - Custom fusion passes and LLM optimizations
3. **PiecewiseBackend** - Graph splitting and piecewise compilation
4. **CudaGraphWrapper** - CUDA graph capture and replay

## Purpose

This skill provides deep knowledge of vLLM's compilation internals:
- How `@support_torch_compile` marks and configures compilable methods
- How vllmBackend applies LLM-specific fusion passes
- How PiecewiseBackend splits graphs at attention boundaries
- How CudaGraphWrapper captures and replays CUDA graphs

## When to Use This Skill

Activate when:
- Implementing or modifying `@support_torch_compile` decorated methods
- Adding new fusion passes to vllmBackend
- Debugging graph splitting in PiecewiseBackend
- Configuring CUDA graph capture
- Understanding compilation performance bottlenecks
- Investigating guard dropping behavior

## Pipeline Overview

```
┌──────────────────────────────────────────────────┐
│   Stage 1: @support_torch_compile Decorator      │
│   - Marks methods for compilation                │
│   - Specifies dynamic dimensions                 │
│   - Creates compilation wrapper                  │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│   Stage 2: vllmBackend & Inductor Passes         │
│   - Custom LLM fusion passes                     │
│   - Pattern matching and replacement             │
│   - Graph optimization                           │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│   Stage 3: PiecewiseBackend                      │
│   - Split at attention operations                │
│   - Separate compilable from eager subgraphs     │
│   - Manage piecewise execution                   │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│   Stage 4: CudaGraphWrapper                      │
│   - Capture CUDA graphs per subgraph             │
│   - Manage static buffers                        │
│   - Replay at inference time                     │
└──────────────────────────────────────────────────┘
```

## Stage 1: @support_torch_compile Decorator

**File**: `vllm/compilation/decorators.py`

The decorator is the entry point - it marks methods for compilation and specifies dynamic dimensions.

### Usage

```python
from vllm.compilation.decorators import support_torch_compile

class LlamaAttention(nn.Module):
    @support_torch_compile(
        dynamic_arg_dims={
            "x": 0,                 # First dim (batch) is dynamic
            "positions": 0,          # First dim (batch) is dynamic
            "kv_cache": (0, 2),     # Dims 0 and 2 are dynamic
        }
    )
    def forward(self, x, positions, kv_cache, attn_metadata):
        # Compilation happens automatically
        q = self.q_proj(x)
        k, v = self.k_proj(x), self.v_proj(x)
        # ... rest of forward pass
        return output
```

### What It Does

1. **Wraps the method** in a compilation-aware wrapper
2. **Specifies dynamic shapes** to Dynamo (which dims can change)
3. **Enables guard dropping** (unsafe but effective for LLM inference)
4. **Triggers compilation** on first call

### Key Files

- `vllm/compilation/decorators.py` - Decorator implementation
- `vllm/compilation/wrapper.py` - `TorchCompileWithNoGuardsWrapper`

### Dynamic Dimensions Explained

```python
dynamic_arg_dims={
    "x": 0,          # x.shape[0] can vary (batch size)
    "positions": 0,   # positions.shape[0] can vary
    "kv_cache": (0, 2),  # kv_cache.shape[0] and shape[2] can vary
}

# Tells Dynamo:
# - Don't add guards like "x.size(0) == 8"
# - Create symbolic shapes: x.shape[0] = s0, x.shape[1] = 4096 (static)
# - Generate code that works for any batch size
```

## Stage 2: vllmBackend & Inductor Passes

**Files**: 
- `vllm/compilation/backends.py`
- `vllm/compilation/passes/`

vllmBackend applies **custom LLM-specific fusion passes** before Inductor compilation.

### Pass Architecture

```python
# vllm/compilation/passes/vllm_inductor_pass.py
class VllmInductorPass:
    """Base class for all vLLM fusion passes"""
    
    def pattern(self) -> PatternMatcher:
        """Define pattern to match in FX graph"""
        raise NotImplementedError
    
    def replacement(self, match: Match) -> Node:
        """Generate optimized replacement"""
        raise NotImplementedError
    
    def apply(self, gm: GraphModule) -> bool:
        """Apply pass to graph, return True if changed"""
        matches = self.pattern().find_matches(gm.graph)
        for match in matches:
            replacement = self.replacement(match)
            gm.graph.replace_subgraph(match, replacement)
        return len(matches) > 0
```

### Example Pass: RoPE + KV Cache Fusion

```python
# vllm/compilation/passes/fusion/rope_kvcache_fusion.py

class RoPEKVCacheFusionPass(VllmInductorPass):
    def pattern(self):
        """Match: RoPE → KV cache update sequence"""
        return PatternMatcher([
            Match("rope_embedding", Var("q"), Var("k"), Var("pos")),
            Match("kv_cache_update", Var("k_rope"), Var("v"), Var("cache"))
        ])
    
    def replacement(self, match):
        """Replace with fused Triton kernel"""
        return FusedRoPEKVCache(
            match["q"], match["k"], match["v"],
            match["pos"], match["cache"]
        )

# Before fusion (2 kernels):
%q_rope, %k_rope = rope_embedding(%q, %k, %positions)
%cache_new = kv_cache_update(%k_rope, %v, %kv_cache)

# After fusion (1 kernel):
%q_rope, %cache_new = fused_rope_kv_cache(%q, %k, %v, %positions, %cache)
```

### All vLLM Fusion Passes

1. **AllReduceRMSNormFusion** - Overlap TP communication with normalization
2. **RoPEKVCacheFusion** - Fuse positional encoding with cache update
3. **SiLUMulQuantFusion** - Fuse activation + multiply + quantization
4. **CollectiveFusion** - Pipeline communication primitives
5. **MultiHeadProjectionFusion** - Fuse Q/K/V projections
6. **... (10+ total)**

## Stage 3: PiecewiseBackend

**File**: `vllm/compilation/piecewise_backend.py`

PiecewiseBackend **splits the graph at attention operations** to enable hybrid compilation.

### Why Split?

**Problem**: Attention is complex and dynamic
- KV cache management
- Flash attention variants
- Variable sequence lengths
- PagedAttention

**Solution**: Leave attention as custom op (eager), compile everything else

### How It Works

```python
class PiecewiseBackend:
    def __init__(self, splitting_ops=["vllm::unified_attention_with_output"]):
        self.splitting_ops = splitting_ops
    
    def __call__(self, gm: GraphModule) -> CompiledModule:
        # 1. Find all attention nodes
        split_nodes = [
            node for node in gm.graph.nodes
            if node.target in self.splitting_ops
        ]
        
        # 2. Split graph at these nodes
        subgraphs = self._split_graph(gm, split_nodes)
        
        # 3. Compile non-attention subgraphs
        compiled_subgraphs = []
        for sg in subgraphs:
            if self._is_attention(sg):
                compiled_subgraphs.append(sg)  # Keep eager
            else:
                compiled_subgraphs.append(
                    self._compile_subgraph(sg)  # Compile with Inductor
                )
        
        # 4. Return piecewise module
        return PiecewiseCompiledModule(compiled_subgraphs)
```

### Example Split

```
Original Graph:
┌──────────────────────────────────────────────────────────┐
│ [Q/K/V Proj] → [RoPE] → [Attn] → [O Proj] → [MLP]       │
└──────────────────────────────────────────────────────────┘

After Splitting:
┌────────────────────┐   ┌────────┐   ┌──────────────┐
│ Subgraph 0         │   │ Subgr1 │   │ Subgraph 2   │
│ [Q/K/V] → [RoPE]   │ → │ [Attn] │ → │ [O] → [MLP]  │
│ (Compiled)         │   │ (Eager)│   │ (Compiled)   │
└────────────────────┘   └────────┘   └──────────────┘
```

### Configuration

```python
from vllm.config.compilation import CompilationConfig

config = CompilationConfig(
    splitting_ops=[
        "vllm::unified_attention_with_output",
        # Add custom ops to split at
    ],
    backend="inductor",  # or "eager" to skip Inductor
)
```

## Stage 4: CudaGraphWrapper

**File**: `vllm/compilation/cuda_graph.py`

CudaGraphWrapper **captures CUDA graphs per compiled subgraph** for maximum performance.

### Piecewise CUDA Graphs

Unlike standard CUDA graphs (entire model), vLLM captures **one graph per compiled subgraph**.

```python
class CudaGraphWrapper:
    def __init__(self, compiled_subgraphs, capture_sizes):
        self.subgraphs = compiled_subgraphs
        self.cuda_graphs = {}
        
        # Capture graph for each batch size
        for batch_size in capture_sizes:
            self.cuda_graphs[batch_size] = self._capture(batch_size)
    
    def _capture(self, batch_size):
        """Capture CUDA graph for given batch size"""
        # Create dummy inputs
        inputs = self._create_dummy_inputs(batch_size)
        
        # Warmup (3 iterations)
        for _ in range(3):
            for sg in self.subgraphs:
                if sg.is_compiled:
                    sg(*inputs)
        
        # Capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph):
            for sg in self.subgraphs:
                if sg.is_compiled:
                    sg(*inputs)  # Captured!
                # Eager subgraphs NOT captured
        
        return graph
    
    def replay(self, inputs, batch_size):
        """Replay captured CUDA graph"""
        # Get graph for this batch size
        graph = self.cuda_graphs.get(batch_size)
        
        if graph:
            # Copy inputs to static buffers
            self._copy_to_static(inputs)
            
            # Replay (~1-2 μs)
            graph.replay()
            
            # Copy outputs from static buffers
            return self._copy_from_static()
        else:
            # Fallback to eager
            return self._run_eager(inputs)
```

### Capture Sizes

```python
# Default capture sizes (powers of 2 + common sizes)
capture_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]

# At runtime: find closest size
batch_size = 10
closest = min(capture_sizes, key=lambda x: abs(x - batch_size))
# → Use graph for batch_size=8
```

### Benefits

| Aspect | Full CUDA Graph | Piecewise CUDA Graph |
|--------|-----------------|----------------------|
| **Memory** | High (entire model) | Low (per subgraph) |
| **Flexibility** | None (all or nothing) | High (eager attention) |
| **Overhead** | ~1 μs | ~1-2 μs per subgraph |
| **Capture time** | Long | Short |

## Entry Points (Actual Implementation)

### Entry Point Flow

```
@support_torch_compile(...)           [decorators.py]
    ↓
TorchCompileWithNoGuardsWrapper       [wrapper.py]
    ↓
torch.compile(..., backend="vllm")
    ↓
CompilerManager                        [backends.py]
    ↓
InductorStandaloneAdaptor/InductorAdaptor  [compiler_interface.py]
    ↓
torch._inductor.standalone_compile()   [PyTorch]
    ↓ (via post_grad_custom_post_pass hook)
PostGradPassManager                    [passes/pass_manager.py]
    ↓
VllmInductorPass (fusion passes)      [passes/fusion/*.py]
    ↓
PiecewiseBackend (graph splitting)    [piecewise_backend.py]
    ↓
CUDAGraph capture                     [piecewise_backend.py]
```

### Key Entry Points by File

**Stage 1: Decorator → Wrapper**
```python
# vllm/compilation/decorators.py
@support_torch_compile(dynamic_arg_dims={"x": 0})
def forward(self, x): ...

# vllm/compilation/wrapper.py
class TorchCompileWithNoGuardsWrapper:
    def __call__(self, *args):
        return torch.compile(self.fn, backend="vllm")(args)
```

**Stage 2a: CompilerManager → InductorAdaptor**
```python
# vllm/compilation/backends.py
class CompilerManager:
    def __init__(self, config):
        self.compiler = make_compiler(config)  # Creates InductorStandaloneAdaptor
    
    def compile(self, graph, inputs, compile_range):
        # Entry point: calls compiler.compile()
        return self.compiler.compile(graph, inputs, ...)

# vllm/compilation/compiler_interface.py  
class InductorStandaloneAdaptor(CompilerInterface):
    def compile(self, graph, example_inputs, ...):
        # KEY ENTRY POINT: Calls PyTorch's standalone Inductor
        from torch._inductor import standalone_compile
        compiled = standalone_compile(graph, example_inputs, ...)
        return compiled, handle
```

**Stage 2b: PassManager (via Inductor Hook)**
```python
# vllm/compilation/backends.py (VllmDynamoBackend.__init__)
# Register PostGradPassManager with Inductor via config
self.inductor_config = {
    "post_grad_custom_post_pass": PostGradPassManager(),  # ← KEY HOOK!
    ...
}

# vllm/compilation/passes/pass_manager.py
class PostGradPassManager(CustomGraphPass):
    def __call__(self, graph: fx.Graph):
        # Inductor calls this AFTER its own passes
        # This is where ALL vLLM fusion passes run
        for pass_ in self.passes:
            pass_(graph)  # ← Runs fusion passes!
```

**Stage 3: PiecewiseBackend**
```python
# vllm/compilation/piecewise_backend.py
class PiecewiseBackend:
    def __call__(self, graph: fx.GraphModule):
        # Split graph at attention operations
        subgraphs = self._split_at_ops(graph, splitting_ops)
        
        # Compile each non-attention subgraph
        for sg in subgraphs:
            if not is_attention(sg):
                compiled = self.compiler.compile(sg)  # ← Calls CompilerManager
```

**Stage 4: CUDA Graph Wrapper**
```python
# vllm/compilation/piecewise_backend.py (integrated)
def capture_cuda_graphs(compiled_subgraphs, capture_sizes):
    graphs = {}
    for size in capture_sizes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for sg in compiled_subgraphs:
                sg(dummy_input_size)  # ← Capture!
        graphs[size] = graph
    return graphs
```

## End-to-End Flow

Putting all 4 stages together:

```python
# 1. Decorator marks method
@support_torch_compile(dynamic_arg_dims={"x": 0})
def forward(self, x):
    q = self.q_proj(x)
    attn_out = ops.unified_attention_with_output(...)  # Split point!
    out = self.o_proj(attn_out)
    return out

# 2. First call triggers compilation
model.forward(x)
    ↓
# Dynamo captures full graph
# CompilerManager → InductorStandaloneAdaptor → standalone_compile()
# PostGradPassManager (via post_grad_custom_post_pass hook) applies fusion passes
# PiecewiseBackend splits at attention
# CudaGraphWrapper captures subgraphs

# 3. Subsequent calls use cached compilation
model.forward(x2)  # Fast! ~10-20% speedup
```

## Key Files by Stage

### Stage 1: Decorator
- `vllm/compilation/decorators.py` - `@support_torch_compile` implementation
- `vllm/compilation/wrapper.py` - `TorchCompileWithNoGuardsWrapper`

### Stage 2: vllmBackend & Passes
- `vllm/compilation/backends.py` - Backend registration
- `vllm/compilation/passes/vllm_inductor_pass.py` - Pass base class
- `vllm/compilation/passes/fusion/` - All fusion passes:
  - `allreduce_rms_fusion.py`
  - `rope_kvcache_fusion.py`
  - `silu_mul_quant_fusion.py`
  - `collective_fusion.py`
  - ... (10+ total)

### Stage 3: PiecewiseBackend
- `vllm/compilation/piecewise_backend.py` - Graph splitting logic
- `vllm/compilation/backends.py` - Subgraph compilation

### Stage 4: CudaGraphWrapper
- `vllm/compilation/cuda_graph.py` - CUDA graph capture/replay
- `vllm/compilation/piecewise_backend.py` - Integration with piecewise

## Configuration

```python
from vllm.config.compilation import CompilationConfig

config = CompilationConfig(
    # Stage 2: Backend selection
    backend="inductor",  # or "eager" to skip Inductor
    
    # Stage 3: Splitting configuration
    splitting_ops=["vllm::unified_attention_with_output"],
    
    # Stage 4: CUDA graph settings
    cudagraph_mode=CUDAGraphMode.PIECEWISE,
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    
    # Optional: compile specific sizes only
    compile_sizes=[1, 8, 16],  # Only compile these batch sizes
)
```

## Debugging by Stage

### Stage 1: Decorator Issues

```python
# Check if decorator is applied
import inspect
method = model_class.forward
print(hasattr(method, '_vllm_torch_compile_wrapped'))  # Should be True

# Verify dynamic dimensions
wrapper = method._vllm_torch_compile_wrapper
print(wrapper.dynamic_arg_dims)  # Should show {"x": 0, ...}
```

### Stage 2: Inductor/Pass Issues

```bash
# Enable Inductor logging
TORCH_LOGS="+inductor,+graph" vllm serve model

# Enable vLLM pass debug logging
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_PATTERN_MATCH_DEBUG=1 \
vllm serve model

# Check which passes ran
# Look for: "Applied fusion pass: RoPEKVCacheFusionPass"

# Inspect Inductor code cache
ls ~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/
cat inductor_code.py  # Generated kernels
```

### Stage 3: Piecewise Backend Issues

```bash
# Enable splitting debug
VLLM_LOGGING_LEVEL=DEBUG vllm serve model
# Look for: "Splitting graph at node: vllm::unified_attention_with_output"

# Verify split points
# Check logs for "Created N subgraphs"
```

### Stage 4: CUDA Graph Issues

```bash
# Disable CUDA graphs to test
vllm serve model -cc.cudagraph_mode=NONE

# Check capture sizes
vllm serve model -cc.cudagraph_capture_sizes='[1,8,16]'

# Enable CUDA graph logging
VLLM_LOGGING_LEVEL=DEBUG vllm serve model
# Look for: "Captured CUDA graph for batch_size=X"
```

## Common CLI Commands

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

# Use standalone Inductor (PyTorch 2.8+)
VLLM_USE_STANDALONE_COMPILE=1 vllm serve model

# Disable all passes
vllm serve model -cc.pass_config.enable_all=false

# Enable specific passes only
vllm serve model \
  -cc.pass_config.fuse_rope_kvcache=true \
  -cc.pass_config.fuse_allreduce_rms=true
```

## Performance Metrics

### Compilation Time
- **Cold start** (no cache): 10-30 seconds
- **Warm start** (cache hit): <1 second
- **Per-subgraph compilation**: 1-5 seconds

### Runtime Performance
- **Overall speedup**: 10-20% vs eager mode
- **Best gains**: Multi-GPU (TP) workloads with fusion passes
- **CUDA graph overhead**: ~1-2 μs per subgraph replay

### Memory Usage
- **Cache on disk**: ~100-500 MB per model
- **Additional runtime**: ~5-10% for CUDA graphs
- **Savings from fusion**: 20-30% (eliminates intermediates)

## Quick Reference

### Most Important Files

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

### Key Environment Variables

```bash
VLLM_USE_STANDALONE_COMPILE=1   # Use torch._inductor.standalone_compile
VLLM_DISABLE_COMPILE_CACHE=1    # Force recompilation
VLLM_LOGGING_LEVEL=DEBUG        # Enable debug logging
VLLM_PATTERN_MATCH_DEBUG=1      # Debug fusion pass matching
TORCH_LOGS="+inductor,+dynamo"  # PyTorch compilation logs
TORCH_TRACE=/tmp/trace          # Save compilation traces
```

## Related Skills

- **[pytorch-dynamo](../pytorch-dynamo/SKILL.md)** - Understanding Dynamo graph capture and symbolic execution
- **[pytorch-inductor](../pytorch-inductor/SKILL.md)** - Understanding Inductor lowering and code generation

## Official Documentation

- [vLLM torch.compile docs](https://docs.vllm.ai/en/latest/design/torch_compile/)
- [Debug guide](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/)
- [PyTorch compile docs](https://pytorch.org/docs/stable/torch.compiler.html)
