# vLLM Compilation Pipeline

End-to-end walkthrough of the 8-stage compilation pipeline from source code to CUDA graph execution.

## Pipeline Overview

```
Stage 1: Decorator Application
    ↓
Stage 2: Wrapper Creation (Guard Dropping)
    ↓
Stage 3: Dynamo Graph Capture
    ↓
Stage 4: Graph Splitting
    ↓
Stage 5: Custom Fusion Passes
    ↓
Stage 6: Inductor Lowering + Codegen
    ↓
Stage 7: CUDA Graph Capture
    ↓
Stage 8: Runtime Execution
```

## Stage 1: Decorator Application

**When**: Model definition time

```python
# vllm/models/llama.py
class LlamaAttention(nn.Module):
    @support_torch_compile(
        dynamic_arg_dims={
            "x": 0,              # Batch dim
            "positions": 0,       # Batch dim
            "kv_cache": (0, 2),  # Batch + seq dims
        }
    )
    def forward(self, x, positions, kv_cache, attn_metadata):
        # Pre-attention compute
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention (custom op - NOT compiled)
        attn_output = ops.unified_attention_with_output(
            q, k, v, kv_cache, attn_metadata
        )
        
        # Post-attention compute
        output = self.o_proj(attn_output)
        return output
```

**Effect**: Marks method for compilation, specifies dynamic dimensions

## Stage 2: Wrapper Creation

**When**: First forward pass

```python
# vllm/compilation/wrapper.py
class TorchCompileWithNoGuardsWrapper:
    def __init__(self, fn, backend, dynamic_arg_dims):
        self.fn = fn
        self.backend = backend
        self.dynamic_arg_dims = dynamic_arg_dims
        self._compiled_fn = None
    
    def __call__(self, *args, **kwargs):
        if self._compiled_fn is None:
            # First call: compile with guard dropping
            self._compiled_fn = torch.compile(
                self.fn,
                backend=self.backend,
                options={
                    "guard_filter_fn": lambda guards: [False] * len(guards)
                },
                dynamic=self._make_dynamic_shapes(args)
            )
        
        # Execute compiled function
        return self._compiled_fn(*args, **kwargs)
```

**Example Execution**:
```python
# First call
wrapper = TorchCompileWithNoGuardsWrapper(forward_fn, backend)
output = wrapper(x, positions, kv_cache, attn_metadata)
# → Triggers compilation!

# Subsequent calls
output = wrapper(x2, positions2, kv_cache2, attn_metadata2)
# → Uses cached compilation, NO guard checking
```

## Stage 3: Dynamo Graph Capture

**When**: Inside `torch.compile()` on first call

**Process**:

1. **Frame Interception**: Dynamo intercepts Python frame
2. **Bytecode Tracing**: Symbolically executes bytecode
3. **Graph Building**: Converts to FX graph
4. **Guard Recording** (then dropped): Records what assumptions were made

**Example**:

```python
# Python code
def forward(x, positions):
    q = self.q_proj(x)           # Matmul
    k = self.k_proj(x)           # Matmul
    v = self.v_proj(x)           # Matmul
    q, k = self.rotary_emb(q, k) # RoPE
    return q, k, v

# FX Graph (simplified)
graph():
    %x : [num_users=3] = placeholder[target=x]
    %q_proj_weight : [num_users=1] = get_attr[target=q_proj.weight]
    %q : [num_users=2] = call_function[target=torch.matmul](args=(%x, %q_proj_weight))
    
    %k_proj_weight : [num_users=1] = get_attr[target=k_proj.weight]
    %k : [num_users=2] = call_function[target=torch.matmul](args=(%x, %k_proj_weight))
    
    %v_proj_weight : [num_users=1] = get_attr[target=v_proj.weight]
    %v : [num_users=1] = call_function[target=torch.matmul](args=(%x, %v_proj_weight))
    
    %positions : [num_users=1] = placeholder[target=positions]
    %q_rope, %k_rope : [num_users=1] = call_function[target=rotary_emb](
        args=(%q, %k, %positions)
    )
    
    return (%q_rope, %k_rope, %v)
```

**Guards Recorded** (then dropped):
```python
guards = [
    "x.size(0) == 8",           # Batch size
    "x.size(1) == 4096",        # Hidden dim
    "x.dtype == torch.float16",
    "x.device == cuda:0",
    "positions.size(0) == 8",
    # ... many more
]

# vLLM drops ALL of these!
```

## Stage 4: Graph Splitting

**When**: In custom backend before Inductor

```python
# vllm/compilation/backends.py
class PiecewiseCompileBackend:
    def __call__(self, gm: GraphModule) -> CompiledModule:
        # Find attention nodes
        attention_nodes = [
            node for node in gm.graph.nodes
            if node.target == ops.unified_attention_with_output
        ]
        
        # Split graph
        subgraphs = self._split_at_nodes(gm, attention_nodes)
        
        # Result: [pre_attn_graph, attn_node, post_attn_graph]
        return PiecewiseCompiledModule(subgraphs)
```

**Example Split**:

```
Original Graph:
[q_proj] → [k_proj] → [v_proj] → [RoPE] → [Attention] → [o_proj]

After Splitting:
Subgraph 0: [q_proj] → [k_proj] → [v_proj] → [RoPE]  # To be compiled
Subgraph 1: [Attention]                               # Custom op, eager
Subgraph 2: [o_proj]                                  # To be compiled
```

## Stage 5: Custom Fusion Passes

**When**: Before Inductor lowering, applied to each compilable subgraph

**Example**: RoPE + KV Cache Fusion

```python
# vllm/compilation/passes/fusion/rope_kvcache_fusion.py

# Pattern to match
class RoPEKVCacheFusionPass(VllmInductorPass):
    def pattern(self):
        return PatternMatcher([
            # RoPE application
            Match(torch.mul, args=[
                Var("q"),
                Match(torch.cos, args=[Var("positions")])
            ]),
            # KV cache update
            Match(ops.kv_cache_update, args=[
                Var("k_rope"),
                Var("v"),
                Var("cache")
            ])
        ])
    
    def replacement(self, match):
        # Fused Triton kernel
        return FusedRoPEKVCacheOp(
            match["q"],
            match["positions"],
            match["cache"]
        )

# Before fusion:
graph():
    %cos = torch.cos(%positions)
    %q_rope = torch.mul(%q, %cos)
    %k_rope = torch.mul(%k, %cos)
    %cache_updated = ops.kv_cache_update(%k_rope, %v, %cache)
    
# After fusion:
graph():
    %q_rope, %cache_updated = ops.fused_rope_kv_cache(
        %q, %k, %v, %positions, %cache
    )
```

**All Fusion Passes Applied**:
1. AllReduce + RMSNorm
2. RoPE + KV Cache
3. SiLU + Mul + Quantization
4. Collective communication fusion
5. Multi-head projection fusion
6. ... (10+ total)

## Stage 6: Inductor Lowering + Codegen

**When**: After fusion passes

**Process**:

1. **FX → LoopIR**: Lower FX graph to loop-level intermediate representation
2. **Kernel Fusion**: Fuse pointwise/reduction ops
3. **Triton Codegen**: Generate Triton kernels for GPU
4. **C++ Codegen**: Generate C++ for CPU ops
5. **Compilation**: Compile to .so files

**Example**:

```python
# FX Graph (after fusion)
%q_proj = torch.matmul(%x, %q_weight)
%k_proj = torch.matmul(%x, %k_weight)
%v_proj = torch.matmul(%x, %v_weight)

# Inductor recognizes: 3 matmuls with same input
# → Fused batched matmul

# Generated Triton Kernel (simplified)
@triton.jit
def fused_qkv_proj_kernel(
    x_ptr, q_w_ptr, k_w_ptr, v_w_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    batch_size, hidden_dim, head_dim
):
    # Single kernel computes all 3 projections
    pid = tl.program_id(0)
    
    # Load input (reused 3 times)
    x = tl.load(x_ptr + pid * hidden_dim)
    
    # Q projection
    q_w = tl.load(q_w_ptr + ...)
    q_out = tl.dot(x, q_w)
    tl.store(q_out_ptr + ..., q_out)
    
    # K projection
    k_w = tl.load(k_w_ptr + ...)
    k_out = tl.dot(x, k_w)
    tl.store(k_out_ptr + ..., k_out)
    
    # V projection
    v_w = tl.load(v_w_ptr + ...)
    v_out = tl.dot(x, v_w)
    tl.store(v_out_ptr + ..., v_out)

# Compiled to CUDA PTX → .so file
```

**Artifacts**:
- `~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/inductor_code.py`
- `~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/compiled.so`

## Stage 7: CUDA Graph Capture

**When**: After compilation, before first execution

```python
# vllm/compilation/cuda_graph.py
class CUDAGraphManager:
    def capture_piecewise(self, compiled_subgraphs, capture_sizes):
        cuda_graphs = {}
        
        for batch_size in capture_sizes:
            # Create dummy inputs
            dummy_inputs = self._create_dummy_inputs(batch_size)
            
            # Warmup
            for _ in range(3):
                for sg in compiled_subgraphs:
                    if sg.is_compiled:
                        sg(*dummy_inputs)
            
            # Capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(graph):
                outputs = []
                for sg in compiled_subgraphs:
                    if sg.is_compiled:
                        out = sg(*dummy_inputs)
                        outputs.append(out)
                    else:
                        outputs.append(None)  # Eager, not captured
            
            cuda_graphs[batch_size] = graph
        
        return cuda_graphs

# Capture for batch sizes: [1, 2, 4, 8, 16, 32, 64, ...]
graphs = manager.capture_piecewise(subgraphs, capture_sizes=[1, 2, 4, 8, ...])
```

**Piecewise vs Full CUDA Graphs**:

```
Full CUDA Graph (standard):
┌────────────────────────────────────────┐
│  [Entire Model in One CUDA Graph]     │
│  - Large memory overhead               │
│  - Inflexible (can't change anything)  │
└────────────────────────────────────────┘

Piecewise CUDA Graphs (vLLM):
┌──────────────┐     ┌──────────────┐
│ Subgraph 0   │ →   │ Subgraph 2   │
│ [CUDA Graph] │  ↓  │ [CUDA Graph] │
└──────────────┘  │  └──────────────┘
                  ↓
            [Subgraph 1]
            [Eager Attn]
            
- Smaller memory per graph
- Flexible attention layer
- ~1-2 μs overhead per graph replay
```

## Stage 8: Runtime Execution

**When**: Inference time

```python
# vllm/compilation/piecewise_backend.py
class PiecewiseCompiledModule:
    def forward(self, *args, **kwargs):
        x = args[0]
        batch_size = x.size(0)
        
        # Find closest CUDA graph size
        graph_size = self._find_closest_size(batch_size)
        
        # Execute each subgraph
        for i, subgraph in enumerate(self.subgraphs):
            if subgraph.is_compiled:
                # Replay CUDA graph
                cuda_graph = self.cuda_graphs[graph_size][i]
                
                # Copy inputs to graph's static buffers
                self._copy_to_static_inputs(x, cuda_graph)
                
                # Replay
                cuda_graph.replay()
                
                # Copy outputs from graph's static buffers
                x = self._copy_from_static_outputs(cuda_graph)
            else:
                # Eager execution (attention)
                x = subgraph(x, **kwargs)
        
        return x
```

**Performance Timeline**:

```
Cold Start (No Cache):
├─ Stage 1-2: <1ms (decorator overhead)
├─ Stage 3: 2-5 seconds (Dynamo tracing)
├─ Stage 4: <100ms (graph splitting)
├─ Stage 5: 1-3 seconds (fusion passes)
├─ Stage 6: 10-20 seconds (Inductor compilation)
└─ Stage 7: 2-5 seconds (CUDA graph capture)
Total: ~15-30 seconds

Warm Start (Cache Hit):
├─ Stage 1-2: <1ms
├─ Load from cache: 100-500ms
└─ Stage 7: 2-5 seconds (CUDA graph capture)
Total: ~3-6 seconds

Runtime (Per Request):
├─ CUDA graph replay: 1-2 μs per subgraph
├─ Eager attention: ~100-500 μs
└─ Total: ~10-20% faster than eager mode
```

## Cache Architecture

**Location**: `~/.cache/vllm/torch_compile_cache/`

**Cache Key Components**:
1. Model code hash (SHA256 of source files)
2. vLLM version
3. PyTorch version
4. CUDA version
5. CompilationConfig settings

**Cache Structure**:
```
<cache_hash>/
├── metadata.json                    # Cache metadata
├── rank_0_0/                        # GPU 0, subgraph 0
│   ├── computation_graph.py         # Original FX graph
│   ├── transformed_code.py          # After Dynamo transforms
│   ├── inductor_code.py             # Generated kernel code
│   ├── compiled.so                  # Compiled kernels
│   └── triton_kernels/
│       ├── kernel_0.py
│       ├── kernel_1.py
│       └── ...
├── rank_0_1/                        # GPU 0, subgraph 1
└── rank_1_0/                        # GPU 1, subgraph 0 (multi-GPU)
```

**Cache Invalidation**:
- Model code change → New hash → Cache miss
- vLLM version upgrade → Cache miss
- Config change (e.g., backend) → Cache miss

**Debug Cache**:
```bash
# View cache
ls -lah ~/.cache/vllm/torch_compile_cache/

# View compiled code
cat ~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/inductor_code.py

# Clear cache
rm -rf ~/.cache/vllm/torch_compile_cache/
```

## Debugging the Pipeline

**Enable Verbose Logging**:
```bash
VLLM_LOGGING_LEVEL=DEBUG \
TORCH_LOGS="+dynamo,+inductor,+graph" \
TORCH_TRACE=/tmp/vllm_trace \
vllm serve model
```

**Inspect at Each Stage**:

1. **Stage 3 (Dynamo)**: Check `/tmp/vllm_trace/dynamo_guards.txt`
2. **Stage 4 (Splitting)**: Enable `VLLM_LOGGING_LEVEL=DEBUG`, look for "Splitting graph at..."
3. **Stage 5 (Fusion)**: Check inductor logs for "Applied fusion pass: ..."
4. **Stage 6 (Codegen)**: Inspect `inductor_code.py` in cache
5. **Stage 7 (CUDA graphs)**: Look for "Captured CUDA graph for batch_size=..."

**Common Issues**:

| Stage | Issue | Fix |
|-------|-------|-----|
| 3 | Graph breaks | Remove print/pdb in model code |
| 4 | Split point not found | Check `splitting_ops` config |
| 5 | Fusion not applied | Pattern doesn't match - check graph structure |
| 6 | OOM during compilation | Reduce `compile_sizes`, add swap |
| 7 | CUDA graph capture fails | Disable with `-cc.cudagraph_mode=NONE` |

## Summary

The vLLM compilation pipeline is a sophisticated 8-stage process that:

1. **Starts** with decorated model code
2. **Captures** full FX graph via Dynamo (with guard dropping)
3. **Splits** at attention boundaries
4. **Optimizes** with LLM-specific fusion passes
5. **Compiles** to efficient Triton/C++ kernels
6. **Captures** piecewise CUDA graphs
7. **Executes** with ~10-20% speedup vs eager

**Key Innovation**: Guard dropping + piecewise compilation enables **one compiled artifact for all batch sizes** with **minimal overhead** at runtime.
