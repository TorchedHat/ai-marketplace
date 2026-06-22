# vLLM Compilation System Architecture

Complete overview of vLLM's custom compilation system built on PyTorch Dynamo + Inductor.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM Model Code                            │
│                  @support_torch_compile                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              TorchCompileWithNoGuardsWrapper                    │
│                 (Guard Dropping Layer)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TorchDynamo                                  │
│            (Full-Graph Capture via Bytecode Tracing)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Graph Splitting                                │
│         (Split at unified_attention_with_output)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CompilerManager                                 │
│          (Cache Management + Backend Dispatch)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              vLLM Custom Fusion Passes                          │
│     (10+ LLM-specific optimizations via pattern matching)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TorchInductor                                  │
│         (Kernel Generation: Triton + C++ codegen)               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Piecewise CUDA Graphs                              │
│     (Automatic capture per subgraph, replay at runtime)         │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Decorator Layer (`@support_torch_compile`)

**File**: `vllm/compilation/decorators.py`

```python
@support_torch_compile(dynamic_arg_dims={"x": 0, "positions": 0})
def forward(self, x, positions):
    # Model code
    return output
```

**Responsibilities**:
- Marks methods for compilation
- Specifies which dimensions are dynamic
- Triggers wrapper creation

### 2. Guard Dropping Wrapper

**File**: `vllm/compilation/wrapper.py`

**Core Innovation**: Removes ALL guards after first compilation

```python
class TorchCompileWithNoGuardsWrapper:
    def __call__(self, *args, **kwargs):
        # First call: capture graph WITH guards
        if not self._compiled:
            compiled = torch.compile(
                self.fn,
                backend=self.backend,
                options={
                    "guard_filter_fn": self._drop_all_guards
                }
            )
            self._compiled = compiled
        
        # Subsequent calls: NO guard checking!
        return self._compiled(*args, **kwargs)
    
    def _drop_all_guards(self, guards):
        return [False for _ in guards]  # Drop ALL!
```

**Why Unsafe?**
- Skips size/dtype/device checks
- Assumes batch-agnostic operations
- May produce wrong results if assumptions violated

**Why It Works**:
- LLM ops (matmul, layernorm, etc.) ARE batch-agnostic
- Dynamic shapes handled via symbolic execution
- Data-dependent branches avoided in critical paths

### 3. TorchDynamo Integration

**Files**: 
- PyTorch: `torch/_dynamo/eval_frame.py`
- vLLM: `vllm/compilation/backends.py`

**Full-Graph Capture Required**:
```python
# Standard torch.compile allows graph breaks
@torch.compile
def forward(x):
    y = torch.matmul(x, w)
    print(y.shape)  # ← Graph break! Compiles 2 separate graphs
    return y

# vLLM requires full-graph (no breaks)
@support_torch_compile
def forward(x):
    y = torch.matmul(x, w)
    # print(y.shape)  # ← Would fail! No graph breaks allowed
    return y
```

**Enforcement**: Set `torch._dynamo.config.suppress_errors = False`

### 4. Graph Splitting

**File**: `vllm/compilation/backends.py`

**Strategy**: Split at attention operations

```python
class PiecewiseCompileBackend:
    def __init__(self, splitting_ops=["vllm::unified_attention_with_output"]):
        self.splitting_ops = splitting_ops
    
    def __call__(self, gm: GraphModule):
        # Find all attention nodes
        split_points = []
        for node in gm.graph.nodes:
            if node.target in self.splitting_ops:
                split_points.append(node)
        
        # Split graph into subgraphs
        subgraphs = self._split_at_nodes(gm, split_points)
        
        # Compile each subgraph
        compiled_subgraphs = []
        for i, sg in enumerate(subgraphs):
            if self._is_attention_subgraph(sg):
                # Attention: leave as custom op (eager)
                compiled_subgraphs.append(sg)
            else:
                # Other ops: compile + CUDA graph
                compiled = self._compile_subgraph(sg)
                compiled_subgraphs.append(compiled)
        
        return PiecewiseCompiledModule(compiled_subgraphs)
```

**Benefits**:
- Attention stays flexible (KV cache, flash attention variants)
- Other ops get CUDA graph speedup
- Smaller CUDA graphs = less memory

### 5. Compiler Manager

**File**: `vllm/compilation/compiler_interface.py`

**Responsibilities**:
1. **Cache Management**: Load/save compiled artifacts
2. **Backend Dispatch**: Route to appropriate compiler backend
3. **Version Tracking**: Invalidate cache on code changes

```python
class CompilerManager:
    def compile(self, gm: GraphModule, config: CompilationConfig):
        # Check cache
        cache_key = self._compute_cache_key(gm, config)
        if cached := self._load_from_cache(cache_key):
            return cached
        
        # Dispatch to backend
        backend = self._get_backend(config.backend)
        compiled = backend(gm)
        
        # Save to cache
        self._save_to_cache(cache_key, compiled)
        
        return compiled
```

**Cache Location**: `~/.cache/vllm/torch_compile_cache/`

**Cache Structure**:
```
<hash>/
├── rank_0_0/
│   ├── computation_graph.py    # Original FX graph
│   ├── transformed_code.py     # Dynamo output
│   ├── inductor_code.py        # Generated kernels
│   └── compiled.so             # Shared library
├── rank_0_1/
└── ...
```

### 6. Custom Fusion Passes

**File**: `vllm/compilation/passes/vllm_inductor_pass.py`

**Architecture**:

```python
class VllmInductorPass:
    """Base class for vLLM fusion passes"""
    
    @abstractmethod
    def pattern(self) -> PatternMatcher:
        """Define pattern to match"""
        pass
    
    @abstractmethod
    def replacement(self, match: Match) -> torch.fx.Node:
        """Generate replacement subgraph"""
        pass
    
    def apply(self, gm: GraphModule) -> bool:
        """Apply pass to graph"""
        matcher = self.pattern()
        matches = matcher.find_matches(gm.graph)
        
        for match in matches:
            replacement = self.replacement(match)
            gm.graph.replace_pattern(match, replacement)
        
        return len(matches) > 0
```

**Pass Registry**:
```python
# All vLLM passes
VLLM_PASSES = [
    AllReduceRMSNormFusionPass(),
    RoPEKVCacheFusionPass(),
    SiLUMulQuantFusionPass(),
    CollectiveFusionPass(),
    # ... more
]

# Apply in order
for pass_cls in VLLM_PASSES:
    changed = pass_cls.apply(gm)
    if changed:
        gm.recompile()
```

### 7. TorchInductor Integration

**Files**: `torch/_inductor/` (PyTorch)

vLLM **adds** custom passes but uses **standard** Inductor:

```
Standard Inductor Passes (PyTorch)
    ↓
vLLM Custom Passes (10+ fusion passes)
    ↓
Inductor Lowering (FX → LoopIR)
    ↓
Triton Codegen (GPU kernels)
    ↓
C++ Codegen (CPU kernels)
    ↓
Compilation (.so files)
```

**No modifications to Inductor core** - purely additive.

### 8. Piecewise CUDA Graphs

**File**: `vllm/compilation/cuda_graph.py`

**Key Insight**: Capture CUDA graph per subgraph, not full model

```python
class PiecewiseCompiledModule(nn.Module):
    def __init__(self, subgraphs):
        self.subgraphs = subgraphs
        self.cuda_graphs = []
        
        # Capture CUDA graph for each compiled subgraph
        for sg in subgraphs:
            if sg.is_compiled:
                graph = self._capture_cuda_graph(sg)
                self.cuda_graphs.append(graph)
            else:
                self.cuda_graphs.append(None)  # Eager execution
    
    def forward(self, *args):
        x = args[0]
        for i, sg in enumerate(self.subgraphs):
            if self.cuda_graphs[i]:
                # Replay CUDA graph
                x = self.cuda_graphs[i].replay(x)
            else:
                # Eager execution (e.g., attention)
                x = sg(x)
        return x
```

**Benefits**:
- **Smaller memory**: Each CUDA graph is small
- **Flexibility**: Non-compiled parts stay dynamic
- **Overhead**: ~1-2 μs per replay (negligible)

## Configuration System

**File**: `vllm/config/compilation.py`

```python
@dataclass
class CompilationConfig:
    mode: CompilationMode = CompilationMode.FULL
    backend: str = "inductor"
    
    # Piecewise compilation
    splitting_ops: List[str] = field(default_factory=lambda: [
        "vllm::unified_attention_with_output"
    ])
    
    # Dynamic shapes
    dynamic_shapes_config: DynamicShapesConfig = field(
        default_factory=lambda: DynamicShapesConfig(
            type=DynamicShapesType.BACKED
        )
    )
    
    # CUDA graphs
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE
    cudagraph_capture_sizes: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512
    ])
    
    # Optional: compile specific sizes only
    compile_sizes: Optional[List[int]] = None

@dataclass
class DynamicShapesConfig:
    type: DynamicShapesType = DynamicShapesType.BACKED
    
    # BACKED: Default, unsafe guard dropping
    # UNBACKED: Safe, no guards added
    # BACKED_SIZE_OBLIVIOUS: Experimental middle ground
```

## CLI Integration

```bash
# Enable compilation
vllm serve model

# Configure dynamic shapes
vllm serve model -cc.dynamic_shapes_config.type=unbacked

# Disable CUDA graphs
vllm serve model -cc.cudagraph_mode=NONE

# Compile specific batch sizes only
vllm serve model -cc.compile_sizes='[1,8,16]'

# Use eager backend (Dynamo only, no Inductor)
vllm serve model -cc.backend=eager

# Disable compilation entirely
vllm serve model --enforce-eager
```

## Key Takeaways

1. **Custom Compiler**: NOT just torch.compile - heavily customized for LLM inference
2. **Guard Dropping**: Core innovation - unsafe but effective for batch-agnostic ops
3. **Piecewise Strategy**: Split at attention, compile rest - best of both worlds
4. **LLM-Specific Passes**: 10+ custom optimizations beyond PyTorch defaults
5. **Production Focus**: Caching, configuration, debugging built-in
6. **Multi-GPU Optimized**: Heavy focus on tensor/sequence parallelism
7. **Additive Design**: Works with standard PyTorch - no core modifications needed
