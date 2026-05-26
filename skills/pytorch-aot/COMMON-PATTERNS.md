# Common Development Patterns

Implementation patterns and code examples for developing in `torch/_functorch`. Focus on practical "how to" guidance.

## When to Use This Skill vs compile-trace-aot

Choose the right skill for your task:

| Scenario | Use This Skill (pytorch-aot) | Use compile-trace-aot |
|----------|------------------------------|----------------------|
| **Implementing** mutation support | ✅ Yes - see "Adding mutation support" | ❌ No |
| **Debugging** functionalization error | ❌ No | ✅ Yes - has TORCH_LOGS workflows |
| **Adding** post-grad pass | ✅ Yes - see "Implementing post-grad pass" | ❌ No |
| **Debugging** gradient mismatch | ❌ No | ✅ Yes - has comparison workflows |
| **Understanding** how partitioning works | ✅ Yes - see ARCHITECTURE.md | ❌ No |
| **Debugging** saved activations issue | ❌ No | ✅ Yes - has tracing workflows |
| **Implementing** vmap support for custom op | ✅ Yes - see "Custom batching rule" | ❌ No |
| **Interpreting** TORCH_LOGS output | ❌ No | ✅ Yes - has file interpretation guide |

**Summary**:
- **pytorch-aot** (this skill) = "How do I implement/build X?"
- **compile-trace-aot** = "Why is my code failing? How do I debug it?"

**Cross-reference**: For all debugging, tracing, and TORCH_LOGS workflows, use `compile-trace-aot` skill.

## Pattern: Adding Mutation Support

**When**: Adding support for a new in-place op or fixing functionalization bug for existing op.

**Where**: `torch/_functorch/_aot_autograd/functional_utils.py`

**Concept**: FunctionalTensor wraps tensors and intercepts operations. For mutations, record them as metadata and emit functional equivalent.

**Steps**:

1. **Identify the mutation op** - e.g., `aten.mul_.Tensor`

2. **Add handler in FunctionalTensor** (if needed):
```python
# In functional_utils.py, FunctionalTensor class

def mul_(self, other):
    # Record mutation
    self._metadata.mutations.append(('mul_', other))

    # Emit functional version
    new_data = self._data * other

    # Update wrapped data
    self._data = new_data

    # Return self for chaining (in-place ops return self)
    return self
```

3. **Handle in functionalization pass** (if dispatch needed):
```python
# In functional_utils.py, functionalization conversion

@torch.library.impl("aten::mul_", "FunctionalTensor")
def mul__functional(self, other):
    # Decompose to functional op
    result = torch.mul(self, other)

    # Record mutation metadata
    ctx.mark_mutation(self, result)

    return result
```

4. **Test**:
```python
# In test/functorch/test_aotdispatch.py

def test_functionalize_mul_inplace(self):
    def fn(x):
        x.mul_(2)
        return x

    inp = torch.randn(3, 4)

    # Compare eager vs functionalized
    eager_out = fn(inp.clone())

    func_fn = make_functional(fn)
    func_out = func_fn(inp.clone())

    self.assertEqual(eager_out, func_out)
```

**Steering**: Query `query_api_docs(query="FunctionalTensor")` for detailed class structure.

## Pattern: Implementing Post-Grad Pass

**When**: Adding FX graph optimization that runs after AOT partitioning.

**Where**: `torch/_inductor/fx_passes/post_grad.py`

**Concept**: Post-grad passes run on forward and backward graphs after partitioning. Use FX pattern matching to find and rewrite patterns.

**Example - Fuse matmul + add**:

```python
# In torch/_inductor/fx_passes/post_grad.py

def fuse_matmul_add(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Fuse matmul + add into addmm.

    Pattern: z = x @ y + bias
    Replacement: z = torch.addmm(bias, x, y)
    """
    graph = gm.graph
    modified = False

    for node in graph.nodes:
        # Match pattern: add(matmul(x, y), bias)
        if node.target == torch.ops.aten.add.Tensor:
            matmul_node = node.args[0]
            bias_node = node.args[1]

            if (isinstance(matmul_node, torch.fx.Node) and
                matmul_node.target == torch.ops.aten.matmul.default):

                x = matmul_node.args[0]
                y = matmul_node.args[1]

                # Create addmm node
                with graph.inserting_before(node):
                    addmm_node = graph.call_function(
                        torch.ops.aten.addmm.default,
                        args=(bias_node, x, y)
                    )

                # Replace uses
                node.replace_all_uses_with(addmm_node)
                graph.erase_node(node)

                modified = True

    if modified:
        gm.recompile()

    return gm
```

**Integration**: Register pass to run automatically:

```python
# In post_grad.py or fx_passes/__init__.py

POST_GRAD_PASSES = [
    fuse_matmul_add,
    # ... other passes
]

def apply_post_grad_passes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for pass_fn in POST_GRAD_PASSES:
        gm = pass_fn(gm)
    return gm
```

**Test**:
```python
# In test/inductor/test_fx_passes.py

def test_fuse_matmul_add(self):
    def fn(x, y, bias):
        return x @ y + bias

    x = torch.randn(10, 20)
    y = torch.randn(20, 30)
    bias = torch.randn(30)

    compiled = torch.compile(fn)

    # Verify output correctness
    eager_out = fn(x, y, bias)
    compiled_out = compiled(x, y, bias)
    self.assertEqual(eager_out, compiled_out)

    # Verify fusion occurred (check graph contains addmm)
    # Use TORCH_LOGS="post_grad_graphs" to inspect
```

**Steering**: Query `query_api_docs(query="post_grad_passes")` for existing passes.

## Pattern: Using vmap for Batched Gradients

**When**: Computing per-sample gradients or per-sample losses.

**Concept**: vmap vectorizes a function over a batch dimension. Combine with grad to get per-sample gradients.

**Basic vmap usage**:
```python
def fn(x):
    return torch.sin(x).sum()

# Batch of inputs: [B, N]
batch_inputs = torch.randn(100, 10)

# Vectorize fn over batch dim
batched_fn = torch.vmap(fn)
batch_outputs = batched_fn(batch_inputs)  # [B] - one output per sample
```

**vmap + grad for per-sample gradients**:
```python
def loss_fn(params, input, target):
    """Loss for single sample."""
    pred = model(params, input)
    return F.mse_loss(pred, target)

# Get gradient function
grad_fn = torch.func.grad(loss_fn, argnums=0)  # Gradient w.r.t. params

# Vectorize over batch
per_sample_grad_fn = torch.func.vmap(
    grad_fn,
    in_dims=(None, 0, 0)  # params same for all, batch over inputs/targets
)

# Compute per-sample gradients
batch_inputs = torch.randn(32, 10)
batch_targets = torch.randn(32, 1)
params = {'weight': torch.randn(10, 1), 'bias': torch.randn(1)}

per_sample_grads = per_sample_grad_fn(params, batch_inputs, batch_targets)
# per_sample_grads['weight']: [32, 10, 1] - gradient for each sample
# per_sample_grads['bias']: [32, 1] - gradient for each sample
```

**in_dims explained**:
- `in_dims=(None, 0, 0)`: Don't batch over params (same for all), batch over dim 0 of inputs and targets
- `in_dims=0`: Batch over dim 0 of all arguments
- `in_dims=(0, 1)`: Batch over dim 0 of first arg, dim 1 of second arg

**out_dims**: Controls where batch dim appears in output (default 0).

**Steering**: Query `query_api_docs(query="vmap")` for detailed in_dims/out_dims semantics.

## Pattern: Making Module Functional

**When**: Need to call nn.Module with different parameters (meta-learning, per-sample analysis).

**make_functional vs functional_call**:

| Use Case | Use make_functional | Use functional_call |
|----------|---------------------|-------------------|
| Extract params once, call many times | ✅ Yes | ❌ No (re-extracts each time) |
| Temporary param swap | ❌ No (modifies module) | ✅ Yes (doesn't modify) |
| vmap over parameters | ✅ Yes | ✅ Yes (both work) |
| Need buffers separate | Use make_functional_with_buffers | Pass buffers in state_dict |

**make_functional pattern**:
```python
import torch.func

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Convert to functional
func_model, params = torch.func.make_functional(model)

# Call with explicit params
output = func_model(params, input)

# Compute gradient w.r.t. params
grad_fn = torch.func.grad(lambda p, x: func_model(p, x).sum())
grads = grad_fn(params, input)
```

**functional_call pattern**:
```python
from torch.func import functional_call

model = nn.Linear(10, 1)

# Create alternate parameters
alt_params = {
    'weight': torch.randn(1, 10),
    'bias': torch.randn(1)
}

# Call model with alternate params (doesn't modify model.weight/bias)
output = functional_call(model, alt_params, input)
```

**Per-sample gradients with functional module**:
```python
def compute_loss(params, input, target):
    pred = func_model(params, input)
    return F.mse_loss(pred, target)

# Gradient function for single sample
grad_fn = torch.func.grad(compute_loss, argnums=0)

# Vectorize over batch
per_sample_grads = torch.func.vmap(
    grad_fn,
    in_dims=(None, 0, 0)  # Same params, batch over inputs/targets
)(params, batch_inputs, batch_targets)
```

**Steering**: Query `query_api_docs(query="make_functional")` for functional module details.

## Pattern: Extending Functionalization

**When**: Custom ops that aren't automatically functionalized, or custom view operations.

**For standard ops** - Add decomposition:
```python
# In torch/_decomp/decompositions.py or custom decomp file

@torch.library.custom_op("mylib::custom_inplace_op", mutates_args={"input"})
def custom_inplace_op(input: torch.Tensor, value: float) -> None:
    """Custom in-place op that isn't automatically functionalized."""
    # C++ implementation
    pass

# Register functional decomposition
@torch.library.register_fake("mylib::custom_inplace_op")
def custom_inplace_op_fake(input: torch.Tensor, value: float) -> None:
    # Fake implementation for tracing
    pass

@torch.library.impl_abstract("mylib::custom_inplace_op")
def custom_inplace_op_abstract(input: torch.Tensor, value: float) -> None:
    # Decompose to functional ops
    result = input + value
    input.copy_(result)  # This will be functionalized automatically
```

**For custom views** - Register ViewMeta:
```python
# In functional_utils.py or custom file

class CustomViewMeta:
    def __init__(self, base, custom_param):
        self.base = base
        self.custom_param = custom_param

    def replay(self, new_base):
        """Reconstruct view from new base tensor."""
        return custom_view_op(new_base, self.custom_param)

# In functionalization pass
def functionalize_custom_view(base_tensor, custom_param):
    # Create functional version
    result = custom_view_op(base_tensor, custom_param)

    # Record view metadata
    view_meta = CustomViewMeta(base_tensor, custom_param)
    register_view_meta(result, view_meta)

    return result
```

**Testing**:
```python
def test_functionalize_custom_op(self):
    def fn(x):
        torch.ops.mylib.custom_inplace_op(x, 2.0)
        return x

    # Test that functionalization preserves semantics
    inp = torch.randn(3, 4, requires_grad=True)

    eager_out = fn(inp.clone())

    # Run through functionalization
    from torch._functorch._aot_autograd.functional_utils import to_fun
    func_inp = to_fun(inp.clone())
    func_out = fn(func_inp)
    func_out = from_fun(func_out)

    self.assertEqual(eager_out, func_out)
```

**Steering**: Query `query_api_docs(query="functionalization")` for detailed API.

## Pattern: Custom Partitioning Policy

**When**: Need to customize which activations are saved vs recomputed.

**Basic config** - Use min-cut:
```python
import torch._functorch.config as functorch_config

# Enable min-cut recomputation
functorch_config.use_min_cut_rematerialization = True

# Now torch.compile will use min-cut partitioning
model = torch.compile(model)
```

**Advanced** - Custom partitioning function:
```python
from torch._functorch.partitioners import default_partition

def custom_partition(joint_graph, fw_outputs):
    """
    Custom partitioning logic.

    Args:
        joint_graph: fx.GraphModule with forward+backward
        fw_outputs: Nodes that are forward outputs

    Returns:
        (forward_graph, backward_graph)
    """
    # Start with default partition
    fw_graph, bw_graph = default_partition(joint_graph, fw_outputs)

    # Custom logic: Never recompute expensive ops
    for node in fw_graph.graph.nodes:
        if is_expensive_op(node):
            # Mark as must-save (implementation-dependent)
            mark_must_save(node)

    return fw_graph, bw_graph

# Use custom partitioner
from torch._functorch.aot_autograd import aot_module_simplified

compiled_model = aot_module_simplified(
    model,
    fw_compiler=inductor_compile,
    partition_fn=custom_partition
)
```

**Integration with activation checkpointing**:
```python
from torch.utils.checkpoint import checkpoint_sequential

# Mark segments for checkpointing
model_with_checkpointing = checkpoint_sequential(model, segments=4)

# AOT will respect checkpointing decisions during partitioning
compiled = torch.compile(model_with_checkpointing)
```

**Config options**:
```python
# In torch/_functorch/config.py
use_min_cut_rematerialization = True  # Enable min-cut
use_dynamic_shapes = True              # Allow dynamic batch size
assume_static_by_default = True        # Assume static shapes unless marked dynamic
```

**Steering**: Query `query_api_docs(query="partitioners")` for partitioning algorithms.

## Pattern: Custom Batching Rule

**When**: Adding vmap support for custom op that doesn't have automatic batching rule.

**Register batching rule**:
```python
import torch.library

# Define custom op
@torch.library.custom_op("mylib::custom_op", mutates_args=())
def custom_op(x: torch.Tensor, param: int) -> torch.Tensor:
    # Implementation
    pass

# Register vmap batching rule
@torch.library.register_vmap("mylib::custom_op")
def custom_op_batch_rule(
    x_bdim,      # (batched_tensor, batch_dim_index) or (tensor, None)
    param,       # Non-tensor args stay same
):
    """
    Batching rule for custom_op.

    Args:
        x_bdim: Tuple of (tensor, batch_dim) where batch_dim is None if unbatched
        param: Regular argument (same for all batch elements)

    Returns:
        (output, output_batch_dim)
    """
    x, x_batch_dim = x_bdim

    if x_batch_dim is None:
        # Input not batched, output not batched
        return custom_op(x, param), None

    # Input batched on dim x_batch_dim
    # Apply op along batch dim
    # Example: if op is element-wise, batch dim is preserved
    output = custom_op(x, param)

    return output, x_batch_dim  # Output batched on same dim
```

**For ops with multiple inputs**:
```python
@torch.library.register_vmap("mylib::binary_op")
def binary_op_batch_rule(x_bdim, y_bdim):
    x, x_batch_dim = x_bdim
    y, y_batch_dim = y_bdim

    # Case 1: Both batched
    if x_batch_dim is not None and y_batch_dim is not None:
        # Move batch dims to dim 0
        if x_batch_dim != 0:
            x = x.movedim(x_batch_dim, 0)
        if y_batch_dim != 0:
            y = y.movedim(y_batch_dim, 0)

        output = binary_op(x, y)
        return output, 0  # Output batched on dim 0

    # Case 2: Only x batched
    elif x_batch_dim is not None:
        # Broadcast y to match x's batch
        output = binary_op(x, y.unsqueeze(x_batch_dim))
        return output, x_batch_dim

    # Case 3: Only y batched
    elif y_batch_dim is not None:
        output = binary_op(x.unsqueeze(y_batch_dim), y)
        return output, y_batch_dim

    # Case 4: Neither batched
    return binary_op(x, y), None
```

**Steering**: Query `query_api_docs(query="batching rules")` for existing examples.

## Quick File Reference

**Where to add code**:

| Task | File | What to Modify |
|------|------|----------------|
| Mutation support | `torch/_functorch/_aot_autograd/functional_utils.py` | FunctionalTensor methods or decompositions |
| Post-grad pass | `torch/_inductor/fx_passes/post_grad.py` | Add pass function, register in POST_GRAD_PASSES |
| Batching rule | Op definition file or `torch/_functorch/vmap.py` | Register with `@torch.library.register_vmap` |
| Partitioning logic | `torch/_functorch/partitioners.py` | Modify partitioning functions |
| AOT config | `torch/_functorch/config.py` | Add config variable |
| AOT entry point | `torch/_functorch/aot_autograd.py` | Modify `aot_function`, `aot_module` |

**Where to add tests**:

| Test Type | File |
|-----------|------|
| AOT compilation | `test/functorch/test_aotdispatch.py` |
| vmap | `test/functorch/test_vmap.py` |
| Grad transforms | `test/functorch/test_ops.py` |
| Functionalization | `test/functorch/test_aotdispatch.py` |
| make_functional | `test/functorch/test_parsing.py` |

**Key entry points**:
- `torch.func.vmap` → `torch/_functorch/apis.py`
- `torch.func.grad` → `torch/_functorch/apis.py`
- `torch.compile` (AOT backend) → `torch/_dynamo/backends.py` → `torch/_functorch/aot_autograd.py`

---

**Cross-references**:
- **Debugging**: Use `compile-trace-aot` skill for TORCH_LOGS and debugging workflows
- **Architecture**: See `ARCHITECTURE.md` for detailed component explanations
- **API Details**: Query steering MCP for function signatures and detailed examples
