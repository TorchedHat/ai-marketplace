# Quick Reference

Fast lookup for common commands, patterns, and locations.

## Debug Commands

### Enable Logging

```python
import torch
import logging

# Basic Dynamo logging
torch._logging.set_logs(dynamo=logging.INFO)

# Include bytecode
torch._logging.set_logs(dynamo=logging.INFO, bytecode=True)

# Verbose mode
torch._dynamo.config.verbose = True
```

### Environment Variables

```bash
# Enable debug mode
TORCH_COMPILE_DEBUG=1

# Verbose output
TORCHDYNAMO_VERBOSE=1

# Report guard failures
TORCHDYNAMO_REPORT_GUARD_FAILURES=1

# Print graph breaks
TORCH_LOGS="+graph_breaks"
```

### Reset Compilation

```python
# Clear compilation cache
torch._dynamo.reset()

# Reset compiled autograd
from torch._dynamo import compiled_autograd
compiled_autograd.reset()
```

## Backend Options

```python
# Eager - just trace, don't compile (for debugging)
torch.compile(backend="eager")

# Inductor - default, full optimization
torch.compile(backend="inductor")

# AOT Eager - AOTAutograd without Inductor
torch.compile(backend="aot_eager")

# Custom backend
@torch._dynamo.register_backend
def my_backend(gm, example_inputs):
    return gm.forward

torch.compile(backend="my_backend")
```

## Running Tests

### Single Test

```bash
# From pytorch directory
python test/dynamo/test_misc.py TestClass.test_method -v
```

### Test File

```bash
python test/dynamo/test_misc.py -v
```

### All Dynamo Tests

```bash
pytest test/dynamo/ -v
```

### With Logging

```bash
PYTORCH_TEST_WITH_DYNAMO=1 python test/dynamo/test_misc.py -v -s
```

## Key File Locations

### Core Files

| Component | Location |
|-----------|----------|
| Frame interception | `torch/_dynamo/eval_frame.py` |
| Convert decision | `torch/_dynamo/convert_frame.py` |
| Symbolic execution | `torch/_dynamo/symbolic_convert.py` |
| Graph building | `torch/_dynamo/output_graph.py` |
| Guards | `torch/_dynamo/guards.py` |
| Utils | `torch/_dynamo/utils.py` |

### Variables

| Type | Location |
|------|----------|
| Base class | `torch/_dynamo/variables/base.py` |
| Tensors | `torch/_dynamo/variables/tensor.py` |
| Lists/Tuples | `torch/_dynamo/variables/lists.py` |
| Dicts | `torch/_dynamo/variables/dicts.py` |
| Functions | `torch/_dynamo/variables/functions.py` |
| User-defined | `torch/_dynamo/variables/user_defined.py` |
| Builder | `torch/_dynamo/variables/builder.py` |

### Polyfills

| Module | Location |
|--------|----------|
| PyTree | `torch/_dynamo/polyfills/pytree.py` |

### Tests

| Category | Location |
|----------|----------|
| Main tests | `test/dynamo/test_*.py` |
| Inductor tests | `test/inductor/test_*.py` |

## Common Imports

```python
# Core Dynamo
import torch._dynamo
from torch._dynamo import variables
from torch._dynamo.variables import VariableTracker
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.output_graph import OutputGraph

# Guards
from torch._dynamo.guards import install_guard, GuardBuilder

# Utils
from torch._dynamo.utils import istype, is_namedtuple
from torch._dynamo.exc import unimplemented

# Bytecode
from torch._dynamo.bytecode_transformation import (
    create_instruction,
    create_call_function,
    create_load_const,
)

# Testing
from torch._dynamo.test_case import TestCase
```

## VariableTracker Quick Methods

```python
class MyVariable(VariableTracker):
    # Required for reconstruction
    def reconstruct(self, codegen):
        """Generate bytecode to recreate this"""

    # For function calls
    def call_function(self, tx, args, kwargs):
        """Handle fn(...) calls"""

    # For method calls
    def call_method(self, tx, name, args, kwargs):
        """Handle obj.method(...) calls"""

    # For attribute access
    def var_getattr(self, tx, name):
        """Handle obj.attr access"""

    # For subscript access
    def call_getitem(self, tx, key):
        """Handle obj[key] access"""

    # Python constant conversion
    def as_python_constant(self):
        """Convert to Python constant if possible"""

    # Type information
    def python_type(self):
        """Return the Python type this represents"""

    # PyTree support
    def call_tree_map_branch(self, tx, tree_map_fn, map_fn, rest, kwargs):
        """Handle pytree.tree_map for this type"""
```

## Common Patterns Cheat Sheet

### Add Opcode Handler

```python
# In symbolic_convert.py
def MY_OPCODE(self, inst):
    arg = self.stack.pop()
    result = arg.call_method(self, "operation", [], {})
    self.push(result)
```

### Create VariableTracker

```python
# In variables/my_type.py
class MyTypeVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def python_type(self):
        return MyType

    def reconstruct(self, codegen):
        codegen.load_import_from("module", "MyType")
        # ... load args ...
        codegen.extend_output(create_call_function(nargs, False))
```

### Register in Builder

```python
# In variables/builder.py __call__
if isinstance(value, MyType):
    return MyTypeVariable(value, source=self.source)
```

### Install Guard

```python
if self.source:
    install_guard(self.source.make_guard(GuardBuilder.TYPE_MATCH))
```

## Debugging Checklist

- [ ] Create minimal repro
- [ ] Enable logging (`torch._logging.set_logs(...)`)
- [ ] Check bytecode capture
- [ ] Check object → variable conversion
- [ ] Check VariableTracker implementation
- [ ] Check bytecode reconstruction
- [ ] Write test case
- [ ] Run related tests

## Configuration Options

```python
# Suppress errors (for debugging)
torch._dynamo.config.suppress_errors = False

# Verbose mode
torch._dynamo.config.verbose = True

# Cache size
torch._dynamo.config.cache_size_limit = 64

# Dynamic shapes
torch._dynamo.config.assume_static_by_default = False

# Capture scalar outputs
torch._dynamo.config.capture_scalar_outputs = True
```

## Graph Break Handling

```python
# Allow graph breaks
@torch.compile
def fn(x):
    # This might cause a graph break
    if x.shape[0] > 10:
        print("Large tensor")  # Print causes graph break
    return x * 2

# Force fullgraph (error on graph break)
@torch.compile(fullgraph=True)
def fn(x):
    return x * 2  # Must not have any graph breaks
```

## PyTree Quick Reference

### Check if Type is PyTree Node

```python
import torch.utils._pytree as pytree

# Explicit registration
is_registered = MyType in pytree.SUPPORTED_NODES

# Implicit registration
is_registered = (
    pytree.is_namedtuple_class(MyType) or
    pytree.is_structseq_class(MyType)
)
```

### Register Custom Type

```python
import torch.utils._pytree as pytree

def flatten_fn(obj):
    return ([obj.field1, obj.field2], None)

def unflatten_fn(values, context):
    return MyType(*values)

pytree.register_pytree_node(MyType, flatten_fn, unflatten_fn)
```

### Fast-Path Check

```python
# In UserDefinedObjectVariable.call_tree_map_branch
is_registered = (
    self.value_type in pytree.SUPPORTED_NODES
    or pytree.is_namedtuple_class(self.value_type)
    or pytree.is_structseq_class(self.value_type)
)
```

## Common Errors and Quick Fixes

| Error | Quick Fix |
|-------|-----------|
| `unhashable type: VariableTracker` | Add to `is_hashable()` in `dicts.py` |
| Wrong tree_map output | Add implicit checks in `call_tree_map_branch()` |
| Graph break on simple op | Add opcode handler or method |
| Frequent recompilation | Check guards, reduce dynamic behavior |
| Slow compilation | Add fast-path or reduce complexity |
| AttributeError on VariableTracker | Implement `var_getattr()` or `call_method()` |

## Useful oneliners

```python
# Check if compilation happened
torch._dynamo.utils.counters["frames"]["ok"] > 0

# See compilation times
torch._dynamo.utils.compile_times()

# Export graph
from torch._dynamo.backends.debugging import debug_backend
torch.compile(backend=debug_backend)

# Check cache hits
torch._dynamo.utils.counters

# Disable compilation for function
fn = torch._dynamo.disable(fn)
```

## Getting Help

- **Docs**: https://pytorch.org/docs/stable/torch.compiler.html
- **GitHub**: https://github.com/pytorch/pytorch
- **Forums**: https://discuss.pytorch.org/c/torch-compile
- **Skill files**: [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md), [ARCHITECTURE.md](ARCHITECTURE.md)
