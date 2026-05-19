# Dynamo Architecture Deep Dive

Comprehensive guide to PyTorch Dynamo's internal architecture.

## Conversion Pipeline

```
Python Function
    ↓
[1] Frame Interception (eval_frame.py)
    ↓
[2] Convert Decision (convert_frame.py) → Tracing vs Fallback
    ↓
[3] Symbolic Execution (symbolic_convert.py - InstructionTranslator)
        - executes bytecode symbolically
        - maintains symbolic stack + environment
        - VariableTrackers are created and manipulated (variables/* - VariableTracker subclasses)
        - emits FX graph nodes
        - emits guards here (inline during execution)
    ↓
[4] Graph Assembly (output_graph.py - OutputGraph)
    ↓
FX GraphModule + Guards → Backend (AOTAutograd/Inductor)
```

## Core Components

### 1. Frame Interception

**File**: `torch/_dynamo/eval_frame.py`

Installs a custom frame evaluation hook that intercepts Python bytecode execution.

**Key functions**:
- `set_eval_frame()` - Installs the hook
- `_custom_eval_frame()` - The hook that decides whether to compile

**What it does**:
- Intercepts function calls at the CPython frame level
- Decides whether to compile or execute normally
- Manages cache of compiled functions

### 2. Convert Decision

**File**: `torch/_dynamo/convert_frame.py`

Decides whether to trace a frame or fall back to normal execution.

**Key function**: `convert_frame_assert()`

**Decisions**:
- Should this frame be traced?
- Has it been compiled before (cache lookup)?
- Should we skip due to config/guards?

### 3. Symbolic Execution

**File**: `torch/_dynamo/symbolic_convert.py`

**Class**: `InstructionTranslator`

The heart of Dynamo - symbolically executes Python bytecode.

**Key methods**:
- `run()` - Main execution loop
- `dispatch_table` - Maps opcodes to handler methods
- `step()` - Executes one bytecode instruction

**How it works**:
```python
for inst in instructions:
    handler = self.dispatch_table[inst.opcode]
    handler(self, inst)
```

**Each opcode has a handler**:
- `LOAD_FAST(self, inst)` - Load local variable
- `CALL_FUNCTION(self, inst)` - Call a function
- `BINARY_ADD(self, inst)` - Add two values
- etc.

### 4. Variable Tracking

**Directory**: `torch/_dynamo/variables/`

**Base**: `VariableTracker` in `base.py`

**Subclasses** (key ones):
- `TensorVariable` - Represents a tensor
- `ConstantVariable` - Represents a constant (int, str, etc.)
- `ListVariable`, `TupleVariable` - Collections
- `ConstDictVariable` - Dictionaries
- `UserDefinedObjectVariable` - Custom classes
- `NamedTupleVariable` - Named tuples
- `UserFunctionVariable` - Python functions
- `BuiltinVariable` - Builtin functions

**Key methods**:
- `reconstruct(codegen)` - Generate bytecode to recreate this object
- `call_method(tx, name, args, kwargs)` - Handle method calls
- `call_function(tx, args, kwargs)` - Handle function calls
- `var_getattr(tx, name)` - Handle attribute access

**Purpose**: Represent Python objects during symbolic execution without actually executing them.

### 5. Graph Assembly

**File**: `torch/_dynamo/output_graph.py`

**Class**: `OutputGraph`

Builds the FX GraphModule from symbolic execution.

**Key methods**:
- `create_node()` - Add a node to the graph
- `create_proxy()` - Create a proxy for a tensor operation
- `compile_and_call_fx_graph()` - Finalize and call backend

**What it produces**:
- FX GraphModule (computational graph)
- Guard functions (validity checks)
- List of graph breaks (if any)

### 6. Guard System

**Files**: `guards.py`, `guard_export.py`, `guard_failures.py`

Ensures compiled code only runs when assumptions hold.

**Types of guards**:
- Shape guards: `tensor.shape[0] == 10`
- Type guards: `type(x) == int`
- Value guards: `config.mode == "train"`
- Source guards: Checks on input sources

**Guard failures**:
- Trigger recompilation
- Logged for debugging
- Can cause performance issues if too frequent

## Key Classes in Detail

### InstructionTranslator

**Location**: `symbolic_convert.py`

**Attributes**:
- `instructions` - Bytecode instructions to execute
- `stack` - Python stack (list of VariableTrackers)
- `locals` - Local variables (dict of VariableTrackers)
- `output` - OutputGraph being built
- `f_globals` - Global variables
- `f_builtins` - Builtin functions

**Execution model**:
1. Start with function arguments as VariableTrackers
2. Execute bytecode instructions symbolically
3. Each instruction manipulates stack/locals with VariableTrackers
4. Graph operations recorded in OutputGraph
5. Return value becomes graph output

### VariableTracker

**Location**: `variables/base.py`

**Attributes**:
- `source` - Where this variable came from (for guards)
- `mutation_type` - How this can be mutated

**Subclass responsibilities**:
- Implement `reconstruct()` to generate bytecode
- Override `call_method()` for method calls
- Override `call_function()` for function calls
- Override `var_getattr()` for attribute access

**Example flow**:
```python
# User code: result = tensor.sum()
# Becomes:
tensor_var = TensorVariable(...)
result_var = tensor_var.call_method(tx, "sum", [], {})
```

### OutputGraph

**Location**: `output_graph.py`

**Attributes**:
- `graph` - FX Graph being built
- `guards` - List of guard functions
- `side_effects` - Tracked side effects
- `current_tracer` - FX tracer instance

**Graph building**:
```python
# User code: y = x + 1
# Becomes graph nodes:
x_node = graph.placeholder('x')
one_node = graph.call_function(operator.add, (x_node, 1))
graph.output(one_node)
```

## VariableBuilder

**File**: `variables/builder.py`

Converts Python objects to VariableTrackers.

**Key method**: `__call__(value)` - Main entry point

**Logic**:
```python
if isinstance(value, torch.Tensor):
    return TensorVariable(...)
elif isinstance(value, (int, float, str)):
    return ConstantVariable(...)
elif isinstance(value, dict):
    return ConstDictVariable(...)
elif is_namedtuple(value):
    return NamedTupleVariable(...)
# etc.
```

## Development Rules

### 1. Place Opcode Logic Correctly

**Opcode handlers belong in**:
- `symbolic_convert.py` for core opcodes
- Specific `VariableTracker` classes for type-specific operations

**Don't**:
- Scatter opcode logic across multiple files
- Put bytecode handling in unrelated files

### 2. Respect Guard Semantics

**Always**:
- Generate guards for assumptions
- Use proper guard builders
- Document what is being guarded

**Never**:
- Bypass the guard system
- Make unchecked assumptions
- Assume constant values without guards

### 3. Use Existing Abstractions

**Do**:
- Use existing VariableTracker classes
- Leverage utility functions in utils.py
- Follow established patterns

**Don't**:
- Create new Dynamo-level APIs
- Reinvent existing functionality
- Introduce inconsistent abstractions

### 4. Maintain VariableTracker Consistency

**All VariableTracker subclasses should**:
- Implement `reconstruct()` correctly
- Handle common operations uniformly
- Properly track mutations
- Generate appropriate guards

### 5. Fit the Architecture

**New features should**:
- Align with the existing pipeline
- Use standard extension points
- Not create parallel systems

**Example**: Don't create a separate tracing system - extend InstructionTranslator.

## Extension Points

### Adding New Opcode Support

**Location**: `symbolic_convert.py`

```python
class InstructionTranslator:
    def MY_NEW_OPCODE(self, inst):
        # Handle the opcode
        value = self.stack.pop()
        result = value.call_method(self, "my_operation", [], {})
        self.push(result)
```

### Creating New VariableTracker

**Location**: `variables/my_type.py`

```python
from .base import VariableTracker

class MyTypeVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen):
        # Generate bytecode to recreate this
        codegen.load_import_from("mymodule", "MyType")
        codegen.extend_output(create_load_const(self.value))
        codegen.extend_output(create_call_function(1, False))

    def call_method(self, tx, name, args, kwargs):
        if name == "my_method":
            # Handle specific method
            return ConstantVariable(42)
        return super().call_method(tx, name, args, kwargs)
```

**Register in VariableBuilder**:

```python
# In variables/builder.py
def __call__(self, value):
    if isinstance(value, MyType):
        return MyTypeVariable(value)
    # ... rest of logic
```

### Adding Backend Integration

**Location**: Backend-specific, but interface in `output_graph.py`

```python
def compile_and_call_fx_graph(self, tx, rv, ...):
    # rv = return value
    # Generate graph
    graph = self.graph
    # Call backend
    compiled = backend_compile(graph, ...)
    # Return compiled function
    return compiled
```

## Common Patterns

### Pattern: Lazy Evaluation

VariableTrackers don't execute - they record operations.

```python
# User code: result = x + y
# Symbolic execution:
x_var = TensorVariable(...)
y_var = TensorVariable(...)
result_var = x_var.call_method(tx, "__add__", [y_var], {})
# No actual addition happens - just recorded in graph
```

### Pattern: Proxy Objects

FX uses proxy objects to build graphs.

```python
# Proxy wraps a graph node
proxy = torch.fx.Proxy(node)
# Operations on proxy create new nodes
result_proxy = proxy + 1  # Creates add node
```

### Pattern: Guard Generation

Always guard assumptions.

```python
# User code uses tensor shape
if tensor.shape[0] > 10:
    ...

# Symbolic execution generates:
# 1. VariableTracker for shape
# 2. Guard: tensor.shape[0] == <actual_value>
# 3. Specialized code for that shape
```

## Backend Integration

### AOTAutograd

**Location**: `torch/_functorch/aot_autograd.py`

Ahead-of-time autograd - generates forward and backward graphs.

**What it does**:
- Takes FX graph
- Generates separate forward/backward graphs
- Handles autograd logic at compile time

### Inductor

**Location**: `torch/_inductor/`

Default backend - generates optimized code.

**What it does**:
- Takes FX graph
- Generates Triton/C++ code
- Optimizes memory layout, fusion, etc.

### Custom Backends

Can register custom backends:

```python
@register_backend
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    # gm = FX GraphModule
    # Return callable that runs the graph
    return gm.forward
```

## Performance Considerations

### Graph Breaks

**What**: Points where Dynamo stops tracing and falls back to eager.

**Causes**:
- Unsupported Python features
- Data-dependent control flow
- Unimplemented operations

**Impact**: Performance degradation (missed optimizations)

**Debug**: `TORCH_COMPILE_DEBUG=1`

### Recompilation

**What**: Recompiling due to guard failures.

**Causes**:
- Changing tensor shapes
- Changing types
- Dynamic behavior

**Impact**: Compilation overhead

**Mitigation**: `torch._dynamo.config.cache_size_limit`

## Testing

### Test Structure

**Location**: `test/dynamo/test_*.py`

**Patterns**:
```python
class TestMyFeature(torch._dynamo.test_case.TestCase):
    def test_case(self):
        def fn(x):
            return x + 1

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)
```

### Test Backends

- `eager` - Just traces, doesn't optimize (for testing tracing)
- `aot_eager` - AOTAutograd without optimization
- `inductor` - Full compilation

## References

- Core pipeline: `eval_frame.py`, `convert_frame.py`, `symbolic_convert.py`
- Variables: `variables/` directory
- Graph building: `output_graph.py`
- Guards: `guards.py`
- Utils: `utils.py`
