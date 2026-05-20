# Common Implementation Patterns

Recipes and patterns for implementing Dynamo features.

## Adding New Opcode Support

**When**: Python introduces a new bytecode opcode or existing one isn't supported.

**Location**: `torch/_dynamo/symbolic_convert.py`

### Pattern

```python
class InstructionTranslator:
    def MY_NEW_OPCODE(self, inst):
        """
        Handle MY_NEW_OPCODE bytecode instruction.

        Args:
            inst: Instruction object with opcode, arg, argval, etc.
        """
        # 1. Pop arguments from stack (if any)
        arg1 = self.stack.pop()
        arg2 = self.stack.pop()

        # 2. Perform operation symbolically
        result = arg1.call_method(self, "my_operation", [arg2], {})

        # 3. Push result back to stack
        self.push(result)
```

### Example: BINARY_ADD

```python
def BINARY_ADD(self, inst):
    """Implement a + b"""
    # Stack: [..., a, b]
    right = self.stack.pop()  # b
    left = self.stack.pop()   # a

    # Call __add__ method symbolically
    result = left.call_method(self, "__add__", [right], {})

    # Stack: [..., result]
    self.push(result)
```

### Tips

- **Stack order matters**: BINARY_ADD pops right operand first!
- **Don't execute**: Record operations, don't run them
- **Use existing methods**: Prefer `call_method()` over custom logic
- **Handle errors**: Use `unimplemented()` for unsupported cases

## Creating New VariableTracker

**When**: Need to represent a new Python type during symbolic execution.

**Location**: `torch/_dynamo/variables/my_type.py`

### Pattern

```python
from .base import VariableTracker
from ..bytecode_transformation import create_instruction

class MyTypeVariable(VariableTracker):
    """Represents MyType objects during symbolic execution."""

    _nonvar_fields = {"my_data", *VariableTracker._nonvar_fields}

    def __init__(self, my_data, **kwargs):
        super().__init__(**kwargs)
        self.my_data = my_data

    def as_python_constant(self):
        """Convert to Python constant if possible."""
        return MyType(self.my_data.as_python_constant())

    def python_type(self):
        """Return the Python type this represents."""
        return MyType

    def reconstruct(self, codegen):
        """Generate bytecode to recreate this object."""
        # Load the class
        codegen.load_import_from("mymodule", "MyType")

        # Load arguments
        codegen.append_output(create_load_const(self.my_data))

        # Call constructor
        codegen.extend_output(create_call_function(1, False))

    def call_method(self, tx, name, args, kwargs):
        """Handle method calls on this object."""
        if name == "my_method":
            # Implement method handling
            return ConstantVariable(42)

        # Fall back to base class
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        """Handle attribute access."""
        if name == "my_attr":
            return ConstantVariable(self.my_data)

        return super().var_getattr(tx, name)
```

### Register in VariableBuilder

**Location**: `torch/_dynamo/variables/builder.py`

```python
class VariableBuilder:
    def __call__(self, value):
        # ... existing logic ...

        # Add your type check (order matters!)
        if isinstance(value, MyType):
            from .my_type import MyTypeVariable
            return MyTypeVariable(
                my_data=self(value.data),  # Recursive for nested data
                source=self.source,
            )

        # ... rest of logic ...
```

### Tips

- **Order matters** in VariableBuilder: Check most specific types first
- **Recursive tracking**: For nested data, recursively call `self()`
- **Source tracking**: Pass `source` for guard generation
- **_nonvar_fields**: Add fields that shouldn't be treated as VariableTrackers
- **python_type()**: Critical for pytree and other type checks

## Implementing call_tree_map_branch

**When**: Adding pytree support for a custom container type.

**Location**: Your VariableTracker class

### Pattern

```python
def call_tree_map_branch(self, tx, tree_map_fn, map_fn, rest, tree_map_kwargs):
    """Handle tree_map for this type."""

    # 1. Validate structure matches across all trees
    other_containers = []
    for candidate in rest:
        if not isinstance(candidate, MyTypeVariable) or \
           len(candidate.items) != len(self.items):
            # Structure mismatch - fall back to tracing
            return self._tree_map_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs
            )
        other_containers.append(candidate)

    # 2. Recursively map over children
    new_items = []
    for i, item in enumerate(self.items):
        # Get corresponding items from other trees
        other_items = [other.items[i] for other in other_containers]

        # Recursive tree_map call
        mapped = item.call_tree_map(
            tx, tree_map_fn, map_fn, other_items, tree_map_kwargs
        )
        new_items.append(mapped)

    # 3. Reconstruct with same type
    return MyTypeVariable(new_items, ...)
```

### Tips

- **Validate structure first**: All trees must have same shape
- **Fall back gracefully**: Use `_tree_map_fallback()` when can't handle
- **Preserve exact types**: Return same type (OrderedDict stays OrderedDict)
- **Recursive calls**: Use `call_tree_map()` not direct recursion

## Handling User-Defined Classes

**When**: Need to support custom classes in compilation.

**Location**: `torch/_dynamo/variables/user_defined.py`

### Pattern: Register with PyTree

```python
# In user code
import torch.utils._pytree as pytree

@dataclass
class MyConfig:
    learning_rate: float
    batch_size: int

# Register as pytree node
pytree.register_dataclass(MyConfig)

# Or custom registration
def flatten_fn(obj):
    return ([obj.learning_rate, obj.batch_size], None)

def unflatten_fn(values, context):
    return MyConfig(*values)

pytree.register_pytree_node(
    MyConfig,
    flatten_fn,
    unflatten_fn,
)
```

### Pattern: Check Registration in Dynamo

```python
# In UserDefinedObjectVariable.call_tree_map_branch
is_registered = (
    self.value_type in pytree.SUPPORTED_NODES
    or pytree.is_namedtuple_class(self.value_type)
    or pytree.is_structseq_class(self.value_type)
    # Add custom check if needed
    or is_my_custom_type_registered(self.value_type)
)
```

## Adding Guard Support

**When**: Need to guard on a new property or assumption.

**Location**: `torch/_dynamo/guards.py`

### Pattern

```python
# In guards.py
class GuardBuilder:
    @staticmethod
    def MY_PROPERTY_GUARD(guard):
        """Guard on my_property of an object."""
        return f"{guard.name}.my_property == {guard.value}"

# Usage in VariableTracker
def var_getattr(self, tx, name):
    if name == "my_property":
        # Install guard
        if self.source:
            install_guard(
                self.source.make_guard(GuardBuilder.MY_PROPERTY_GUARD)
            )
        return ConstantVariable(self.value.my_property)
```

### Tips

- **Guard what you assume**: If you assume a value, guard it
- **Source required**: Can't guard without a source
- **Guard granularity**: Balance between too specific and too general

## Implementing Fast-Path for Operations

**When**: Operation traces slowly or produces verbose graphs.

**Pattern**: Implement direct handling instead of tracing.

### Example: tree_map fast-path

```python
# In UserFunctionVariable
def call_function(self, tx, args, kwargs):
    # Check if this is a special function
    if self._is_tree_map_function():
        return self._maybe_call_tree_map_fastpath(tx, args, kwargs)

    # Normal function call handling
    return super().call_function(tx, args, kwargs)

def _maybe_call_tree_map_fastpath(self, tx, args, kwargs):
    """Fast-path for tree_map operations."""
    map_fn = args[0]
    first_tree = args[1]
    rest = args[2:]

    # Delegate to first tree's handler
    return first_tree.call_tree_map(tx, self, map_fn, rest, kwargs)
```

### Tips

- **Detection first**: Reliably detect when fast-path applies
- **Delegate appropriately**: Let types handle their own logic
- **Fall back gracefully**: Tracing is the fallback
- **Document**: Explain why fast-path exists

## Handling Side Effects

**When**: Operations have side effects that must be tracked.

**Pattern**:

```python
# In VariableTracker
def call_method(self, tx, name, args, kwargs):
    if name == "mutating_method":
        # Track side effect
        tx.output.side_effects.mutation(self)

        # Perform operation
        result = ...

        return result
```

### Side Effect Types

- **Mutation**: Object is modified
- **Store**: Variable stored in container
- **Load**: Variable loaded from container

## Triggering Graph Breaks Properly

**When**: Implementing code that needs to trigger a graph break (unsupported operation, export mode restriction, etc.).

**Pattern**: Use the hint system for proper categorization and user guidance.

```python
from torch._dynamo.exc import unimplemented
from torch._dynamo import graph_break_hints

# Example: Unsupported data-dependent control flow
if not self._can_trace_operation(tx):
    unimplemented(
        gb_type="data_dependent_control_flow",  # Category (no dynamic strings)
        context=f"control flow depends on {var_name}",  # Dynamic details
        explanation="Cannot trace data-dependent if statement",
        hints=[*graph_break_hints.FUNDAMENTAL],  # Hint category
    )
```

### Hint Categories

| Hint | Meaning | When to Use |
|------|---------|-------------|
| `SUPPORTABLE` | Could be implemented with effort | Feature request, not fundamental limitation |
| `FUNDAMENTAL` | Inherent limitation of tracing | Data-dependent control flow, dynamic operations |
| `DIFFICULT` | Very hard to implement | Complex Python semantics, CPython internals |
| `DYNAMO_BUG` | Internal Dynamo error | Dynamo implementation bug, not user code |
| `USER_ERROR` | User code problem | Invalid usage, type errors |
| `CAUSED_BY_EARLIER_GRAPH_BREAK` | Cascading break | Break caused by previous break |

### Components of `unimplemented()`

```python
unimplemented(
    gb_type="category_name",           # Static category (for grouping)
    context=f"details: {dynamic_val}",  # Dynamic context (for debugging)
    explanation="User-facing explanation of why this breaks",
    hints=[*graph_break_hints.CATEGORY],  # Categorization hints
)
```

### Examples

**Data-dependent control flow**:
```python
# In VariableTracker.call_method when tracing if statement
if depends_on_tensor_value:
    unimplemented(
        gb_type="data_dependent_control_flow",
        context=f"if condition depends on tensor {tensor_var}",
        explanation="Cannot determine branch at trace time",
        hints=[*graph_break_hints.FUNDAMENTAL],
    )
```

**Unsupported Python feature**:
```python
# In InstructionTranslator opcode handler
unimplemented(
    gb_type="unsupported_builtin",
    context=f"builtin function {fn_name} not supported",
    explanation=f"Dynamo doesn't support {fn_name} builtin",
    hints=[*graph_break_hints.SUPPORTABLE],
)
```

**Export mode restriction**:
```python
# In VariableTracker method when export=True
if tx.export and not self._export_compatible():
    unimplemented(
        gb_type="export_mode_restriction",
        context=f"operation {op_name} not allowed in export",
        explanation="This operation requires graph break, not allowed in export",
        hints=[*graph_break_hints.FUNDAMENTAL],
    )
```

### Tips

- **`gb_type`**: Use consistent names (check existing `unimplemented()` calls)
- **`context`**: Include dynamic values to help debugging
- **`explanation`**: User-facing message - explain *why* it breaks
- **`hints`**: Choose appropriate category - helps users understand if it's a bug or limitation

## Polyfill Pattern

**When**: Need to inline a C/C++ function for tracing.

**Location**: `torch/_dynamo/polyfills/`

### Pattern

```python
from ..decorators import substitute_in_graph

@substitute_in_graph(
    original_cpp_function,
    can_constant_fold_through=True,
)
def my_polyfill(arg1, arg2, kwarg1=None):
    """
    Python implementation of original_cpp_function.

    Dynamo will inline this instead of tracing through C++.
    """
    # Implement in pure Python
    # Can call other polyfills
    result = ...
    return result
```

### Tips

- **Same signature**: Must match original function
- **Pure Python**: No C++ calls (or they must be polyfilled too)
- **can_constant_fold_through**: Set to True if pure function
- **Documentation**: Explain what the original does

## Testing Patterns

### Basic Test

```python
class TestMyFeature(torch._dynamo.test_case.TestCase):
    def test_basic_case(self):
        def fn(x):
            return my_operation(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)
```

### Test with Multiple Inputs

```python
def test_various_inputs(self):
    def fn(x, y):
        return x + y

    # Test different shapes
    for size in [4, 8, 16]:
        x = torch.randn(size)
        y = torch.randn(size)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
```

### Test Guard Behavior

```python
def test_guards(self):
    def fn(x):
        if x.shape[0] > 10:
            return x * 2
        return x

    # First compilation
    x1 = torch.randn(12)
    opt_fn = torch.compile(fn)
    res1 = opt_fn(x1)

    # Should recompile (different shape)
    x2 = torch.randn(8)
    res2 = opt_fn(x2)

    # Should NOT recompile (same size as x1)
    x3 = torch.randn(12)
    res3 = opt_fn(x3)
```

## Anti-Patterns (Avoid These)

### ❌ Executing During Tracing

```python
# WRONG
def call_method(self, tx, name, args, kwargs):
    # Don't actually execute!
    result = self.value.my_method()
    return ConstantVariable(result)
```

### ✅ Correct

```python
def call_method(self, tx, name, args, kwargs):
    # Symbolically execute
    return self._call_method_handler(tx, name, args, kwargs)
```

### ❌ Bypassing Guards

```python
# WRONG - assuming without guarding
def var_getattr(self, tx, name):
    return ConstantVariable(self.value.attr)
```

### ✅ Correct

```python
def var_getattr(self, tx, name):
    if self.source:
        install_guard(self.source.make_guard(GuardBuilder.HASATTR))
    return ConstantVariable(self.value.attr)
```

### ❌ Creating Parallel Systems

```python
# WRONG - don't create alternative tracing
def my_custom_trace(fn):
    # Custom tracing logic
    ...
```

### ✅ Correct

```python
# Extend existing system
class MyOpcode(InstructionTranslator):
    def MY_OPCODE(self, inst):
        # Add to existing system
        ...
```

## Quick Reference

| Task | File | Method |
|------|------|--------|
| Add opcode | `symbolic_convert.py` | `def OPCODE_NAME(self, inst)` |
| New type | `variables/my_type.py` | `class MyTypeVariable(VariableTracker)` |
| Register type | `variables/builder.py` | Add to `__call__` |
| PyTree support | Your VariableTracker | `def call_tree_map_branch()` |
| Add guard | `guards.py` | `GuardBuilder.MY_GUARD` |
| Polyfill | `polyfills/` | `@substitute_in_graph` decorator |
| Test | `test/dynamo/test_*.py` | `class Test...(TestCase)` |
