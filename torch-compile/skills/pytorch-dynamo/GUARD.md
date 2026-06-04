# Dynamo Guard Generation Architecture

Guide to understanding how guards are generated in PyTorch Dynamo and how to write guard-free code for dynamic shapes.

## Table of Contents

- [What Are Guards?](#what-are-guards)
- [FakeTensors and SymInts](#faketensors-and-symints)
- [The Python/C++ Boundary](#the-pythonc-boundary)
- [Concrete vs Symbolic APIs](#concrete-vs-symbolic-apis)
- [Guard Generation Code Path](#guard-generation-code-path)
- [Best Practices for C++ Code](#best-practices-for-c-code)
- [Common Patterns](#common-patterns)
- [Testing for Guards](#testing-for-guards)
- [When Guards Are Acceptable](#when-guards-are-acceptable)
- [Quick Reference](#quick-reference)

---

## What Are Guards?

**Guards** are runtime checks that verify compilation assumptions are still valid.

```python
# During compilation with dynamic=True
x = torch.randn(batch_size, 10)  # batch_size is symbolic

# If C++ code calls x.size(0), Dynamo creates a guard:
# "This compiled code is only valid when batch_size == 32"
```

**Why Guards Exist:**
- Compilation requires some concrete information (tensor rank, dtypes)
- Guards validate that a compiled graph can be reused
- **Too many guards** = recompilation overhead = defeats `dynamic=True`

**Guard Failure = Recompilation:**
```python
compiled_f = torch.compile(f, dynamic=True)
compiled_f(torch.randn(32, 10))  # Compiles with batch_size guard: 32
compiled_f(torch.randn(64, 10))  # Guard fails! Recompiles for 64
```

**Goal:** Create **rank guards** (acceptable), not **value guards** on dimension sizes.

---

## FakeTensors and SymInts

### FakeTensors

During `torch.compile()` tracing, tensors are replaced with **FakeTensors**:

```python
# Real execution:
x = torch.randn(3, 5)  # Real tensor with concrete shape [3, 5]

# During torch.compile(dynamic=True) tracing:
x = FakeTensor(...)    # Fake tensor with shape [s0, s1] (symbolic)
```

**FakeTensors** have no data (just metadata), have symbolic shapes (SymInts), intercept method calls, and enable tracing without executing kernels.

### SymInts

**SymInt** = Symbolic Integer - represents a dimension that could have any value.

```python
x.size(0)      # Concrete: Returns int (3)
x.sym_size(0)  # Symbolic: Returns SymInt (s0)
```

**Key insight:** SymInts represent unknowns without forcing concrete values.

---

## The Python/C++ Boundary

### How C++ Can Trigger Guards

When Python calls a C++ ATen op during tracing:

```python
result = F.cross_entropy(input, target, weight=weight)
```

```cpp
// C++ side (LossNLL.cpp):
static Tensor cross_entropy_loss_prob_target(...) {
  const auto n_classes = self.size(class_dim);  // ⚠️ Triggers guard!
  TORCH_CHECK(weight.numel() == n_classes);     // ⚠️ Triggers guard!
}
```

**What happens:**
1. `self` is a **FakeTensor** with symbolic shapes
2. C++ calls `self.size(class_dim)` expecting `int64_t`
3. FakeTensor intercepts via `__torch_dispatch__`
4. Must return concrete `int` to C++
5. **Guard created:** "n_classes == 5"
6. Concrete value returned to C++

**Problem:** Even with Python `dynamic=True`, C++ forced specialization.

---

## Concrete vs Symbolic APIs

### The Problem: Concrete APIs

```cpp
// These trigger guards during tracing:
int64_t size = self.size(dim);           // Forces SymInt → int64_t → guard
int64_t numel = weight.numel();          // Forces SymInt → int64_t → guard
IntArrayRef sizes = self.sizes();        // Forces all dims → guards
```

### The Solution: Symbolic APIs

```cpp
// These preserve symbolic information:
SymInt sym_size = self.sym_size(dim);         // Returns SymInt, no guard
SymInt sym_numel = weight.sym_numel();        // Returns SymInt, no guard
SymIntArrayRef sym_sizes = self.sym_sizes();  // Returns SymInts, no guards
```
---

## Guard Generation Code Path

### Complete Flow: C++ → Python → Guard

```
C++ ATen Op
  const auto n = self.size(1);
          ↓ PyBind11
FakeTensor.__torch_dispatch__
  torch/_subclasses/fake_tensor.py:876
  Intercepts: torch.ops.aten.size.default
          ↓
_DISPATCH_META_HANDLERS
  torch/_subclasses/fake_tensor.py:3430
  lambda args: tuple(int(s) for s in ...)  ← Calls int()!
          ↓
SymInt.__int__()
  torch/__init__.py:462
  return self.node.int_()
          ↓
SymNode.int_()
  torch/fx/experimental/sym_node.py:501
  return self.guard_int("", 0)
          ↓
SymNode.guard_int()
  torch/fx/experimental/sym_node.py:553
  r = self.evaluate()
          ↓
ShapeEnv.evaluate_sym_node()
  torch/fx/experimental/symbolic_shapes.py
  ✅ GUARD RECORDED: "s0 == 5"
  Returns concrete value to C++
```

**Key Files:**
- Guard creation: `torch/fx/experimental/symbolic_shapes.py::ShapeEnv`
- FakeTensor dispatch: `torch/_subclasses/fake_tensor.py::__torch_dispatch__`
- SymInt/SymNode: `torch/__init__.py::SymInt`, `torch/fx/experimental/sym_node.py`

---

## Best Practices for C++ Code

### ✅ DO: Use Symbolic APIs

```cpp
// Good - preserves symbolic shapes
const auto n_classes = self.sym_size(class_dim);
const auto weight_numel = weight.sym_numel();

TORCH_CHECK(
    !weight.defined() || (weight.dim() == 1 && weight.sym_numel() == n_classes),
    "weight must match n_classes but got ", weight.sym_sizes()
);
```

### ❌ DON'T: Use Concrete APIs on Dynamic Dimensions

```cpp
// Bad - creates guards
const auto n_classes = self.size(class_dim);        // Guard: s0 == 5
const auto weight_numel = weight.numel();            // Guard: s1 == 5
TORCH_CHECK(weight.numel() == n_classes, "...");    // Multiple guards!
```

### ⚠️ EXCEPTION: Rank (ndim) Always Specializes

```cpp
// This is OK - rank always specializes
const auto ndim = self.dim();           // Returns int (not SymInt)
const auto class_dim = ndim == 1 ? 0 : 1;
// Dynamic shapes = dynamic *sizes*, NOT dynamic *rank*
```

**Why:** Number of dimensions must be known at compile time.

### Pattern: Dimension Index is Concrete, Size is Symbolic

```cpp
const auto class_dim = 1;                         // int (which dimension)
const auto n_classes = self.sym_size(class_dim);  // SymInt (size at dim 1)
```

**Remember:**
- **Index into dimensions:** Concrete `int`
- **Size of dimension:** Symbolic `SymInt`

---

## Common Patterns

### Size Comparisons

```cpp
// ❌ Bad: TORCH_CHECK(weight.numel() == self.size(1), "Size mismatch");
// ✅ Good: TORCH_CHECK(weight.sym_numel() == self.sym_size(1), "Size mismatch");
```

### Error Messages

```cpp
// ❌ Bad: "Expected size ", self.size(0), " but got ", other.size(0)
// ✅ Good: "Expected size ", self.sym_size(0), " but got ", other.sym_size(0)
```

**Why:** Even unused error messages trigger guards during tracing!

### Loop Bounds

```cpp
// ❌ Bad: for (int64_t i = 0; i < self.size(0); i++) { ... }
// ✅ Good: for (const auto i : c10::irange(self.sym_size(0))) { ... }
```

### Allocating Tensors

```cpp
// ❌ Bad: auto output = at::empty({self.size(0), self.size(1)});
// ✅ Good: auto output = at::empty_symint(self.sym_sizes());
```

---

## Testing for Guards

### Check for Recompilation

```python
import torch
import torch._dynamo as dynamo

compiled_op = torch.compile(my_op, dynamic=True)

compiled_op(torch.randn(3, 5), torch.randn(5))   # First call
compiled_op(torch.randn(3, 10), torch.randn(10)) # Different size

print(f"Compilations: {dynamo.utils.compile_times()}")
# Should be 1 if no guards, 2 if guards created
```

### Enable Guard Logging

```python
import logging
torch._logging.set_logs(dynamo=logging.DEBUG)
compiled_op(x, w)  # Guards will be printed
```

Look for: `GUARDS: - L['x'].size()[1] == 5  # ← Unwanted guard!`

---

## When Guards Are Acceptable

### ✅ Acceptable Guards

- **Rank:** `self.dim()` - different ranks need different code paths
- **Data Type:** `self.scalar_type()` - different dtypes need different kernels
- **Device:** `self.device()` - CPU vs CUDA need different implementations
- **Boolean Flags:** `if (reduction == "mean")` - control flow decisions

### ❌ Avoid Guards On

- **Tensor Sizes:** `self.size(0)` - defeats `dynamic=True`
- **Batch Dimensions:** Most common dynamic dimension
- **Sequence Lengths:** Common in NLP models

---

## Quick Reference

### API Cheat Sheet

| Concrete (creates guards) | Symbolic (no guards) | Use Case |
|---------------------------|---------------------|----------|
| `.size(dim)` → `int64_t` | `.sym_size(dim)` → `SymInt` | Get dimension size |
| `.sizes()` → `IntArrayRef` | `.sym_sizes()` → `SymIntArrayRef` | Get all sizes |
| `.numel()` → `int64_t` | `.sym_numel()` → `SymInt` | Total elements |
| `.stride(dim)` → `int64_t` | `.sym_stride(dim)` → `SymInt` | Get stride |
| `.strides()` → `IntArrayRef` | `.sym_strides()` → `SymIntArrayRef` | All strides |

### Decision Tree

```
Need size/numel/stride?
├─ Dimension INDEX? → int (always concrete)
├─ Rank (ndim)? → .dim() (concrete, OK)
├─ Dynamic dimension (batch/seq_len)? → .sym_size(dim) ✅
├─ Fixed dimension (e.g., RGB=3)? → .size(dim) (OK)
└─ Error message only? → .sym_size(dim) ✅
```

---

## Real-World Example

### Before: Creates Guards

```cpp
// aten/src/ATen/native/LossNLL.cpp
static Tensor cross_entropy_loss_prob_target(...) {
  const auto class_dim = self.dim() == 1 ? 0 : 1;        // OK
  const auto n_classes = self.size(class_dim);           // ❌ Guard!
  TORCH_CHECK(!weight.defined() ||
      (weight.dim() == 1 && weight.numel() == n_classes), // ❌ Guards!
      "weight should be defined for all ", n_classes,
      " classes but got ", weight.sizes());               // ❌ Guard!
}
```

**Problem:** Creates guards on `n_classes`, defeats `dynamic=True`.

### After: Guard-Free

```cpp
// aten/src/ATen/native/LossNLL.cpp (fixed)
static Tensor cross_entropy_loss_prob_target(...) {
  const auto class_dim = self.dim() == 1 ? 0 : 1;        // OK
  const auto n_classes = self.sym_size(class_dim);       // ✅ Symbolic!
  TORCH_CHECK(!weight.defined() ||
      (weight.dim() == 1 && weight.sym_numel() == n_classes), // ✅ Symbolic!
      "weight should be defined for all ", n_classes,
      " classes but got ", weight.sym_sizes());           // ✅ Symbolic!
}
```

**Benefits:** Single compiled version works for all class counts, no recompilation.

---

## Summary

**Guards = Recompilation Overhead**

To avoid unnecessary guards in C++ ATen ops:

1. Use `sym_*` APIs for sizes, numel, strides
2. Rank specialization is fine (`.dim()`)
3. SymInts work everywhere, no branching needed
4. Test with varying sizes to catch guards
5. Even error messages can trigger guards

**Remember:**
- **Dimension index:** Concrete `int` (which dimension)
- **Dimension size:** Symbolic `SymInt` (value at that dimension)
- **Rank (ndim):** Always concrete (expected specialization)

**Key insight:** The `sym_*` APIs work transparently for both concrete and symbolic tensors without branching or performance overhead.
