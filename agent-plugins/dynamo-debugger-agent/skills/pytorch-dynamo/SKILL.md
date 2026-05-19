---
name: pytorch-dynamo
description: Expert guidance for PyTorch Dynamo (torch.compile) development and debugging. Covers symbolic execution, VariableTracker system, pytree integration, guard failures, guard generation, FakeTensors, SymInts, symbolic shapes, dynamic shapes, C++ ATen ops, .size() vs .sym_size(), .numel() vs .sym_numel(), and bytecode tracing.
---

# PyTorch Dynamo Expert

Expert guidance for working with PyTorch's Dynamo compilation system (`torch.compile`).

## Quick Start

**Working with torch.compile?** Start here:
- Debugging compilation errors → See [debugging-guide.md](debugging-guide.md)
- Understanding architecture → See [architecture.md](architecture.md)
- Implementing features → See [common-patterns.md](common-patterns.md)
- Quick commands → See [quick-reference.md](quick-reference.md)

**Working with pytree operations?** See [pytree-integration.md](pytree-integration.md)

**Writing C++ ATen ops for dynamic shapes?** See [GUARD.md](GUARD.md)

## What is Dynamo?

Dynamo is PyTorch's JIT compiler that intercepts Python bytecode execution to trace and optimize PyTorch models. It powers `torch.compile()`.

**Core idea**: Intercept Python frame execution → Symbolically execute bytecode → Generate FX graph → Compile with backend

## When to Use This Skill

Activate when:
- Working with `torch.compile()` or `torch/_dynamo/` code
- Debugging guard failures or compilation errors
- Implementing new VariableTracker types or opcode handlers
- Understanding pytree integration with compilation
- Adding support for new Python features/types
- Writing tests in `test/dynamo/`

## Core Concepts (30-Second Version)

### Frame Interception
Dynamo installs a custom frame evaluation hook to intercept Python execution.

**File**: `torch/_dynamo/eval_frame.py`

### Symbolic Execution
Instead of executing Python bytecode normally, Dynamo executes it symbolically using `InstructionTranslator`.

**File**: `torch/_dynamo/symbolic_convert.py`

### Variable Tracking
Python objects are represented as `VariableTracker` subclasses during symbolic execution.

**Directory**: `torch/_dynamo/variables/`

### Guard System
Guards ensure compiled code only runs when assumptions hold (e.g., tensor shapes, types).

**Files**: `guards.py`, `guard_failures.py`

### Graph Generation
Symbolic execution produces an FX GraphModule that backends can optimize.

**File**: `output_graph.py`

## Common Tasks

### Debug a Compilation Error

1. Create minimal repro
2. Enable logging: `torch._logging.set_logs(dynamo=logging.INFO, bytecode=True)`
3. Follow workflow in [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md)

### Add Support for a New Type

1. Create `MyTypeVariable` in `torch/_dynamo/variables/`
2. Update `VariableBuilder` to recognize it
3. See [COMMON-PATTERNS.md](COMMON-PATTERNS.md#creating-new-variabletracker)

### Fix a pytree Integration Bug

1. Understand fast-path optimization in [PYTREE-INTEGRATION.md](PYTREE-INTEGRATION.md)
2. Check `call_tree_map_branch()` implementation
3. Ensure both explicit and implicit registration checks

## Key Files Quick Map

```
torch/_dynamo/
├── eval_frame.py          # Frame interception entry point
├── convert_frame.py       # Decides whether to trace
├── symbolic_convert.py    # InstructionTranslator (main bytecode interpreter)
├── output_graph.py        # Graph builder
├── guards.py              # Guard system
├── variables/
│   ├── base.py           # VariableTracker base class
│   ├── tensor.py         # TensorVariable
│   ├── lists.py          # List/Tuple/NamedTuple variables
│   ├── dicts.py          # Dict variables
│   ├── functions.py      # Function variables (includes tree_map fast-path)
│   └── user_defined.py   # UserDefinedObjectVariable (custom classes)
└── polyfills/
    └── pytree.py         # PyTree polyfills for tracing
```

## Progressive Disclosure

- **Getting started**: This file
- **Understanding internals**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Debugging issues**: [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md)
- **Implementing features**: [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
- **PyTree specifics**: [PYTREE-INTEGRATION.md](PYTREE-INTEGRATION.md)
- **Guard architecture**: [GUARD.md](GUARD.md)
- **Quick lookups**: [QUICK-REFERENCE.md](QUICK-REFERENCE.md)

## Development Principles

1. **Respect the architecture** - Don't create parallel systems
2. **Guard everything** - Never bypass the guard system
3. **Use existing abstractions** - Don't invent new Dynamo APIs
4. **Maintain consistency** - VariableTrackers should behave uniformly
5. **Place logic correctly** - Opcode handlers belong in specific places

See [ARCHITECTURE.md](ARCHITECTURE.md#development-rules) for details.

## Getting Help

**Compilation error?** → [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md)
**Guard failure?** → [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md#guard-failures)
**Adding feature?** → [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
**PyTree issue?** → [PYTREE-INTEGRATION.md](PYTREE-INTEGRATION.md)
**C++ ATen ops & guards?** → [GUARD.md](GUARD.md)
**Need quick command?** → [QUICK-REFERENCE.md](QUICK-REFERENCE.md)
