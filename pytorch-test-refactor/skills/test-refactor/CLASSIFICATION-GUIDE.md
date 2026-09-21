# Hardware Classification Decision Tree

Use this decision tree to classify each test class in a PyTorch test file. Walk through the questions top-down — the first match wins.

## What Counts as a Test Class

The TEST_LINTER identifies test classes by the presence of `test_*` methods, not by inheritance. Any class containing methods whose name starts with `test` is treated as a test class and must declare `hw_classification`. This includes intermediate base classes and mixins that define test methods — not just leaf classes.

Classes with no `test_*` methods (pure helpers, mixins with only utility methods) do not need `hw_classification`.

## hw_classification Attribute Form

The linter only accepts the exact forms below. Aliased imports are not recognized.

```python
hw_classification = HardwareClassification.GENERIC
hw_classification: HardwareClassification = HardwareClassification.GENERIC
```

The `HardwareClassification` name must appear literally in both the value and (if present) the annotation. Importing `HardwareClassification as HC` and writing `hw_classification = HC.GENERIC` will fail the linter.

## Decision Tree

```
Is the test about multi-device / distributed behavior?
├── YES → Does it work across accelerator types (gloo, any backend)?
│   ├── YES → MULTI_DEVICE_GENERIC (planned, not yet in linter)
│   └── NO (NCCL-only, multi-GPU CUDA, etc.) → MULTI_DEVICE_SPECIFIC (planned, not yet in linter)
│
└── NO → Does the test reference any specific accelerator?
    ├── NO (pure CPU logic, dispatcher, autograd, serialization) → GENERIC
    │
    └── YES → Can the test work on ANY accelerator (not just CUDA)?
        ├── YES (aten op numerics, device tensor ops, backend integration) → ACCELERATOR
        └── NO (CUDA memory, XPU kernel, MPS graph, TunableOp) → device-specific (CPU/CUDA/MPS/XPU)
```

## Category Details

### GENERIC

Tests that verify shared CPU-side logic with no ties to any accelerator.

**Indicators:**
- No `torch.cuda.*`, `torch.xpu.*`, `torch.mps.*` calls
- No device string literals (`"cuda"`, `"xpu"`, `"mps"`)
- No device-specific decorators (`@onlyCUDA`, `@onlyXPU`)
- Tests dispatcher, autograd mechanics, serialization, JIT, Python API behavior
- FakePG distributed tests (they mock the process group and run on CPU)

**Examples:**
- Testing that `torch.save` / `torch.load` roundtrips correctly
- Testing autograd graph construction
- Testing op schema validation
- Testing Python API argument parsing

**Infrastructure:**
```python
class TestFooGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC
```

### ACCELERATOR

Tests that check on-device behavior and should run across ALL accelerators. CPU is included as a device.

**Indicators:**
- Tests aten operator numerics (correctness, dtype handling)
- Tests tensor operations that should work identically on any device
- Uses or should use `self.device_type` / `device` parameter
- Currently uses `@onlyCUDA` but the logic isn't actually CUDA-specific — it just needs *some* accelerator
- Uses `DeviceTypeTestBase` or `instantiate_device_type_tests`

**Examples:**
- Testing `torch.add` numerics across dtypes
- Testing convolution forward/backward correctness
- Testing embedding lookup on device tensors
- Testing that tensor creation on device works (`torch.randn(3, device=...)`)

**Key question:** "If I replace CUDA with XPU/MPS/PrivateUse1, would this test still make sense and be expected to pass?" If yes → ACCELERATOR.

**Infrastructure:**
```python
class TestFooAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_something(self, device):
        x = torch.randn(3, 3, device=device)
        # ...

instantiate_device_type_tests(TestFooAccelerator, globals())
```

### Device-Specific (CPU / CUDA / MPS / XPU)

Tests locked to a single device because they test hardware-specific functionality.

**Indicators:**
- Tests CUDA-specific APIs: `torch.cuda.memory_allocated()`, CUDA graphs, `cudnn` settings
- Tests XPU-specific kernels or APIs
- Tests MPS graph API or Metal-specific behavior
- Tests TunableOp (CUDA/ROCm-specific tuning)
- Tests backend-specific library integration (cuBLAS, cuDNN, cuSOLVER)
- The test would NOT make sense on another accelerator

**Examples:**
- Testing CUDA memory management (`torch.cuda.empty_cache()`)
- Testing CUDA graph capture and replay
- Testing cuDNN benchmark mode behavior
- Testing TunableOp tunable parameters

**Key question:** "Does this test exercise functionality that only exists on one specific device?" If yes → use the corresponding device-specific classification.

**Infrastructure:**
```python
class TestFooCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cuda_memory(self, device):
        # CUDA-specific test
        torch.cuda.empty_cache()
        # ...

instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")
```

Use the appropriate enum value: `HardwareClassification.CUDA`, `.CPU`, `.XPU`, or `.MPS`.

### MULTI_DEVICE_GENERIC (Planned)

Tests that check multi-device / distributed behavior and should work across accelerator types.

> **Note:** `MULTI_DEVICE_GENERIC` is not yet in the `HardwareClassification` enum or the TEST_LINTER. When it is added, the linter rules will be updated. For now, classify these tests using the best available category and document the intent.

**Indicators:**
- Tests distributed collectives (all_reduce, broadcast, etc.) using backend-agnostic APIs
- Tests `torch.distributed` with gloo or any-backend configuration
- Tests multi-device tensor movement that isn't backend-specific
- Tests distributed training patterns (DDP, FSDP) at the API level

**Examples:**
- Testing all_reduce correctness with gloo backend
- Testing DDP wrapper behavior
- Testing distributed checkpoint save/load

**Infrastructure:**
```python
class TestDistributedGeneric(TestCase):
    hw_classification = HardwareClassification.MULTI_DEVICE_GENERIC
```

### MULTI_DEVICE_SPECIFIC (Planned)

Tests requiring multiple devices of a specific type.

> **Note:** `MULTI_DEVICE_SPECIFIC` is not yet in the `HardwareClassification` enum or the TEST_LINTER. When it is added, the linter rules will be updated. For now, classify these tests using the best available category and document the intent.

**Indicators:**
- Tests NCCL-specific collectives or optimizations
- Tests multi-GPU CUDA functionality (peer-to-peer, NVLink)
- Tests that require `torch.cuda.device_count() > 1`
- Tests XPU-specific multi-device features

**Examples:**
- Testing NCCL all_reduce performance characteristics
- Testing CUDA peer-to-peer memory access
- Testing multi-GPU CUDA graph capture

**Infrastructure:**
```python
class TestDistributedCUDA(TestCase):
    hw_classification = HardwareClassification.MULTI_DEVICE_SPECIFIC
```

## Edge Cases

### Tests with `@onlyCUDA` that are actually ACCELERATOR

Many tests use `@onlyCUDA` simply because CUDA was the only accelerator when the test was written. If the test logic works on any device:
- Remove `@onlyCUDA`
- Move to an ACCELERATOR class
- Replace `"cuda"` with `device`

### Tests that skip on certain devices

A test can be ACCELERATOR even if it skips on some devices. The classification is about intent, not runtime behavior:
```python
class TestOpsAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipCUDAIfNoCudnn  # Skips if cudnn not available, but still ACCELERATOR
    def test_conv_backward(self, device):
        # This test is ACCELERATOR — it runs on any device that supports conv
        ...

instantiate_device_type_tests(TestOpsAccelerator, globals())
```

### setUp/tearDown with device-specific code

If setUp references a specific device (e.g., saving tf32 settings), guard it:
```python
def setUp(self):
    super().setUp()
    if self.device_type == "cuda":
        self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
```

### Helper classes and mixins

Non-test classes (helper classes, mixins, base classes without `test_*` methods) do NOT need `hw_classification`. Only classes that contain `test_*` methods need it.

### Parametrized tests

Tests using `@parametrize` can be any category. The parametrization is orthogonal to hardware classification. A parametrized test that varies dtypes across devices is ACCELERATOR. A parametrized test that varies CPU-only parameters is GENERIC.

## CI Linter Rules (TEST_LINTER)

The `TEST_LINTER` (registered in `.lintrunner.toml`) enforces the invariants below on all test files matching `test/**/test_*.py` and `test/**/*_test.py`. Files in the allowlist (`tools/linter/adapters/test_linter_allowlist.json`) are temporarily exempt. Newly created test files are NOT in the allowlist and are checked immediately. When refactoring a file to comply, remove it from the allowlist.

### GENERIC

| Rule | Enforced |
|------|----------|
| Class must NOT be used with `instantiate_device_type_tests` | Yes |
| Test methods must NOT accept `device` or `devices` parameter | Yes |

### ACCELERATOR

| Rule | Enforced |
|------|----------|
| Class must be used with `instantiate_device_type_tests` | Yes |
| Every test method must accept `device` or `devices` parameter | Yes |
| Test methods must NOT use device-specific `@only*` decorators (except `@onlyAccelerator`) | Yes |
| `instantiate_device_type_tests` must NOT use `only_for` — use `except_for` (blacklist approach) instead | Yes |

Forbidden `@only*` decorators in ACCELERATOR classes: `@onlyCPU`, `@onlyCUDA`, `@onlyMPS`, `@onlyXPU`, `@onlyHPU`, `@onlyPRIVATEUSE1`, `@onlyOn`, `@onlyCUDAAndPRIVATEUSE1`, `@onlyNativeDeviceTypes`, `@onlyNativeDeviceTypesAnd`.

### CPU / CUDA / MPS / XPU (Device-Specific)

| Rule | Enforced |
|------|----------|
| Class must be used with `instantiate_device_type_tests` | Yes |
| Every test method must accept `device` or `devices` parameter | Yes |
| `instantiate_device_type_tests` must use `only_for` matching the device exactly (e.g., `only_for='cuda'`) | Yes |
| `instantiate_device_type_tests` must NOT use `except_for` | Yes |
| MPS classes must pass `allow_mps=True` in `instantiate_device_type_tests` | Yes |
| XPU classes must pass `allow_xpu=True` in `instantiate_device_type_tests` | Yes |

### MULTI_DEVICE_GENERIC / MULTI_DEVICE_SPECIFIC

Not yet enforced by the TEST_LINTER. These classifications are planned for future linter support.

### Allowlist

All existing test files (approximately 1,191 files) are in the allowlist and are exempt from linter checks. The allowlist is a transitional mechanism — as files are refactored to comply, they should be removed from the allowlist so the linter enforces their compliance going forward.

To temporarily exempt a new file, add it to `tools/linter/adapters/test_linter_allowlist.json`. This should only be done as a last resort for files that cannot yet be migrated.
