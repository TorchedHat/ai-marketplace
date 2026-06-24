# Hardware Classification Decision Tree

Use this decision tree to classify each test class in a PyTorch test file. Walk through the questions top-down — the first match wins.

## Decision Tree

```
Is the test about multi-device / distributed behavior?
├── YES → Does it work across accelerator types (gloo, any backend)?
│   ├── YES → MULTI_DEVICE_GENERIC
│   └── NO (NCCL-only, multi-GPU CUDA, etc.) → MULTI_DEVICE_SPECIFIC
│
└── NO → Does the test reference any specific accelerator?
    ├── NO (pure CPU logic, dispatcher, autograd, serialization) → GENERIC
    │
    └── YES → Can the test work on ANY accelerator (not just CUDA)?
        ├── YES (aten op numerics, device tensor ops, backend integration) → DEVICE_GENERIC
        └── NO (CUDA memory, XPU kernel, MPS graph, TunableOp) → DEVICE_SPECIFIC
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

### DEVICE_GENERIC

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

**Key question:** "If I replace CUDA with XPU/MPS/PrivateUse1, would this test still make sense and be expected to pass?" If yes → DEVICE_GENERIC.

**Infrastructure:**
```python
class TestFooDeviceGeneric(DeviceTypeTestBase):
    hw_classification = HardwareClassification.DEVICE_GENERIC

    def test_something(self):
        x = torch.randn(3, 3, device=self.device_type)
        # ...

instantiate_device_type_tests(TestFooDeviceGeneric, globals())
```

### DEVICE_SPECIFIC

Tests locked to a single accelerator because they test hardware-specific functionality.

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

**Key question:** "Does this test exercise functionality that only exists on one specific device?" If yes → DEVICE_SPECIFIC.

**Infrastructure:**
```python
class TestFooCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cuda_memory(self):
        # CUDA-specific test
        torch.cuda.empty_cache()
        # ...
```

Use the appropriate enum value: `HardwareClassification.CUDA`, `.XPU`, or `.MPS`.

### MULTI_DEVICE_GENERIC

Tests that check multi-device / distributed behavior and should work across accelerator types.

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

> **Note:** If `MULTI_DEVICE_GENERIC` is not yet in the `HardwareClassification` enum, add it:
> ```python
> # In torch/testing/_internal/common_utils.py
> MULTI_DEVICE_GENERIC = "multi_device_generic"
> ```

### MULTI_DEVICE_SPECIFIC

Tests requiring multiple devices of a specific type.

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

> **Note:** If `MULTI_DEVICE_SPECIFIC` is not yet in the `HardwareClassification` enum, add it:
> ```python
> # In torch/testing/_internal/common_utils.py
> MULTI_DEVICE_SPECIFIC = "multi_device_specific"
> ```

## Edge Cases

### Tests with `@onlyCUDA` that are actually DEVICE_GENERIC

Many tests use `@onlyCUDA` simply because CUDA was the only accelerator when the test was written. If the test logic works on any device:
- Remove `@onlyCUDA`
- Move to a DEVICE_GENERIC class
- Replace `"cuda"` with `self.device_type`

### Tests that skip on certain devices

A test can be DEVICE_GENERIC even if it skips on some devices. The classification is about intent, not runtime behavior:
```python
class TestOpsDeviceGeneric(DeviceTypeTestBase):
    hw_classification = HardwareClassification.DEVICE_GENERIC

    @skipCUDAIfNoCudnn  # Skips if cudnn not available, but still device-generic
    def test_conv_backward(self):
        # This test is device-generic — it runs on any device that supports conv
        ...
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

Non-test classes (helper classes, mixins, base classes without test methods) do NOT need `hw_classification`. Only classes that contain `test_*` methods need it.

### Parametrized tests

Tests using `@parametrize` can be any category. The parametrization is orthogonal to hardware classification. A parametrized test that varies dtypes across devices is DEVICE_GENERIC. A parametrized test that varies CPU-only parameters is GENERIC.
