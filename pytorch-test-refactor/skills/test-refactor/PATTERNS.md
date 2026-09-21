# Refactoring Patterns — Before/After Examples

Concrete code transformations for the most common refactoring scenarios.

## Pattern 1: Adding hw_classification to an Existing Class

The simplest case — the class is already well-structured, just needs the attribute.

### Before

```python
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSerialization(TestCase):
    def test_save_load(self):
        x = torch.randn(3, 3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            y = torch.load(f)
        self.assertEqual(x, y)

if __name__ == "__main__":
    run_tests()
```

### After

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)

class TestSerialization(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_save_load(self):
        x = torch.randn(3, 3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            y = torch.load(f)
        self.assertEqual(x, y)

if __name__ == "__main__":
    run_tests()
```

**What changed:** Added `HardwareClassification` import and `hw_classification` attribute. Nothing else.

---

## Pattern 2: Converting a Hardcoded CUDA Test to ACCELERATOR

Tests that hardcode `"cuda"` but test behavior that works on any device.

### Before

```python
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_cuda import TEST_CUDA

class TestBinaryOps(TestCase):
    @unittest.skipIf(not TEST_CUDA, "no CUDA")
    def test_add_on_device(self):
        x = torch.randn(3, 3, device="cuda")
        y = torch.randn(3, 3, device="cuda")
        z = x + y
        self.assertEqual(z.device.type, "cuda")
        self.assertEqual(z.shape, (3, 3))

    @unittest.skipIf(not TEST_CUDA, "no CUDA")
    def test_mul_on_device(self):
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x * y
        self.assertEqual(z.shape, (3, 3))

    def test_add_cpu(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        self.assertEqual(z.shape, (3, 3))

if __name__ == "__main__":
    run_tests()
```

### After

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)
from torch.testing._internal.common_device_type import (
    DeviceTypeTestBase,
    instantiate_device_type_tests,
)


class TestBinaryOpsGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_add_cpu(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        self.assertEqual(z.shape, (3, 3))


class TestBinaryOpsAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_add_on_device(self, device):
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = x + y
        self.assertEqual(z.device.type, device)
        self.assertEqual(z.shape, (3, 3))

    def test_mul_on_device(self, device):
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = x * y
        self.assertEqual(z.shape, (3, 3))


instantiate_device_type_tests(TestBinaryOpsAccelerator, globals())

if __name__ == "__main__":
    run_tests()
```

**What changed:**
- Split into GENERIC (CPU-only `test_add_cpu`) and ACCELERATOR (device tests)
- `"cuda"` → `device` parameter
- `.cuda()` → `.to(device)` (or use `device=device` in creation)
- Removed `@unittest.skipIf(not TEST_CUDA, ...)` — the framework handles device availability
- Changed parent to `DeviceTypeTestBase`
- Added `instantiate_device_type_tests()` call
- Every test method in ACCELERATOR class accepts `device` parameter (linter-enforced)

---

## Pattern 3: Splitting a Mixed Class

A class with tests spanning multiple categories.

### Before

```python
class TestConv(TestCase):
    def test_conv1d_shape(self):
        # Pure CPU logic test
        conv = torch.nn.Conv1d(3, 6, 3)
        x = torch.randn(1, 3, 10)
        self.assertEqual(conv(x).shape, (1, 6, 8))

    @onlyCUDA
    def test_conv2d_forward(self):
        # Device test — should work on any accelerator
        conv = torch.nn.Conv2d(3, 6, 3).to("cuda")
        x = torch.randn(1, 3, 10, 10, device="cuda")
        y = conv(x)
        self.assertEqual(y.shape, (1, 6, 8, 8))

    @onlyCUDA
    def test_cudnn_benchmark(self):
        # CUDA-specific — cudnn is CUDA-only
        with torch.backends.cudnn.flags(benchmark=True):
            conv = torch.nn.Conv2d(3, 6, 3).cuda()
            x = torch.randn(1, 3, 10, 10, device="cuda")
            conv(x)  # Should use cudnn benchmark
```

### After

```python
class TestConvGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_conv1d_shape(self):
        conv = torch.nn.Conv1d(3, 6, 3)
        x = torch.randn(1, 3, 10)
        self.assertEqual(conv(x).shape, (1, 6, 8))


class TestConvAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_conv2d_forward(self, device):
        conv = torch.nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(1, 3, 10, 10, device=device)
        y = conv(x)
        self.assertEqual(y.shape, (1, 6, 8, 8))


class TestConvCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cudnn_benchmark(self):
        with torch.backends.cudnn.flags(benchmark=True):
            conv = torch.nn.Conv2d(3, 6, 3).cuda()
            x = torch.randn(1, 3, 10, 10, device="cuda")
            conv(x)


instantiate_device_type_tests(TestConvAccelerator, globals())
instantiate_device_type_tests(TestConvCUDA, globals(), only_for="cuda")
```

**What changed:**
- One class became three, each with single responsibility
- `@onlyCUDA` removed from the ACCELERATOR test (framework handles it)
- `@onlyCUDA` removed from CUDA-specific test (class-level `hw_classification` handles it)
- Device references converted in the ACCELERATOR class — test methods accept `device` parameter
- CUDA class uses `instantiate_device_type_tests` with `only_for="cuda"` (linter-enforced for device-specific classes)
- ACCELERATOR class must NOT use `only_for` (linter-enforced — use `except_for` for blacklist)

---

## Pattern 4: Handling setUp/tearDown During Split

When the original class has setUp/tearDown that references specific devices.

### Before

```python
class TestLinalg(TestCase):
    def setUp(self):
        super().setUp()
        self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.prev_tf32
        super().tearDown()

    def test_matmul_cpu(self):
        # CPU-only test
        ...

    def test_matmul_device(self):
        # Device test
        x = torch.randn(3, 3, device="cuda")
        ...
```

### After

```python
class TestLinalgGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_matmul_cpu(self):
        ...


class TestLinalgAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        if self.device_type == "cuda":
            self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        if self.device_type == "cuda" and hasattr(self, "prev_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = self.prev_tf32
        super().tearDown()

    def test_matmul_device(self, device):
        x = torch.randn(3, 3, device=device)
        ...


instantiate_device_type_tests(TestLinalgAccelerator, globals())
```

**What changed:**
- setUp/tearDown guarded with `if self.device_type == "cuda"` since tf32 is CUDA-specific
- GENERIC class doesn't need setUp/tearDown at all (no CUDA references)
- ACCELERATOR test methods accept `device` parameter (linter-enforced)

---

## Pattern 5: Converting `instantiate_parametrized_tests` to `instantiate_device_type_tests`

When a file already uses parametrized tests but not device-type tests.

### Before

```python
class TestOps(TestCase):
    @parametrize("dtype", [torch.float32, torch.float64])
    def test_unary_op(self, dtype):
        x = torch.randn(3, 3, dtype=dtype, device="cuda")
        y = torch.sin(x)
        self.assertEqual(y.dtype, dtype)

instantiate_parametrized_tests(TestOps)
```

### After

```python
class TestOpsAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(torch.float32, torch.float64)
    def test_unary_op(self, device, dtype):
        x = torch.randn(3, 3, dtype=dtype, device=device)
        y = torch.sin(x)
        self.assertEqual(y.dtype, dtype)

instantiate_device_type_tests(TestOpsAccelerator, globals())
```

**What changed:**
- `TestCase` → `DeviceTypeTestBase`
- `@parametrize("dtype", [...])` → `@dtypes(...)` (use the device-type framework's dtype parametrization)
- `"cuda"` → `device` parameter
- `instantiate_parametrized_tests` → `instantiate_device_type_tests`
- Test methods accept `device` parameter (linter-enforced for ACCELERATOR classes)

---

## Pattern 6: Real-World Example — test_accelerator.py (PR #185211)

This is an actual merged PR showing the refactoring pattern.

### Before

```python
class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        for device_type in ["cuda", "xpu", "mps"]:
            if is_device_available(device_type):
                accelerator = torch.accelerator.current_accelerator()
                self.assertEqual(accelerator.type, device_type)
```

### After

```python
class TestAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_current_accelerator(self, device):
        accelerator = torch.accelerator.current_accelerator()
        self.assertEqual(accelerator.type, device)

instantiate_device_type_tests(TestAccelerator, globals())
```

**What changed:**
- Removed manual iteration over `["cuda", "xpu", "mps"]`
- Used `device` parameter — the framework provides the device
- `instantiate_device_type_tests` generates per-device test classes automatically
- Test methods accept `device` parameter (linter-enforced for ACCELERATOR classes)

---

## Pattern 7: Real-World Example — test_embedding.py (PR #187922)

Renaming classes to match the classification convention.

### Before

```python
class TestEmbeddingNN(TestCase):
    # 18 CPU-only tests

class TestEmbeddingNNDeviceType(TestCase):
    # 29 device-agnostic tests
```

### After

```python
class TestEmbeddingGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC
    # 18 CPU-only tests

class TestEmbeddingAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR
    # 29 device-agnostic tests

instantiate_device_type_tests(TestEmbeddingAccelerator, globals())
```

**What changed:**
- `TestEmbeddingNN` → `TestEmbeddingGeneric` (clarifies classification)
- `TestEmbeddingNNDeviceType` → `TestEmbeddingAccelerator` (clarifies classification)
- Added `hw_classification` to both
- Ensured `instantiate_device_type_tests` is called

---

## Common Device Reference Conversions

| Before | After |
|--------|-------|
| `"cuda"` | `device` (parameter) |
| `"cuda:0"` | `f"{device}:0"` |
| `torch.device("cuda")` | `torch.device(device)` |
| `x.cuda()` | `x.to(device)` |
| `x.cpu()` then `x.cuda()` | `x.to("cpu")` then `x.to(device)` |
| `torch.cuda.is_available()` | Remove check (framework handles availability) |
| `@unittest.skipIf(not TEST_CUDA, ...)` | Remove (framework handles) |
| `@onlyCUDA` | Remove if ACCELERATOR; keep if device-specific (CUDA) |
| `torch.cuda.device_count()` | Keep if in device-specific (CUDA) or MULTI_DEVICE class |
| `torch.cuda.memory_allocated()` | Keep — this is CUDA-specific, belongs in device-specific (CUDA) |

## TEST_LINTER Compliance

The `TEST_LINTER` ([PR #190173](https://github.com/pytorch/pytorch/pull/190173)) enforces these rules via CI. Violations block the PR.

### hw_classification Attribute Form

Only these exact forms are recognized by the linter (aliased imports are **not** accepted):

```python
hw_classification = HardwareClassification.GENERIC
hw_classification: HardwareClassification = HardwareClassification.GENERIC
```

Using a string (`hw_classification = "GENERIC"`) or an alias (`HC = HardwareClassification; hw_classification = HC.GENERIC`) will fail the linter.

### instantiate_device_type_tests Rules

| Classification | `only_for` | `except_for` | Special |
|---|---|---|---|
| **GENERIC** | Must NOT use `instantiate_device_type_tests` | N/A | Test methods must NOT accept `device`/`devices` param |
| **ACCELERATOR** | Must NOT use `only_for` | May use `except_for` (blacklist approach) | Test methods MUST accept `device`/`devices` param |
| **CPU** | Must use `only_for="cpu"` | Must NOT use `except_for` | Test methods MUST accept `device`/`devices` param |
| **CUDA** | Must use `only_for="cuda"` | Must NOT use `except_for` | Test methods MUST accept `device`/`devices` param |
| **MPS** | Must use `only_for="mps"` | Must NOT use `except_for` | Must pass `allow_mps=True`; test methods MUST accept `device`/`devices` param |
| **XPU** | Must use `only_for="xpu"` | Must NOT use `except_for` | Must pass `allow_xpu=True`; test methods MUST accept `device`/`devices` param |

### ACCELERATOR Decorator Rules

ACCELERATOR classes must not use device-specific `@only*` decorators on test methods. These are forbidden:
`@onlyCPU`, `@onlyCUDA`, `@onlyMPS`, `@onlyXPU`, `@onlyHPU`, `@onlyPRIVATEUSE1`, `@onlyOn`, `@onlyCUDAAndPRIVATEUSE1`, `@onlyNativeDeviceTypes`, `@onlyNativeDeviceTypesAnd`

The only allowed "only" decorator is `@onlyAccelerator`.

### Allowlist

Files not yet migrated are listed in `tools/linter/adapters/test_linter_allowlist.json`. To exempt a file temporarily, add it to the allowlist. To graduate a file, remove it from the allowlist and ensure all test classes comply with the rules above.
