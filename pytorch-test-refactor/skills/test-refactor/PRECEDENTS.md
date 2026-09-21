---
last_updated: 2026-07-18
update_cadence: weekly
---

# Precedents — Reviewer Guidance

Corrections from upstream reviewers on openreg/hw_classification PRs. These override generic guidance in SKILL.md/PATTERNS.md. Grouped by workflow phase — check the section matching your current phase.

This file is the sole source of reviewer guidance. It is refreshed at a regular cadence (currently weekly) by the maintainer.

## Classify

Do not classify `@onlyCPU` tests as GENERIC automatically.
Many are historical — the test logic may be device-agnostic.
Examine the actual logic first. If it works on accelerators, keep it in ACCELERATOR and remove the decorator. Defer to follow-up PRs if uncertain.
Why: @fffrog rejected auto-classification in [#185881](https://github.com/pytorch/pytorch/pull/185881).

Do not classify a test as MULTI_ACCELERATOR based on class name alone.
A class with `world_size = 1` or `@skip_if_lt_x_gpu(1)` that runs on a single GPU should be CUDA, not MULTI_ACCELERATOR_CUDA. Match classification to actual runtime requirements.
Why: @mansiag05 caught misclassification in [#190082](https://github.com/pytorch/pytorch/pull/190082).

## Refactor

Do not include device names (CUDA/XPU/MPS) in class names passed to `instantiate_device_type_tests`.
Use neutral names — the framework auto-appends the device suffix.
  ✗ `TestFooCUDA` → `instantiate_device_type_tests` → `TestFooCUDACUDA`
  ✓ `TestFoo` → `instantiate_device_type_tests` → `TestFooCUDA`
Why: @albanD caught this in [#185802](https://github.com/pytorch/pytorch/pull/185802).

Do not rename classes that aren't being split.
Just add `hw_classification`. Only rename when splitting into multiple classes.
  ✗ `TestReductions` → `TestReductionsDevice` (unnecessary churn)
  ✓ `TestReductions` + `hw_classification = HardwareClassification.ACCELERATOR`
Why: @fffrog rejected gratuitous renames in [#185881](https://github.com/pytorch/pytorch/pull/185881).

Device-specific classes must use `instantiate_device_type_tests` with `only_for`.
The TEST_LINTER (PR [#190173](https://github.com/pytorch/pytorch/pull/190173)) now requires `instantiate_device_type_tests` for all device-specific classes, with `only_for` matching the device. This supersedes earlier guidance from #185881/#185802 that avoided `instantiate_device_type_tests` for device-specific classes.
  ✓ `TestReductionCPU` with `hw_classification = HardwareClassification.CPU` + `instantiate_device_type_tests(TestReductionCPU, globals(), only_for="cpu")`
  ✗ `TestReductionCPU` with `hw_classification = HardwareClassification.CPU` and no `instantiate_device_type_tests` call
Why: TEST_LINTER PR [#190173](https://github.com/pytorch/pytorch/pull/190173) formalized the requirement.

Do not manually create per-device subclasses from a mixin for device variation.
Use `instantiate_device_type_tests()` + decorators (`@onlyCUDA`, `@skipMPS`, `@onlyAccelerator`) instead.
  ✗ `TestFooMixin` → `TestFooCUDA(TestFooMixin)` + `TestFooXPU(TestFooMixin)`
  ✓ `TestFoo` → `instantiate_device_type_tests(TestFoo, globals())`
Why: @jbschlosser rejected manual per-device subclassing in [#188331](https://github.com/pytorch/pytorch/pull/188331).
Note: This does NOT prohibit copying shared test methods onto a class via `setattr` to work around `instantiate_device_type_tests` only renaming methods directly on the class (not inherited ones). That pattern is a valid workaround for `MultiProcessTestCase`-based distributed tests where standard inheritance doesn't interact correctly with `instantiate_device_type_tests`.

Do not create separate classes for tests that just need an accelerator.
Use `@onlyAccelerator` decorator within the existing device-type class instead of `@onlyCUDA`.
  ✗ Extract `test_foo` into new `TestBarCUDA` class
  ✓ Keep `test_foo` in device-type class with `@onlyAccelerator`
Why: @fffrog guided this approach in [#185881](https://github.com/pytorch/pytorch/pull/185881).