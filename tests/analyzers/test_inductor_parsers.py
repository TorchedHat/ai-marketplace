"""Tests for Inductor stage parsers."""

import pytest

from analyzers.inductor_parsers import (
    parse_fusion_decisions,
    parse_ir_post_fusion,
    parse_output_code,
)


class TestParseFusionDecisions:
    """Test parse_fusion_decisions parser."""

    @pytest.mark.asyncio
    async def test_fusion_logs(self):
        """Test parsing fusion decision logs."""
        log = """
Fusing buf0 (Pointwise) with buf1 (Pointwise)
  Reason: vertical fusion (producer-consumer)
  Kernel: kernel0

Cannot fuse buf2 (Reduction) with buf1
  Reason: different iteration space
"""
        result = await parse_fusion_decisions(log)
        assert "Fusion Decisions" in result
        assert "fused" in result.lower() or "fusing" in result.lower()


class TestParseIRPostFusion:
    """Test parse_ir_post_fusion parser."""

    @pytest.mark.asyncio
    async def test_loopbody_ir(self):
        """Test parsing LoopBody IR."""
        ir = """
tmp0 = ops.load('buf0', index0)
tmp1 = ops.add(tmp0, 1.0)
tmp2 = ops.mul(tmp1, 2.0)
ops.store('buf1', index0, tmp2)
"""
        result = await parse_ir_post_fusion(ir)
        assert "LoopBody IR" in result
        assert "Load:" in result
        assert "Store:" in result


class TestParseOutputCode:
    """Test parse_output_code parser."""

    @pytest.mark.asyncio
    async def test_triton_kernel(self):
        """Test parsing Triton kernel code."""
        code = """
import triton
import triton.language as tl

@triton.jit
def kernel0(in_ptr, out_ptr, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK
    tmp0 = tl.load(in_ptr + xindex)
    tmp1 = tl.maximum(tmp0, 0.0)
    tl.store(out_ptr + xindex, tmp1)
"""
        result = await parse_output_code(code)
        assert "Generated Kernel" in result
        assert "Triton" in result
        assert "kernel0" in result

    @pytest.mark.asyncio
    async def test_cpp_kernel(self):
        """Test parsing C++ kernel code."""
        code = """
#include <ATen/ATen.h>

extern "C" void kernel_cpu(float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = std::max(in[i], 0.0f);
    }
}
"""
        result = await parse_output_code(code)
        assert "Generated Kernel" in result
        assert "C++" in result
