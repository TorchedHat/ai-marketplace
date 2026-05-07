"""Tests for Inductor stage analyzers."""

import pytest

from analyzers.inductor_trace import (
    analyze_loopbody,
    analyze_lowering,
    analyze_triton_codegen,
    parse_fusion_decisions,
)


# Real paths from compilation run
MODEL_DIR = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__0_inference_0.0"


class TestParseFusionDecisions:
    """Tests for parse_fusion_decisions function."""

    @pytest.mark.asyncio
    async def test_parse_fusion_basic(self):
        """Test parsing fusion decisions."""
        fusion_log = """
FusionDecision: buf0 (Pointwise) <- producer
FusionDecision: buf1 (Pointwise) <- consumer
  ✓ Ranges match: [10, 100]
  ✓ Vertical fusion (producer-consumer)
  → Fused into 1 kernel

FusionDecision: buf1_fused (Pointwise) <- producer
FusionDecision: buf2 (Reduction) <- consumer
  ✗ Cannot fuse: Incompatible iteration spaces
"""
        result = await parse_fusion_decisions(fusion_log)
        
        assert "Fusion Analysis:" in result
        assert "Successful Fusions: 1" in result
        assert "Failed Fusions: 1" in result


class TestAnalyzeTritonCodegen:
    """Tests for analyze_triton_codegen function."""

    @pytest.mark.asyncio
    async def test_triton_codegen_basic(self):
        """Test Triton kernel analysis."""
        triton_file = f"{MODEL_DIR}/output_code.py"
        
        with open(triton_file) as f:
            content = f.read()
        
        result = await analyze_triton_codegen(content)
        
        assert "Triton Codegen Analysis:" in result
        assert "Kernels" in result


class TestAnalyzeLowering:
    """Tests for analyze_lowering function."""

    @pytest.mark.asyncio
    async def test_lowering_basic(self):
        """Test lowering analysis."""
        ir_file = f"{MODEL_DIR}/ir_pre_fusion.txt"
        fx_file = f"{MODEL_DIR}/fx_graph_readable.py"
        
        result = await analyze_lowering(ir_file, fx_file)
        
        assert "Lowering Analysis:" in result
        assert "ATen Operations" in result
        assert "IR Operations" in result


class TestAnalyzeLoopbody:
    """Tests for analyze_loopbody function."""

    @pytest.mark.asyncio
    async def test_loopbody_basic(self):
        """Test LoopBody analysis."""
        lb_file = f"{MODEL_DIR}/ir_post_fusion.txt"
        
        result = await analyze_loopbody(lb_file)
        
        assert "LoopBody Analysis:" in result or "Operations:" in result
        assert "ops." in result  # Should have ops.* operations
