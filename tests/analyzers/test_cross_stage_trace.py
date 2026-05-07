"""Tests for cross-stage analyzers."""

import pytest

from analyzers.cross_stage_trace import search_ir, trace_operation


MODEL_DIR = "torch_compile_debug/run_2026_05_07_20_40_41_161544-pid_3044825/torchinductor/model__0_inference_0.0"


class TestTraceOperation:
    """Tests for trace_operation function."""

    @pytest.mark.asyncio
    async def test_trace_operation_basic(self):
        """Test tracing operation through pipeline."""
        result = await trace_operation("relu", MODEL_DIR)
        
        assert "Operation Trace:" in result
        assert "Stages Found:" in result
        assert "Pipeline Summary:" in result

    @pytest.mark.asyncio
    async def test_trace_operation_missing_dir(self):
        """Test error handling for missing directory."""
        result = await trace_operation("relu", "/nonexistent/dir")
        
        assert "Error" in result


class TestSearchIR:
    """Tests for search_ir function."""

    @pytest.mark.asyncio
    async def test_search_ir_dynamo(self):
        """Test searching Dynamo stage."""
        result = await search_ir("relu", "dynamo", MODEL_DIR)
        
        assert "Search Results:" in result or "Dynamo" in result

    @pytest.mark.asyncio
    async def test_search_ir_inductor(self):
        """Test searching Inductor stage."""
        result = await search_ir("relu", "inductor", MODEL_DIR)
        
        assert "Search Results:" in result or "Inductor" in result

    @pytest.mark.asyncio
    async def test_search_ir_all_stages(self):
        """Test searching all stages."""
        result = await search_ir("sum", "all", MODEL_DIR)

        assert "Pattern:" in result
        # Should show results from multiple stages
        assert result.count("===") >= 3  # Section separators for Dynamo, AOT, Inductor, LoopBody

    @pytest.mark.asyncio
    async def test_search_ir_invalid_stage(self):
        """Test error handling for invalid stage."""
        result = await search_ir("relu", "invalid_stage", MODEL_DIR)
        
        assert "Error" in result
        assert "Invalid stage" in result

    @pytest.mark.asyncio
    async def test_search_ir_missing_dir(self):
        """Test error handling for missing directory."""
        result = await search_ir("relu", "dynamo", "/nonexistent/dir")
        
        assert "Error" in result
