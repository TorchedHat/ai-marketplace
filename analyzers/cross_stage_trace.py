"""
Cross-Stage Analyzers

Tools that span multiple compilation stages:
- trace_operation: Follow operation through entire pipeline
- search_ir: Search across all stages
"""

import re
from pathlib import Path


async def trace_operation(operation: str, debug_dir: str) -> str:
    """
    Trace an operation through all compilation stages.

    Searches for operation in:
    - FX graphs (Dynamo)
    - AOT graphs (if training)
    - Inductor IR (lowering)
    - LoopBody IR (ops.*)
    - Triton kernel (codegen)
    """
    debug_path = Path(debug_dir)

    if not debug_path.exists():
        return f"Error: Debug directory not found: {debug_dir}"

    # Search each stage
    stages = {
        "Dynamo (FX)": _search_fx_graphs(operation, debug_path),
        "AOT (Joint)": _search_aot_graphs(operation, debug_path),
        "Inductor (IR)": _search_inductor_ir(operation, debug_path),
        "LoopBody": _search_loopbody(operation, debug_path),
        "Triton": _search_triton(operation, debug_path)
    }

    # Count how many stages found the operation
    found_count = sum(1 for results in stages.values() if results)

    return f"""Operation Trace: {operation}

Debug Directory: {debug_dir}
Stages Found: {found_count}/5

{_format_trace_results(stages)}

Pipeline Summary:
{_generate_pipeline_summary(operation, stages)}
"""


async def search_ir(pattern: str, stage: str, debug_dir: str) -> str:
    """
    Search IR files for a pattern.

    Supports:
    - Regex patterns
    - Operation names
    - Buffer names
    - IR node types

    Args:
        pattern: Regex pattern to search for
        stage: Stage to search in (dynamo, aot, inductor, loopbody, all)
        debug_dir: Path to torch_compile_debug directory

    Returns:
        Formatted search results with file paths, line numbers, and context
    """
    debug_path = Path(debug_dir)

    if not debug_path.exists():
        return f"Error: Debug directory not found: {debug_dir}"

    # Search based on stage
    if stage == "dynamo":
        results = _search_fx_graphs_with_lines(pattern, debug_path)
        stage_name = "Dynamo (FX Graphs)"
    elif stage == "aot":
        results = _search_aot_graphs_with_lines(pattern, debug_path)
        stage_name = "AOT (Autograd)"
    elif stage == "inductor":
        results = _search_inductor_ir_with_lines(pattern, debug_path)
        stage_name = "Inductor (IR)"
    elif stage == "loopbody":
        results = _search_loopbody_with_lines(pattern, debug_path)
        stage_name = "LoopBody (ops.*)"
    elif stage == "all":
        results = {
            "Dynamo": _search_fx_graphs_with_lines(pattern, debug_path),
            "AOT": _search_aot_graphs_with_lines(pattern, debug_path),
            "Inductor": _search_inductor_ir_with_lines(pattern, debug_path),
            "LoopBody": _search_loopbody_with_lines(pattern, debug_path)
        }
        return _format_multi_stage_search(pattern, results)
    else:
        return f"Error: Invalid stage '{stage}'. Valid stages: dynamo, aot, inductor, loopbody, all"

    # Format single-stage results
    return _format_single_stage_search(pattern, stage_name, results)


def _search_fx_graphs(pattern: str, debug_path: Path) -> list[dict]:
    """Search FX graph files for operation."""
    results = []

    # Search fx_graph_*.py files
    for fx_file in debug_path.glob("fx_graph_*.py"):
        try:
            content = fx_file.read_text()

            # Search for operation in FX graph (Python code format)
            # Pattern: torch.ops.aten.{operation}.default or torch.ops.aten.{operation}(
            matches = re.finditer(
                rf'torch\.ops\.aten\.{re.escape(pattern)}[\.\(]',
                content,
                re.MULTILINE
            )

            for match in matches:
                # Get context (a few lines before/after)
                start = max(0, match.start() - 200)
                end = min(len(content), match.end() + 200)
                context = content[start:end]

                results.append({
                    'file': fx_file.name,
                    'match': match.group(0),
                    'context': context[:500]  # Limit context
                })

        except Exception:
            pass  # Skip files with errors

    return results


def _search_fx_graphs_with_lines(pattern: str, debug_path: Path) -> list[dict]:
    """
    Search FX graph files for pattern with line numbers.

    Returns results with line numbers and surrounding context.
    """
    results = []

    # Search fx_graph_*.py files
    for fx_file in debug_path.glob("fx_graph_*.py"):
        try:
            lines = fx_file.read_text().splitlines()

            # Search each line for the pattern
            # Use regex search to support complex patterns
            pattern_re = re.compile(pattern, re.IGNORECASE)

            for line_num, line in enumerate(lines, start=1):
                if pattern_re.search(line):
                    # Get surrounding context (3 lines before/after)
                    context_start = max(0, line_num - 4)
                    context_end = min(len(lines), line_num + 3)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        'file': fx_file.name,
                        'line': line_num,
                        'match': line.strip(),
                        'context': context_lines
                    })

        except Exception:
            pass  # Skip files with errors

    return results


def _search_aot_graphs(pattern: str, debug_path: Path) -> list[dict]:
    """Search AOT graph files for operation."""
    results = []

    # Search model__*__*.py files (joint, forward, backward)
    for aot_file in debug_path.glob("model__*__*.py"):
        try:
            content = aot_file.read_text()

            # Search for aten operation
            matches = re.finditer(
                rf'aten\.{re.escape(pattern)}',
                content,
                re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                start = max(0, match.start() - 200)
                end = min(len(content), match.end() + 200)
                context = content[start:end]

                results.append({
                    'file': aot_file.name,
                    'match': match.group(0),
                    'context': context[:500]
                })

        except Exception:
            pass

    return results


def _search_aot_graphs_with_lines(pattern: str, debug_path: Path) -> list[dict]:
    """
    Search AOT graph files for pattern with line numbers.

    Returns results with line numbers and surrounding context.
    """
    results = []

    # Search model__*__*.py files (joint, forward, backward)
    for aot_file in debug_path.glob("model__*__*.py"):
        try:
            lines = aot_file.read_text().splitlines()

            # Search each line for the pattern
            pattern_re = re.compile(pattern, re.IGNORECASE)

            for line_num, line in enumerate(lines, start=1):
                if pattern_re.search(line):
                    # Get surrounding context
                    context_start = max(0, line_num - 4)
                    context_end = min(len(lines), line_num + 3)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        'file': aot_file.name,
                        'line': line_num,
                        'match': line.strip(),
                        'context': context_lines
                    })

        except Exception:
            pass

    return results


def _search_inductor_ir(pattern: str, debug_path: Path) -> list[dict]:
    """Search Inductor IR files for operation or lowering."""
    results = []

    # Search ir_*.txt files (pre-fusion IR)
    for ir_file in debug_path.glob("ir_*.txt"):
        if 'post_fusion' in ir_file.name:
            continue  # Skip, handled by _search_loopbody

        try:
            content = ir_file.read_text()

            # Search for operation references
            # Look for ops.{operation}( in loop bodies or node descriptions
            matches = re.finditer(
                rf'ops\.{re.escape(pattern)}\(',
                content,
                re.MULTILINE
            )

            for match in matches:
                start = max(0, match.start() - 300)
                end = min(len(content), match.end() + 300)
                context = content[start:end]

                results.append({
                    'file': ir_file.name,
                    'match': match.group(0),
                    'context': context[:500]
                })

        except Exception:
            pass

    return results


def _search_inductor_ir_with_lines(pattern: str, debug_path: Path) -> list[dict]:
    """
    Search Inductor IR files for pattern with line numbers.

    Returns results with line numbers and surrounding context.
    """
    results = []

    # Search ir_*.txt files (pre-fusion IR)
    for ir_file in debug_path.glob("ir_*.txt"):
        if 'post_fusion' in ir_file.name:
            continue  # Skip, handled by _search_loopbody

        try:
            lines = ir_file.read_text().splitlines()

            # Search each line for the pattern
            pattern_re = re.compile(pattern, re.IGNORECASE)

            for line_num, line in enumerate(lines, start=1):
                if pattern_re.search(line):
                    # Get surrounding context
                    context_start = max(0, line_num - 4)
                    context_end = min(len(lines), line_num + 3)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        'file': ir_file.name,
                        'line': line_num,
                        'match': line.strip(),
                        'context': context_lines
                    })

        except Exception:
            pass

    return results


def _search_loopbody(pattern: str, debug_path: Path) -> list[dict]:
    """Search LoopBody IR files for ops.* operations."""
    results = []

    # Search ir_post_fusion*.txt files
    for lb_file in debug_path.glob("ir_post_fusion*.txt"):
        try:
            content = lb_file.read_text()

            # Search for ops.{operation} or related ops
            matches = re.finditer(
                rf'ops\.(load|store|{re.escape(pattern)})',
                content,
                re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                start = max(0, match.start() - 200)
                end = min(len(content), match.end() + 200)
                context = content[start:end]

                results.append({
                    'file': lb_file.name,
                    'match': match.group(0),
                    'context': context[:500]
                })

        except Exception:
            pass

    return results


def _search_loopbody_with_lines(pattern: str, debug_path: Path) -> list[dict]:
    """
    Search LoopBody IR files for pattern with line numbers.

    Returns results with line numbers and surrounding context.
    """
    results = []

    # Search ir_post_fusion*.txt files
    for lb_file in debug_path.glob("ir_post_fusion*.txt"):
        try:
            lines = lb_file.read_text().splitlines()

            # Search each line for the pattern
            pattern_re = re.compile(pattern, re.IGNORECASE)

            for line_num, line in enumerate(lines, start=1):
                if pattern_re.search(line):
                    # Get surrounding context
                    context_start = max(0, line_num - 4)
                    context_end = min(len(lines), line_num + 3)
                    context_lines = lines[context_start:context_end]

                    results.append({
                        'file': lb_file.name,
                        'line': line_num,
                        'match': line.strip(),
                        'context': context_lines
                    })

        except Exception:
            pass

    return results


def _search_triton(pattern: str, debug_path: Path) -> list[dict]:
    """Search Triton kernel files for operation."""
    results = []

    # Search output_code.py
    output_file = debug_path / "output_code.py"
    if output_file.exists():
        try:
            content = output_file.read_text()

            # Search for operation in Triton code
            import re
            # Look for triton operations or function calls
            matches = re.finditer(
                rf'(tl\.{re.escape(pattern)}|def.*{re.escape(pattern)})',
                content,
                re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                start = max(0, match.start() - 300)
                end = min(len(content), match.end() + 300)
                context = content[start:end]

                results.append({
                    'file': output_file.name,
                    'match': match.group(0),
                    'context': context[:500]
                })

        except Exception:
            pass

    return results


def _format_trace_results(stages: dict[str, list]) -> str:
    """Format trace results for display."""
    lines = []
    for stage_name, results in stages.items():
        lines.append(f"\n{stage_name}:")
        if results:
            lines.append(f"  Found in {len(results)} location(s)")
            # Show first result
            r = results[0]
            lines.append(f"  File: {r.get('file', 'unknown')}")
            lines.append(f"  Match: {r.get('match', 'N/A')}")
            if len(results) > 1:
                lines.append(f"  ... and {len(results) - 1} more matches")
        else:
            lines.append("  ✗ Not found")
    return "\n".join(lines)


def _format_search_results(results) -> str:
    """Format search results for display."""
    if isinstance(results, dict):
        # Multiple stages
        lines = []
        for stage, stage_results in results.items():
            lines.append(f"\n{stage}:")
            if stage_results:
                lines.append(f"  {len(stage_results)} match(es)")
                for r in stage_results[:3]:  # Show first 3
                    lines.append(f"    • {r.get('file')}: {r.get('match')[:60]}...")
            else:
                lines.append("  No matches")
        return "\n".join(lines)
    else:
        # Single stage
        if not results:
            return "No matches found"
        lines = [f"Found {len(results)} match(es):"]
        for r in results[:5]:  # Show first 5
            lines.append(f"\n  File: {r.get('file')}")
            lines.append(f"  Match: {r.get('match')}")
        if len(results) > 5:
            lines.append(f"\n... and {len(results) - 5} more")
        return "\n".join(lines)


def _generate_pipeline_summary(operation: str, stages: dict[str, list]) -> str:
    """Generate a summary of how operation flows through pipeline."""
    pipeline = []

    if stages["Dynamo (FX)"]:
        pipeline.append("Python source")
        pipeline.append("↓")
        pipeline.append("FX Graph (aten ops)")

    if stages["AOT (Joint)"]:
        pipeline.append("↓")
        pipeline.append("AOT Autograd (joint/partitioned)")

    if stages["Inductor (IR)"]:
        pipeline.append("↓")
        pipeline.append("Inductor IR (Pointwise/Reduction)")

    if stages["LoopBody"]:
        pipeline.append("↓")
        pipeline.append("LoopBody (ops.* operations)")

    if stages["Triton"]:
        pipeline.append("↓")
        pipeline.append("Triton Kernel (GPU code)")

    if not pipeline:
        return "Operation not found in any compilation stage"

    return "\n".join(pipeline)


def _format_single_stage_search(pattern: str, stage_name: str, results: list[dict]) -> str:
    """
    Format search results for a single stage.

    Shows file grouping with line numbers and context.
    """
    if not results:
        return f"""IR Search Results:

Pattern: {pattern}
Stage: {stage_name}

No matches found.
"""

    # Group results by file
    by_file = {}
    for result in results:
        filename = result['file']
        if filename not in by_file:
            by_file[filename] = []
        by_file[filename].append(result)

    # Format output
    lines = [
        "IR Search Results:",
        "",
        f"Pattern: {pattern}",
        f"Stage: {stage_name}",
        f"Found: {len(results)} match(es) in {len(by_file)} file(s)",
        ""
    ]

    # Show results grouped by file
    for filename, file_results in by_file.items():
        lines.append(f"--- {filename} ---")

        # Show up to 5 matches per file
        for i, result in enumerate(file_results[:5]):
            line_num = result.get('line', '?')
            match = result.get('match', '')

            lines.append(f"  Line {line_num}: {match[:80]}")

            # Show context if available
            context = result.get('context', [])
            if context and i < 2:  # Show context for first 2 matches
                lines.append("  Context:")
                for ctx_line in context[:3]:  # First 3 lines of context
                    lines.append(f"    {ctx_line[:70]}")

        if len(file_results) > 5:
            lines.append(f"  ... and {len(file_results) - 5} more matches")

        lines.append("")

    return "\n".join(lines)


def _format_multi_stage_search(pattern: str, results: dict[str, list[dict]]) -> str:
    """
    Format search results for multiple stages.

    Shows summary per stage with total counts.
    """
    total_matches = sum(len(stage_results) for stage_results in results.values())

    lines = [
        "IR Search Results (All Stages):",
        "",
        f"Pattern: {pattern}",
        f"Total Matches: {total_matches}",
        ""
    ]

    # Show results for each stage
    for stage_name, stage_results in results.items():
        lines.append(f"=== {stage_name} ===")

        if not stage_results:
            lines.append("  No matches")
            lines.append("")
            continue

        # Group by file
        by_file = {}
        for result in stage_results:
            filename = result['file']
            if filename not in by_file:
                by_file[filename] = []
            by_file[filename].append(result)

        lines.append(f"  Found: {len(stage_results)} match(es) in {len(by_file)} file(s)")

        # Show first few matches
        for filename, file_results in list(by_file.items())[:3]:
            lines.append(f"  • {filename}")
            for result in file_results[:2]:  # First 2 matches per file
                line_num = result.get('line', '?')
                match = result.get('match', '')
                lines.append(f"    Line {line_num}: {match[:60]}")

        if len(by_file) > 3:
            lines.append(f"  ... and {len(by_file) - 3} more files")

        lines.append("")

    return "\n".join(lines)
