# torch-compile-ai - Code Guidelines

## Purpose

MCP server for parsing torch.compile debug output. Provides 9 tools for analyzing FX graphs, fusion decisions, Triton codegen, and cross-stage transformations.

**For architecture and repository structure, see [REPO_ARCH.md](REPO_ARCH.md)**.

## Code Guidelines

**TDD**: All functions require a corresponding test. Python tests use Pytest. Follow Red-Green-Refactor phases.

**Code Style**:

Python: ruff · pyright type checker · Google-style docstrings (no types in docstring body) · module-level docstrings on every .py file · docstrings on every test function · functions <200 lines (prefer <100)

**Type Safety**:
- Use modern Python 3.10+ type hints (list, dict, set, tuple - not List, Dict, Set, Tuple)
- Full type annotations on all function signatures
- No `Any` unless absolutely necessary
- Specific exception types in except blocks (no bare `except`)

**Import Style**:
- Avoid local imports
- Use explicit module imports (e.g., `from analyzers.dynamo_parsers import parse_graph_breaks`)
- Import from modules, not package `__init__.py`

**Documentation**:
- Google-style docstrings with Args and Returns sections
- No type information in docstrings (types are in hints)
- Module-level docstrings explaining purpose
- Test docstrings explaining what is being tested

**Error Handling**:
- Avoid try/except unless for I/O operations or external API calls
- Let errors propagate for debugging
- Use specific FileNotFoundError, ValueError, etc. instead of generic Exception

**Code Organization**:
- One test file per source file (tests/analyzers/test_dynamo_parsers.py for analyzers/dynamo_parsers.py)
- Test classes named Test<FunctionName> (e.g., TestParseGraphBreaks)
- Keep functions focused and under 200 lines

## Testing

All parsers tested with realistic stdout/file content:

```bash
# All tests
pytest tests/analyzers/ -v

# Specific stage
pytest tests/analyzers/test_dynamo_parsers.py -v
```

**Test Requirements**:
- Use realistic stdout strings or file content (not synthetic data)
- Mock only external dependencies (never mock internal functions)
- Test both success and error cases
- Verify output format and key information

## Design Decisions

### Async Functions for Consistency
All parsers are async (`async def`) even though most don't do I/O, for consistency with MCP async tool handlers.

### String Returns
All parsers return formatted markdown strings (not JSON) for human-readable output in Claude conversations.

### Content Strings, Not File Paths
All parsers take content strings as input, never file paths. Caller is responsible for reading files.

**Why:** Separates file I/O from parsing logic, makes testing easier, works with both stdout and files.

### Test-First Development
Write test before implementation to validate requirements and ensure testability.

### Strong Typing Required
All function signatures fully typed. No `Any` without justification. Use modern type hints (list vs List).

### Simple, Focused Parsers
Each parser does one thing: parse a specific type of torch.compile output. No multi-file analysis, no complex diff logic.

**Removed complexity:**
- Multi-file diff analysis (analyze_partitioning, analyze_lowering)
- Cross-stage operation tracing (trace_operation, search_ir)
- Complex mutation detection (analyze_functionalization)

**Current approach:**
- Simple single-file/stdout parsing
- Clear mapping: 1 IR level = 1 parser
- Content strings only
