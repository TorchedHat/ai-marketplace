# torch-compile-ai - Code Guidelines

## Purpose

Multi-agent system for PyTorch torch.compile development and debugging. Provides stage-specific skills, steering MCP server for API documentation, and structured agent definitions.

**For architecture and repository structure, see [REPO_ARCH.md](REPO_ARCH.md)**.

## Code Guidelines

**Code Style**:

Python: ruff · pyright type checker · Google-style docstrings (no types in docstring body) · module-level docstrings on every .py file · docstrings on every test function · functions <200 lines (prefer <100)

**Type Safety**:
- Use modern Python 3.10+ type hints (list, dict, set, tuple - not List, Dict, Set, Tuple)
- Full type annotations on all function signatures
- No `Any` unless absolutely necessary
- Specific exception types in except blocks (no bare `except`)

**Import Style**:
- Avoid local imports
- Use explicit module imports
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
- One test file per source file
- Test classes named Test<FunctionName>
- Keep functions focused and under 200 lines

## Testing

Multi-agent scenarios and skill tests:

```bash
# All tests
pytest tests/ -v

# Specific tests
pytest tests/multi-agent/ -v
```

**Test Requirements**:
- Use realistic torch.compile output
- Mock only external dependencies
- Test both success and error cases

## Design Decisions

### Direct Log Interpretation
Claude reads TORCH_LOGS output and debug files directly using skill guidance. No intermediate parsing - full context available for better analysis.

### Skill-Based Architecture
Skills provide structured guidance for debugging each compilation stage (Dynamo, AOT, Inductor) rather than automated parsing.

### Steering MCP Integration
API documentation and semantic search over PyTorch modules (dynamo, functorch, inductor) via steering MCP server.
