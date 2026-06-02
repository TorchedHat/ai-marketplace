# ai-marketplace - Code Guidelines

## Purpose

Curated marketplace of AI tools, skills, and agents for PyTorch development workflows. Currently focused on torch.compile debugging with plans to expand to other PyTorch subsystems (torchvision, distributed, performance optimization).

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
