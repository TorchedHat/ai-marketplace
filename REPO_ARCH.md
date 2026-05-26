# Repository Architecture

## Overview

Claude Code plugin for PyTorch torch.compile debugging. Provides 8 skills, 5 agents, and MCP server integration for semantic API search across Dynamo, AOT Autograd, and Inductor compilation stages.

**Current Version**: Plugin-based architecture (migrated from symlink-based organization)

## Project Structure

```
torch-compile-ai/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
│
├── skills/                       # 8 debugging skills (auto-discovered)
│   ├── compile-overview/
│   │   └── SKILL.md             # Meta: Pipeline overview & routing
│   ├── compile-bisect/
│   │   └── SKILL.md             # Meta: Automated failure isolation
│   ├── pytorch-dynamo/
│   │   └── SKILL.md             # Implementation: Dynamo internals
│   ├── compile-trace-dynamo/
│   │   └── SKILL.md             # Tracing: Dynamo debugging
│   ├── pytorch-aot/
│   │   └── SKILL.md             # Implementation: AOT internals
│   ├── compile-trace-aot/
│   │   └── SKILL.md             # Tracing: AOT debugging
│   ├── pytorch-inductor/
│   │   └── SKILL.md             # Implementation: Inductor internals
│   └── compile-trace-inductor/
│       └── SKILL.md             # Tracing: Inductor debugging
│
├── agents/                       # 5 specialist agents (auto-discovered)
│   ├── coordinator.md           # Workflow orchestration
│   ├── dynamo-expert.md         # Dynamo specialist
│   ├── aot-expert.md            # AOT specialist
│   ├── inductor-expert.md       # Inductor specialist
│   └── bisector.md              # Bisection specialist
│
├── scripts/
│   └── ensure-setup.sh          # SessionStart hook: auto-install deps
│
├── .mcp.json                     # MCP server (steering) config
│
├── tests/                        # pytest tests
│
├── INSTALL.md                    # Installation guide
├── README.md                     # User documentation
└── CLAUDE.md                     # Code guidelines
```

### Legacy Directories (Preserved for Reference)

These directories contain the old organization but are **not used by the plugin**:

```
├── vertical-plugins/             # Old: Source organization by stage
├── agent-plugins/                # Old: Agent bundles with symlinks
├── coordinator/                  # Old: Coordinator vertical
├── managed-agent-cookbooks/      # Old: Deployment manifests
└── schemas/                      # Old: JSON schemas
```

These can be safely removed or kept for historical reference.

## Architecture

### Plugin System

**Auto-Discovery:**
- Skills in `skills/` directory are automatically discovered
- Agents in `agents/` directory are automatically discovered
- No explicit path configuration needed in plugin.json

**Installation:**
```bash
git clone https://github.com/morrison-turnansky/torch-compile-ai.git
cd torch-compile-ai
claude --plugin-dir .
```

**Auto-Setup (SessionStart Hook):**
- Installs `acp-steering-mcp` package
- Indexes PyTorch modules (dynamo, inductor, functorch)
- Runs via `scripts/ensure-setup.sh`

### Skill Organization

**Three Skill Types:**

1. **Meta Skills** - Workflow guidance
   - `compile-overview`: Pipeline overview & bisect-first workflow
   - `compile-bisect`: Automated failure isolation

2. **Tracing Skills** - User-level debugging
   - `compile-trace-dynamo`: TORCH_LOGS, graph breaks, FX graphs
   - `compile-trace-aot`: Functionalization, partitioning
   - `compile-trace-inductor`: Fusion decisions, scheduling, codegen

3. **Implementation Skills** - Contributor-level internals
   - `pytorch-dynamo`: VariableTracker, guards, bytecode tracing
   - `pytorch-aot`: vmap, functionalization, AOT pipeline
   - `pytorch-inductor`: Lowerings, scheduler, Triton templates

**Skill Naming:**
- When using `--plugin-dir`: `/skill-name` (no namespace)
- When installed via marketplace: `/torch-compile-ai:skill-name`

### Agent Organization

**Five Specialist Agents:**

1. **coordinator** - Orchestrates debugging workflow
   - Skills: compile-overview
   - Routes to specialist agents

2. **dynamo-expert** - Dynamo debugging
   - Skills: pytorch-dynamo, compile-trace-dynamo
   - Handles graph breaks, guards, VariableTracker

3. **aot-expert** - AOT debugging
   - Skills: pytorch-aot, compile-trace-aot
   - Handles functionalization, decomposition, partitioning

4. **inductor-expert** - Inductor debugging
   - Skills: pytorch-inductor, compile-trace-inductor
   - Handles lowerings, fusion, Triton codegen

5. **bisector** - Automated bisection
   - Skills: compile-bisect
   - Isolates failures to specific stage/operation

**Agent Definition Format:**
```yaml
---
name: agent-name
skills:
  - skill-one
  - skill-two
callable_agents:
  - other-agent
---
# Agent prompt...
```

### MCP Integration

**Steering Server:**
- Provides semantic search over PyTorch API documentation
- Indexed modules: torch._dynamo, torch._inductor, torch._functorch
- Auto-configured via `.mcp.json`

**Usage:**
```bash
# Skills can query API docs automatically
# Example: "What's the signature for Pointwise.__init__?"
```

## Design Principles

### 1. Direct Log Interpretation
Claude reads TORCH_LOGS output and debug files directly with skill guidance. No intermediate parsing - full context available for better analysis.

### 2. Auto-Discovery
Skills and agents are discovered automatically from standard directories. No manual configuration in plugin.json needed.

### 3. Portability
No hardcoded paths. PyTorch source location auto-detected or configurable via `PYTORCH_SRC` environment variable.

### 4. Self-Contained
All dependencies installed automatically via SessionStart hook. No manual setup required.

## Development Workflow

### Adding a New Skill

1. Create directory in `skills/`:
```bash
mkdir skills/my-new-skill
```

2. Add `SKILL.md` with frontmatter:
```markdown
---
name: my-new-skill
description: Brief description
---

# Skill Content
...
```

3. Restart Claude Code - skill auto-discovered

### Adding a New Agent

1. Create file in `agents/`:
```bash
touch agents/my-agent.md
```

2. Add YAML frontmatter:
```yaml
---
name: my-agent
skills:
  - skill-one
  - skill-two
---
# Agent prompt...
```

3. Restart Claude Code - agent auto-discovered

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_specific.py -v
```

### Updating Documentation

After making changes, update:
- `README.md` - User-facing documentation
- `INSTALL.md` - Installation instructions
- `REPO_ARCH.md` - This file (architecture)
- `CLAUDE.md` - Code guidelines

## Migration History

### v1.0.0 - Plugin Architecture (Current)
- ✅ Migrated from symlink-based to plugin-based
- ✅ Removed setup.sh (replaced with SessionStart hook)
- ✅ Removed all symlinks
- ✅ Simplified to auto-discovery
- ✅ Made fully portable (no hardcoded paths)

### Previous - Symlink Architecture
- Skills in `vertical-plugins/` as source of truth
- Symlinks from `agent-plugins/*/skills/` to vertical plugins
- Symlinks from `.claude/skills/` to vertical plugins
- Manual `setup.sh` to create symlinks
- Complex multi-directory organization

## Future Enhancements

### Marketplace Distribution
Once repo is public:
```bash
claude plugin marketplace add https://raw.githubusercontent.com/.../marketplace.json
claude plugin install torch-compile-ai
```

### Additional Skills
Potential future skills:
- `compile-trace-pre-grad`: Pre-grad pass debugging
- `compile-trace-post-grad`: Post-grad pass debugging
- `pytorch-fx`: FX graph manipulation

### Additional Agents
Potential future agents:
- `performance-optimizer`: Performance analysis specialist
- `memory-debugger`: Memory optimization specialist

## Troubleshooting

### Skills Not Appearing

Check structure:
```bash
# Verify plugin.json exists
ls .claude-plugin/plugin.json

# Verify skills exist
ls skills/*/SKILL.md

# Should show 8 files
```

### MCP Server Errors

Check steering installation:
```bash
which acp-steering-mcp
# or
uv pip install git+https://github.com/ambient-code/steering.git
```

### Indexing Failed

Check PyTorch source:
```bash
# Set env var if not in standard location
export PYTORCH_SRC=/path/to/pytorch

# Verify
ls $PYTORCH_SRC/torch/_dynamo
```

## References

- **Plugin Documentation**: https://code.claude.com/docs/en/plugins
- **Skill Documentation**: https://code.claude.com/docs/en/skills
- **MCP Documentation**: https://code.claude.com/docs/en/mcp
