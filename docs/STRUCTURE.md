# Plugin Structure

## Directory Layout

```
torch-compile-ai/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── .claude-marketplace/
│   └── marketplace.json         # Marketplace configuration
├── skills/                       # Auto-discovered skills
│   ├── compile-overview/
│   ├── compile-bisect/
│   ├── pytorch-dynamo/
│   ├── compile-trace-dynamo/
│   ├── pytorch-aot/
│   ├── compile-trace-aot/
│   ├── pytorch-inductor/
│   └── compile-trace-inductor/
├── agents/                       # Agent definitions
│   ├── coordinator.md
│   ├── dynamo-expert.md
│   ├── aot-expert.md
│   ├── inductor-expert.md
│   └── bisector.md
├── scripts/
│   └── ensure-setup.sh          # SessionStart hook
├── hooks.json                    # Hook definitions
├── tests/                        # pytest tests
├── docs/                         # Documentation
├── README.md                     # User guide
├── INSTALL.md                    # Installation instructions
├── REPO_ARCH.md                  # Architecture overview
└── CLAUDE.md                     # Code guidelines
```

## Key Files

### .claude-plugin/plugin.json
Plugin manifest with:
- Metadata (name, version, description)
- Skills path: `./skills/`
- Agents array: individual .md files
- Hooks path: `./hooks.json`
- MCP servers configuration

### hooks.json
SessionStart hook that runs `scripts/ensure-setup.sh` to:
- Install acp-steering-mcp package
- Index PyTorch modules (dynamo, inductor, functorch)

### skills/
Each skill is a subdirectory with a `SKILL.md` file containing:
- YAML frontmatter (name, description)
- Skill content (guidance, examples, patterns)

### agents/
Each agent is a .md file with:
- YAML frontmatter (name, skills, callable_agents)
- Agent prompt (identity, workflow, deliverables)

## Auto-Discovery

Claude Code automatically discovers:
- Skills in `./skills/` directory
- Agents listed in plugin.json `agents` array
- MCP servers in `mcpServers` config

No manual configuration needed beyond plugin.json.
