---
name: plugin-writer
description: Create new Claude Code plugins from scratch with proper directory structure (.claude-plugin/plugin.json, skills/, agents/), or add existing plugins to marketplace.json. Use when creating a plugin, setting up plugin metadata, configuring MCP servers, or adding plugins to a marketplace.
---

# Plugin Creator

Create well-structured Claude Code plugins with proper metadata, directory structure, and marketplace integration.

## When to use this Skill

Use this Skill when:
- Creating a new Claude Code plugin from scratch
- Setting up plugin.json and marketplace.json files
- Adding existing plugins to a marketplace
- Configuring MCP servers for a plugin
- Organizing skills and agents into a plugin

## Quick start

**Create a new plugin:**
```bash
# 1. Create plugin directory structure
mkdir -p my-plugin/.claude-plugin
mkdir -p my-plugin/{skills,agents,hooks,scripts}

# 2. Create plugin.json
# 3. Create marketplace.json (if building a marketplace)
# 4. Add skills and agents
# 5. Configure MCP servers (optional)
```

**Add existing plugin to marketplace:**
```bash
# Edit .claude-plugin/marketplace.json
# Add plugin entry with source URL
```

## Plugin Structure

A complete plugin follows this structure:

```
my-plugin/
├── .claude-plugin/          # Plugin metadata (required)
│   ├── plugin.json          # Plugin configuration
│   └── marketplace.json     # Marketplace metadata (optional)
├── skills/                  # User-invocable skills
│   └── skill-name/
│       └── SKILL.md         # Skill definition
├── agents/                  # Specialized AI agents
│   └── agent-name.md        # Agent definition
├── hooks/                   # Lifecycle hooks (optional)
│   └── hooks.json           # Hook configuration
├── scripts/                 # Setup scripts (optional)
│   └── ensure-setup.sh      # Auto-setup on SessionStart
├── settings.json            # MCP server config (optional)
└── pyproject.toml           # Package metadata (optional)
```

## Instructions

### Part 1: Creating a new plugin from scratch

#### Step 1: Gather plugin information

Ask the user for:
- **Plugin name**: Lowercase, hyphens only (e.g., `torch-compile`, `data-tools`)
- **Display name**: Human-readable (e.g., `"PyTorch AI Marketplace"`)
- **Description**: Brief description of what the plugin provides
- **Author name**: Plugin author
- **Version**: Semantic version (default: `"1.0.0"`)
- **Skills location**: Path to skills directory (default: `"./skills/"`)
- **Agents**: List of agent file paths (e.g., `["./agents/expert.md"]`)
- **MCP servers**: Any MCP servers to configure (optional)

#### Step 2: Create directory structure

```bash
mkdir -p <plugin-name>/.claude-plugin
mkdir -p <plugin-name>/skills
mkdir -p <plugin-name>/agents
mkdir -p <plugin-name>/hooks
mkdir -p <plugin-name>/scripts
```

#### Step 3: Create plugin.json

Create `.claude-plugin/plugin.json` with this structure:

```json
{
  "name": "plugin-name",
  "displayName": "Plugin Display Name",
  "version": "1.0.0",
  "description": "Plugin description",
  "skills": "./skills/",
  "agents": [
    "./agents/agent-name.md"
  ],
  "author": {
    "name": "Author Name"
  },
  "homepage": "https://github.com/username/repo",
  "license": "BSD-3-Clause",
  "keywords": [
    "keyword1",
    "keyword2"
  ],
  "mcpServers": {
    "server-name": {
      "command": "command-name",
      "args": [],
      "env": {}
    }
  }
}
```

**Field requirements:**
- `name`: Lowercase, hyphens only, must be unique
- `displayName`: Human-readable plugin name
- `version`: Semantic version (e.g., `"1.0.0"`)
- `description`: Brief description (< 1024 chars)
- `skills`: Path to skills directory (auto-discovery)
- `agents`: Array of agent file paths (explicit paths)
- `author.name`: Plugin author
- `mcpServers`: Optional MCP server configurations

**Skills discovery:**
- If `skills: "./skills/"` is set, all skills in that directory are auto-discovered
- Skills must have `SKILL.md` files with proper frontmatter
- No need to list individual skills in plugin.json

**Agents:**
- Must explicitly list agent file paths
- Agent files must have YAML frontmatter with `name` field
- Agents are NOT auto-discovered

**MCP Servers:**
- Optional - only include if plugin needs MCP integration
- Each server needs: `command`, optional `args`, optional `env`
- The `command` should be the executable name or path

#### Step 4: Create settings.json (if using MCP servers)

If the plugin uses MCP servers, create `settings.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "command-name",
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

**When to create settings.json:**
- Plugin provides or requires MCP server integration
- Need to set environment variables for MCP servers
- Want to provide default MCP configuration

**Difference from plugin.json mcpServers:**
- `plugin.json mcpServers`: Declares what MCP servers the plugin provides/requires
- `settings.json mcpServers`: Provides default configuration for those servers
- Both can coexist and complement each other

#### Step 5: Add skills and agents

**Skills:**
- User should already have skills they want to add
- Copy skill directories to `skills/` folder
- Verify each skill has proper `SKILL.md` with frontmatter
- Skills are auto-discovered from the skills directory

**Agents:**
- User should already have agent definitions they want to add
- Copy agent `.md` files to `agents/` folder
- Add agent paths to `plugin.json` agents array
- Agents must be explicitly listed (no auto-discovery)

**Verification:**
```bash
# Check skills have SKILL.md
ls skills/*/SKILL.md

# Check agents exist
ls agents/*.md

# Verify frontmatter in skills
head -10 skills/*/SKILL.md

# Verify frontmatter in agents
head -10 agents/*.md
```

#### Step 6: Create pyproject.toml (optional)

If the plugin is a Python package:

```toml
[project]
name = "plugin-name"
version = "1.0.0"
description = "Plugin description"
authors = [{name = "Author Name"}]
license = {text = "BSD-3-Clause"}
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Part 2: Adding existing plugin to marketplace

If building a marketplace of plugins, create `marketplace.json`.

#### Step 1: Create marketplace.json structure

Create `.claude-plugin/marketplace.json`:

```json
{
  "name": "marketplace-name",
  "displayName": "Marketplace Display Name",
  "description": "Marketplace description",
  "owner": {
    "name": "Owner Name",
    "email": "owner@example.com"
  },
  "plugins": [
    {
      "name": "plugin-name",
      "description": "Plugin description",
      "category": "development",
      "source": {
        "source": "url",
        "url": "https://github.com/username/repo.git",
        "ref": "main"
      }
    }
  ]
}
```

**Field requirements:**
- `name`: Marketplace identifier (lowercase, hyphens)
- `displayName`: Human-readable marketplace name
- `description`: Brief marketplace description
- `owner.name`: Marketplace owner
- `owner.email`: Contact email
- `plugins`: Array of plugin entries

**Plugin entry fields:**
- `name`: Plugin name (must match plugin's plugin.json name)
- `description`: Brief plugin description
- `category`: Plugin category (e.g., `"development"`, `"productivity"`)
- `source.source`: Always `"url"` for git repositories
- `source.url`: Git repository URL
- `source.ref`: Git branch/tag (e.g., `"main"`, `"v1.0.0"`)

#### Step 2: Add plugin to marketplace

To add a new plugin to an existing marketplace.json:

```json
{
  "plugins": [
    // ... existing plugins ...
    {
      "name": "new-plugin-name",
      "description": "What this plugin does",
      "category": "development",
      "source": {
        "source": "url",
        "url": "https://github.com/username/new-plugin.git",
        "ref": "main"
      }
    }
  ]
}
```

**Categories:**
- `"development"`: Development tools and workflows
- `"productivity"`: Productivity and automation
- `"debugging"`: Debugging and diagnostics
- `"testing"`: Testing and validation
- `"documentation"`: Documentation generation
- Custom categories are allowed

## Examples

### Example 1: Create a simple plugin

**User request:** "Create a plugin called 'data-tools' with skills for CSV and JSON processing"

**Steps:**

1. Create directory structure:
```bash
mkdir -p data-tools/.claude-plugin
mkdir -p data-tools/{skills,agents,hooks,scripts}
```

2. Create `data-tools/.claude-plugin/plugin.json`:
```json
{
  "name": "data-tools",
  "displayName": "Data Processing Tools",
  "version": "1.0.0",
  "description": "Tools for CSV, JSON, and data file processing",
  "skills": "./skills/",
  "agents": [],
  "author": {
    "name": "Your Name"
  },
  "license": "MIT",
  "keywords": ["data", "csv", "json", "processing"]
}
```

3. Copy user's skills to `data-tools/skills/`
4. Verify skills have SKILL.md files

### Example 2: Plugin with MCP server

**User request:** "Create a plugin with an MCP server for API documentation lookup"

1. Create plugin structure (same as above)

2. Create `plugin.json` with mcpServers:
```json
{
  "name": "api-docs",
  "displayName": "API Documentation",
  "version": "1.0.0",
  "description": "Semantic API documentation search",
  "skills": "./skills/",
  "agents": ["./agents/api-expert.md"],
  "author": {
    "name": "Your Name"
  },
  "mcpServers": {
    "api-lookup": {
      "command": "api-lookup-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

3. Create `settings.json`:
```json
{
  "mcpServers": {
    "api-lookup": {
      "command": "api-lookup-mcp",
      "env": {
        "API_INDEX_PATH": "~/.api-docs/index"
      }
    }
  }
}
```

### Example 3: Add plugin to marketplace

**User request:** "Add the torchtalk plugin to my marketplace"

Edit `.claude-plugin/marketplace.json`:

```json
{
  "name": "ai-marketplace",
  "displayName": "PyTorch AI Marketplace",
  "description": "Curated AI tools for PyTorch development",
  "owner": {
    "name": "Your Name",
    "email": "you@example.com"
  },
  "plugins": [
    {
      "name": "torchtalk",
      "description": "MCP server for PyTorch codebase analysis",
      "category": "development",
      "source": {
        "source": "url",
        "url": "https://github.com/TorchedHat/torchtalk.git",
        "ref": "main"
      }
    }
  ]
}
```

## Best practices

1. **Plugin naming:**
   - Use lowercase with hyphens
   - Be descriptive but concise
   - Avoid generic names

2. **Skills organization:**
   - One skill per capability
   - Use `./skills/` for auto-discovery
   - Verify all skills have SKILL.md

3. **Agent organization:**
   - Explicitly list all agents in plugin.json
   - Use descriptive agent names
   - Keep agent files in `agents/` directory

4. **MCP servers:**
   - Only include if actually needed
   - Document required environment variables
   - Test MCP server connectivity

5. **Marketplace entries:**
   - Use specific descriptions
   - Categorize appropriately
   - Pin to stable refs (not `main` for production)

6. **Version management:**
   - Start at `1.0.0`
   - Follow semantic versioning
   - Update version in plugin.json when making changes

## Validation checklist

Before finalizing a plugin:

- [ ] Directory structure created (`.claude-plugin/`, `skills/`, `agents/`)
- [ ] `plugin.json` has all required fields
- [ ] `plugin.json` name is lowercase, hyphens only
- [ ] Skills directory path set correctly in plugin.json
- [ ] All agents explicitly listed in plugin.json agents array
- [ ] Skills have proper SKILL.md files with frontmatter
- [ ] Agents have proper frontmatter with name field
- [ ] MCP servers configured (if needed)
- [ ] `settings.json` created (if using MCP servers)
- [ ] Marketplace.json updated (if adding to marketplace)
- [ ] Plugin source URL is correct (if in marketplace)

## Troubleshooting

**Skills not discovered:**
- Verify `skills: "./skills/"` is set in plugin.json
- Check that each skill has SKILL.md file
- Verify SKILL.md has proper YAML frontmatter

**Agents not found:**
- Check agents are explicitly listed in plugin.json agents array
- Verify agent paths are correct (relative to plugin root)
- Ensure agent files have YAML frontmatter with `name` field

**MCP server not connecting:**
- Verify command is in PATH or use absolute path
- Check environment variables in settings.json
- Test MCP server command manually

**Plugin not in marketplace:**
- Verify marketplace.json syntax
- Check plugin name matches plugin.json name
- Ensure source URL is accessible

## Output format

When creating a plugin, I will:

1. Ask clarifying questions about plugin requirements
2. Create directory structure
3. Generate plugin.json with proper metadata
4. Create settings.json if MCP servers are needed
5. Verify skills and agents are properly configured
6. Add to marketplace.json if requested
7. Provide validation checklist
8. Show testing instructions

The result will be a complete, working Claude Code plugin following best practices.
