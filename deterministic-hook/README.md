# Deterministic Hook System

This plugin provides a deterministic hook system for Claude Code that enables automatic skill activation suggestions and guardrails based on user prompts.

## What's Included

- **Hook Infrastructure**: Production-ready UserPromptSubmit hook for automatic skill activation
- **Skill Developer Guide**: Comprehensive documentation for creating and managing Claude Code skills
- **Reference Documentation**: Complete guides on trigger types, hook mechanisms, patterns, and troubleshooting

## Directory Structure

```
deterministic-hook/
├── .claude-plugin/
│   └── plugin.json              # Plugin configuration
├── hooks/
│   ├── hooks.json               # Hook configuration
│   ├── skill_activation_prompt.py  # Prompt analyzer
│   └── README.md                # Hook documentation
├── skills/
│   └── skill-developer/         # Skill creation guide
│       ├── SKILL.md             # Main skill documentation
│       ├── HOOK_MECHANISMS.md   # Hook system reference
│       ├── TRIGGER_TYPES.md     # Trigger pattern guide
│       ├── PATTERNS_LIBRARY.md  # Common patterns
│       ├── ADVANCED.md          # Advanced techniques
│       ├── SKILL_RULES_REFERENCE.md  # skill-rules.json reference
│       └── TROUBLESHOOTING.md   # Debugging guide
└── README.md                    # This file
```

## How It Works

When the deterministic-hook plugin is installed:

1. **Automatic registration**: The `hooks.json` file registers the `UserPromptSubmit` hook with Claude Code
2. **Prompt analysis**: When users submit prompts, `skill_activation_prompt.py` analyzes them against patterns in `skill-rules.json`
3. **Smart suggestions**: If keywords or patterns match, Claude sees a suggestion to use the relevant skill
4. **Skill activation**: Users can then invoke the suggested skill using the Skill tool

## Files

### hooks.json

Defines the hook configuration for the plugin:

```json
{
  "description": "Automatic skill activation suggestions based on user prompts",
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/hooks/skill_activation_prompt.py",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

**Key features:**
- Uses `${CLAUDE_PLUGIN_ROOT}` for plugin-relative paths
- Fires on every user prompt submission
- 5-second timeout for fast response

### skill_activation_prompt.py

Python script that:
- Reads user prompt from stdin (JSON format)
- Loads trigger rules from `skill-rules.json`
- Matches keywords and regex patterns
- Outputs suggestions in `additionalContext` format

**No dependencies required** - uses only Python 3 standard library.

### skill-rules.json

Defines trigger patterns for each skill:

```json
{
  "version": "1.0.0",
  "skills": {
    "skill-name": {
      "type": "domain",
      "enforcement": "suggest",
      "priority": "high",
      "promptTriggers": {
        "keywords": ["keyword1", "keyword2"],
        "intentPatterns": ["regex.*pattern"]
      }
    }
  }
}
```

**Priority levels:**
- **critical**: Required skills (shown with ⚠️)
- **high**: Strongly recommended (shown with 📚)
- **medium**: Suggested (shown with 💡)
- **low**: Optional (shown with 📌)

## Adding New Skill Triggers

To add activation triggers for a new skill:

1. **Add entry to skill-rules.json**:
   ```json
   {
     "skills": {
       "your-skill-name": {
         "type": "domain",
         "enforcement": "suggest",
         "priority": "high",
         "promptTriggers": {
           "keywords": ["keyword1", "keyword2"],
           "intentPatterns": ["pattern.*match"]
         }
       }
     }
   }
   ```


## Using the Skill Developer

The plugin includes the `skill-developer` skill which provides comprehensive guidance for:

- Creating new Claude Code skills
- Configuring skill-rules.json
- Understanding trigger patterns (keywords, intent patterns, file paths, content patterns)
- Working with hooks (UserPromptSubmit, PreToolUse)
- Implementing progressive disclosure
- Following the 500-line rule
- Debugging skill activation issues

Invoke with: `/skill-developer` or it auto-activates when you mention skill creation.

## Environment Variables

- `CLAUDE_PLUGIN_ROOT`: Set by Claude Code when running as plugin hook (points to plugin directory)
- `CLAUDE_PROJECT_DIR`: Fallback for local development (points to project directory)

## Installation

This plugin is part of the PyTorch AI Marketplace. It will be automatically available when you install the ai-marketplace plugin.

## References

- [Claude Code Hooks Documentation](http://code.claude.com/docs/en/hooks)
- See `skills/skill-developer/` for comprehensive skill development documentation
