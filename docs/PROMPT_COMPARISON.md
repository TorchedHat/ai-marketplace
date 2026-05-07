# Prompt Size Comparison

## Standard vs Concise Versions

| Prompt | Standard | Concise | Reduction |
|--------|----------|---------|-----------|
| coordinator.md | 325 lines | 126 lines | 61% |
| dynamo-expert.md | 179 lines | 95 lines | 47% |
| inductor-expert.md | 229 lines | 131 lines | 43% |
| **Total** | **733 lines** | **352 lines** | **52%** |

## When to Use Each

### Standard Prompts (`.md`)
**Use when:**
- Learning the system (detailed examples help)
- Complex multi-domain tasks requiring full context
- Debugging routing issues
- Training new team members

**Advantages:**
- Complete examples showing expected output format
- Detailed guidelines for quality reports
- Comprehensive error handling scenarios
- Full workflow descriptions

**Disadvantages:**
- Larger context window usage (~733 lines)
- May slow down simple tasks with unnecessary detail

### Concise Prompts (`-concise.md`)
**Use when:**
- Production deployment (minimize context)
- Simple, focused tasks
- Budget-conscious API usage
- Fast iteration needed

**Advantages:**
- 52% smaller context footprint
- Faster to parse and execute
- Lower API costs
- Streamlined decision making

**Disadvantages:**
- Less guidance for edge cases
- Fewer examples to reference
- May require more iterations for complex tasks

## Recommendation

**Start with concise** for production use. Switch to standard if:
- Routing accuracy drops below 80%
- Synthesis quality degrades
- Edge cases not handled well

## Context Window Impact

**Full multi-agent session (worst case):**

Standard prompts:
```
Coordinator: 325 lines
Dynamo expert: 179 lines  
Inductor expert: 229 lines
Total: 733 lines (~15-20KB)
```

Concise prompts:
```
Coordinator: 126 lines
Dynamo expert: 95 lines
Inductor expert: 131 lines
Total: 352 lines (~7-10KB)
```

**Savings: ~8-10KB per multi-agent session**

With average 50KB skill files, total context:
- Standard: 733 lines + 100KB skills = ~115-120KB
- Concise: 352 lines + 100KB skills = ~107-110KB

**Effective reduction: 7-9% on full session**

Note: Biggest context savings comes from not loading all skills upfront (60-70% reduction) rather than prompt size.

## JSON Structure Changes

Both versions use **flat JSON structure** for specialist outputs:

**Old (nested):**
```json
{
  "findings": {
    "key_insight": "...",
    "relevant_files": [...],
    "related_concepts": [...]
  },
  "performance_notes": "..."
}
```

**New (flat):**
```json
{
  "insight": "...",
  "files": [...],
  "concepts": [...],
  "perf": "..."
}
```

**Benefits:**
- Easier to parse and extract specific fields
- Smaller JSON payload
- Clearer field names (insight vs findings.key_insight)
- No nesting overhead

## Migration Guide

**From standard to concise:**

1. Replace prompt files:
   ```bash
   cp coordinator-concise.md coordinator.md
   cp dynamo-expert-concise.md dynamo-expert.md
   cp inductor-expert-concise.md inductor-expert.md
   ```

2. Test routing accuracy on existing tasks
3. Monitor synthesis quality
4. Adjust decision tree keywords if needed

**From concise back to standard:**

1. Restore original files (keep `-concise.md` as backup)
2. Review detailed examples for edge cases
3. Update team documentation

## Metrics to Track

When switching versions, monitor:
- **Routing accuracy** - % of correct specialist suggestions
- **Synthesis quality** - Coherent, actionable guidance
- **Context usage** - Lines/tokens per session
- **Task completion rate** - % of tasks successfully completed
- **Iteration count** - Average turns needed per task

Target: Concise should maintain 90%+ metrics of standard while using 50% less prompt context.
