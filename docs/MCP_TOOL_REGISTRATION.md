# MCP Tool Registration - Standard Practices

Research on the standard way to register MCP (Model Context Protocol) tools as of 2026.

## Official MCP Specification

The [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25) defines the authoritative requirements for tool registration.

### Tool Capabilities

Servers that support tools **MUST** declare the `tools` capability:

```json
{
  "capabilities": {
    "tools": {
      "listChanged": true
    }
  }
}
```

`listChanged` indicates whether the server will emit notifications when the list of available tools changes.

### Tool Discovery (tools/list)

The [specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) defines the `tools/list` endpoint:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "cursor": "optional-cursor-value"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "title": "Weather Information Provider",
        "description": "Get current weather information for a location",
        "inputSchema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name or zip code"
            }
          },
          "required": ["location"]
        }
      }
    ],
    "nextCursor": "next-page-cursor"
  }
}
```

### Tool Structure

Each tool definition **MUST** include:

- **name**: Unique identifier (1-128 chars, case-sensitive, alphanumeric + `_`, `-`, `.`)
- **description**: Human-readable description of functionality
- **inputSchema**: JSON Schema defining expected parameters
  - **MUST** be a valid JSON Schema object (not null)
  - Defaults to JSON Schema 2020-12 if no `$schema` field present
  - For tools with no parameters: `{ "type": "object", "additionalProperties": false }`

Optional fields:
- **title**: Human-readable name for display
- **icons**: Array of icons for UI
- **outputSchema**: JSON Schema for structured output validation
- **annotations**: Properties describing tool behavior
- **execution**: Execution-related properties (e.g., `taskSupport`)

### Tool Invocation (tools/call)

Tools are invoked via the `tools/call` endpoint:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "New York"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in New York:\nTemperature: 72°F"
      }
    ],
    "isError": false
  }
}
```

## Python SDK Implementation

The [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) provides two approaches:

### 1. Low-Level Server Pattern (What We Use)

For maximum control, use the `Server` class with explicit decorators:

```python
from mcp.server import Server
from mcp.types import TextContent, Tool

app = Server("debug-tracer")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="parse_graph_breaks",
            description="Parse TORCH_LOGS graph_breaks output",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "TORCH_LOGS=graph_breaks output"
                    }
                },
                "required": ["log_content"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations."""
    if name == "parse_graph_breaks":
        result = await parse_graph_breaks(arguments["log_content"])
        return [TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")
```

**Pros:**
- Full control over tool definitions
- Explicit schema specification
- Easy to understand data flow

**Cons:**
- More boilerplate
- Manual schema maintenance

### 2. FastMCP Pattern (Higher-Level)

FastMCP provides automatic schema generation from type hints:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("debug-tracer")

@mcp.tool()
async def parse_graph_breaks(log_content: str) -> str:
    """Parse TORCH_LOGS graph_breaks output (Dynamo stage)."""
    # Implementation
    return result
```

**Pros:**
- Less boilerplate
- Automatic schema from type hints
- Cleaner code

**Cons:**
- Less control over schema
- Magic behavior via decorators

## Our Current Implementation

We use the **low-level Server pattern** in `server.py`:

```python
from mcp.server import Server
from mcp.types import TextContent, Tool

app = Server("debug-tracer")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="parse_dynamo_guards",
            description="Parse guard failures from logs",
            inputSchema={...}
        ),
        # ... 12 more tools
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Route to appropriate analyzer
    if name == "parse_dynamo_guards":
        result = await dynamo.parse_dynamo_guards(arguments["log_path"])
    # ... handle other tools
    return [TextContent(type="text", text=result)]
```

### Why This Pattern?

1. **Explicit control** - We need precise schema definitions for torch.compile paths
2. **Clear routing** - Easy to see which tool maps to which analyzer
3. **Flexible returns** - Can return multiple content types (text, images, etc.)
4. **Documentation** - Schema shows exactly what each tool expects

## Compliance Check

✅ **Our implementation is fully compliant:**

- ✅ Declares `tools` capability (implicitly via Server class)
- ✅ Implements `tools/list` endpoint (via `@app.list_tools()`)
- ✅ Implements `tools/call` endpoint (via `@app.call_tool()`)
- ✅ Returns `list[Tool]` with proper schema
- ✅ Tool names follow naming conventions (alphanumeric + underscore)
- ✅ All tools have `name`, `description`, `inputSchema`
- ✅ `inputSchema` uses JSON Schema format
- ✅ Returns `list[TextContent]` from tool calls
- ✅ Handles errors properly (ValueError for unknown tools)

## Best Practices

Based on the [specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools):

### Tool Naming

**SHOULD:**
- Be 1-128 characters
- Use only: A-Z, a-z, 0-9, `_`, `-`, `.`
- Be unique within server
- Be case-sensitive

**Examples:**
- ✅ `parse_dynamo_guards`
- ✅ `analyze_fx_graph`
- ✅ `find_graph_breaks`
- ❌ `parse guards` (no spaces)
- ❌ `analyze,fx,graph` (no commas)

### Input Schema

**MUST:**
- Be valid JSON Schema object
- Not be `null`
- Have `type: "object"` as root

**For no parameters:**
```json
{
  "type": "object",
  "additionalProperties": false
}
```

**For required parameters:**
```json
{
  "type": "object",
  "properties": {
    "log_path": {
      "type": "string",
      "description": "Path to debug log file"
    }
  },
  "required": ["log_path"]
}
```

### Error Handling

Two types of errors:

**1. Protocol Errors** (structural issues):
```python
raise ValueError(f"Unknown tool: {name}")
```
Returns JSON-RPC error to client.

**2. Tool Execution Errors** (business logic):
```python
return [TextContent(
    type="text",
    text="File not found: debug.log"
)], isError=True
```
Returns error in tool result (LLM can see and retry).

### Security

Servers **MUST:**
- Validate all tool inputs
- Implement proper access controls
- Rate limit tool invocations
- Sanitize tool outputs

Clients **SHOULD:**
- Prompt for user confirmation on sensitive operations
- Show tool inputs before calling server
- Implement timeouts
- Log tool usage

## Recommendations for Our Server

Our implementation is solid, but we could consider:

### Optional Enhancements

1. **Add `title` field** for better UI display:
```python
Tool(
    name="parse_dynamo_guards",
    title="Dynamo Guard Parser",  # NEW
    description="Parse guard failures from debug logs",
    inputSchema={...}
)
```

2. **Add `outputSchema`** for structured validation:
```python
Tool(
    name="find_graph_breaks",
    description="Find all graph breaks in logs",
    inputSchema={...},
    outputSchema={  # NEW
        "type": "object",
        "properties": {
            "total_breaks": {"type": "integer"},
            "break_locations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
)
```

3. **Support structured output** for machine-readable results:
```python
return [TextContent(
    type="text",
    text=json.dumps(result)  # For backwards compatibility
)], structuredContent=result  # NEW: structured output
```

4. **Implement `listChanged` notification** if tools can change dynamically:
```python
app = Server("debug-tracer")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [...], listChanged=True  # NEW
```

## Migration to FastMCP?

**Recommendation: Stay with low-level pattern**

Reasons:
- Current implementation is clear and maintainable
- Explicit schemas are important for documentation
- No significant benefit from FastMCP for our use case
- Migration would add risk without clear payoff

FastMCP is better for:
- Simple tools with basic parameters
- Rapid prototyping
- When type hints fully describe the schema

Our tools have:
- Complex path parameters (torch_compile_debug/run_*/...)
- Detailed descriptions needed
- Specific file format requirements

## References

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP Tools Documentation](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Registry](https://registry.modelcontextprotocol.io/)

## Summary

**Our implementation follows MCP best practices:**
- ✅ Uses standard `@app.list_tools()` and `@app.call_tool()` decorators
- ✅ Returns proper Tool objects with valid schemas
- ✅ Handles errors appropriately
- ✅ Tool names follow conventions
- ✅ Input schemas are valid JSON Schema

**No changes required**, but optional enhancements like `title` and `outputSchema` could improve the developer experience.
