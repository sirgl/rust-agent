# Rust Agent

An autonomous agent powered by LLMs (Anthropic Claude / DeepSeek) with ACP (Agent Client Protocol) support, built-in tools, MCP client, and subagent orchestration.

## Quick Start

### 1. API Key

Create a `token.properties` file in the project root (already in `.gitignore`):

```properties
# For Anthropic (default):
KEY=sk-ant-...

# For DeepSeek (optional):
DEEPSEEK_KEY=sk-...
```

Alternatively, set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export DEEPSEEK_API_KEY=sk-...
```

### 2. Interactive REPL (CLI)

```bash
# Release build (faster runtime):
./run-repl.sh

# Debug build (faster compilation):
./run-repl.sh --debug
```

### 3. ACP Server Mode

Start the ACP WebSocket server:

```bash
cargo run -p acp-agent
```

The server listens on the port configured in the environment or defaults.

## Configuration

### Model Selection

Set the model via environment variables (precedence: `ACP_MODEL` > `DEEPSEEK_MODEL` > `ANTHROPIC_MODEL`):

```bash
# DeepSeek
export ACP_MODEL=deepseek-v4-flash

# Or via token.properties:
ACP_MODEL=deepseek-v4-flash
```

### LLM Backend Selection

The agent automatically selects the LLM backend based on available keys:

| Backend | Env var | `token.properties` key | Default model |
|---------|---------|-----------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `KEY` | `claude-sonnet-4-20250514` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_KEY` | `deepseek-chat` |

You can also explicitly set the backend via `ACP_LLM_BACKEND`:

```bash
export ACP_LLM_BACKEND=anthropic        # Anthropic only
export ACP_LLM_BACKEND=deepseek         # DeepSeek only
export ACP_LLM_BACKEND=anthropic+deepseek # both (default)
```

### Other Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_BASE_URL` | Custom Anthropic API endpoint |
| `ANTHROPIC_EFFORT` | Claude effort level (`low`/`medium`/`high`) |
| `ANTHROPIC_THINKING` | Enable/disable thinking (`0`/`false` to disable) |
| `ANTHROPIC_THINKING_BUDGET` | Thinking budget in tokens |
| `ANTHROPIC_PROMPT_CACHING` | Enable/disable prompt caching (`0`/`false` to disable) |
| `DEEPSEEK_BASE_URL` | Custom DeepSeek API endpoint |
| `DEEPSEEK_EFFORT` | DeepSeek effort level |
| `DEEPSEEK_THINKING` | Enable/disable DeepSeek thinking |
| `DEEPSEEK_MAX_TOKENS` | Max output tokens for DeepSeek |
| `ACP_AGENT_NAME` | Advertised agent name |
| `ACP_AGENT_SYSTEM_PROMPT` | Custom system prompt |
| `ACP_SESSION_DIR` | Directory for session persistence |
| `ACP_TOKEN_PROPERTIES_PATH` | Override path to `token.properties` |

## Project Structure

```
crates/
  agent-core/      — Core turn engine, update sink, and client access traits
  turn-anthropic/  — Anthropic/DeepSeek LLM backend (Messages API)
  turn-replay/     — Deterministic replay backend for testing
  tools-builtin/   — Built-in filesystem and terminal tools
  mcp-client/      — MCP protocol client
  subagents/       — Subagent spawning and management
  orchestrated/    — Orchestrated execution mode
  llm-inspector/   — LLM request/response inspector
  acp-agent/       — ACP WebSocket server
  cli-agent/       — Interactive CLI REPL
```

## Development

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run a specific crate
cargo run -p cli-agent
cargo run -p acp-agent
```

## License

Apache-2.0