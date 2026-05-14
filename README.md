# Foreman

An agentic CLI coding assistant with a Planner-Worker paradigm, async context compaction, and a dense JSON project "Brain."

## Quick Start

```bash
pip install -e .
foreman init        # Scaffold .foreman/ and build .foreman/context.json
foreman start       # Launch the TUI (hydrates context in background if empty)
```

## Commands (in TUI input bar)

| Command | Description |
|---------|-------------|
| `/model primary <model>` | Switch the primary (generation) model |
| `/model summary <model>` | Switch the summary (compaction) model |
| `/model` | Show current models |
| `/model list` | Show available model presets |
| `/plan <feature>` | Generate and stage an implementation plan |
| `/compact` | Force context compaction now (uses summary model) |
| `/compact-self` | Force self-compaction (primary model writes its own summary) |
| `/status` | Show session and token stats |
| `/keys` | Show API key status |
| `/config [key] [value]` | View/set configuration |
| `/context` | Refresh `.foreman/context.json` using the summary model |
| `/clear` | Clear chat display |
| `/new` | Start a new session |
| `/quit` | Exit Foreman |

## Model Switching

Both models can be quickly changed from the TUI:

```
/model primary gemini/gemini-2.5-pro
/model summary gemini/gemini-2.5-flash
/model list                    # see all presets
```

Or use any LiteLLM-compatible model string:
```
/model primary gpt-4o
/model primary deepseek/deepseek-chat
/model summary anthropic/claude-sonnet-4-20250514
```

## CLI Commands

```bash
foreman init                    # Create .foreman/ scaffold
foreman start                   # Launch the TUI
foreman start --repo /path      # Launch in a specific repo
foreman config                  # View all config
foreman config primary_model    # View a specific key
foreman config primary_model gemini/gemini-2.5-pro  # Set a key
foreman sessions list           # List saved sessions
foreman context build           # Build/refresh .foreman/context.json
```

## Architecture

```
foreman/
  brain/         Session CRUD, dense JSON context assembly/hydration
  tokens/        tiktoken counter, token budget management
  models/        Model profiles, LiteLLM router, API key management
  compact/       Token monitor, summarizer, context injection
  implement/     Plan generation, execution, SEARCH/REPLACE patcher
  tui/           Textual app, widgets, workers, CSS theme
  config.py      Configuration loading & defaults
  cli.py         CLI entry point
  init.py        Project scaffold (.foreman/)
```

## Dependencies

- `tiktoken` — local token counting
- `litellm` — multi-provider LLM routing
- `pydantic-ai` — structured output enforcement
- `textual` + `rich` — terminal UI
- `aiofiles` — async file I/O
- `tree-sitter` — AST-aware code patching
