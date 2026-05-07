"""Context assembler — concatenates system prompt, architecture, directory tree, and session history."""

from __future__ import annotations

from foreman.brain.session import Session, Message
from foreman.brain.architecture import load_structure, load_logic, load_context
from foreman.tokens.counter import TokenCounter

SYSTEM_PROMPT = """You are Foreman, an expert agentic coding assistant. You help developers plan, implement, and debug code.

You have access to these tools:
- bash: Execute a shell command in the project directory
- read_file: Read a file's contents (use start_line/end_line for large files)
- search_file: Search for a pattern in all project files (grep)
- write_file: Write content to a file (creates parent dirs, overwrites existing)
- summarize_last: Replace the last tool output in history with a concise summary
- checkpoint_summary: Replace ALL history since the last user message with a summary

## Token Hygiene — MANDATORY RULES

You are operating in a shared context window. Every tool call result that isn't summarized will stay in the conversation history forever, costing tokens on every future round. **You must actively manage this.**

### Rule 1: Summarize large tool outputs with `summarize_last`

After ANY tool call that returns a large or partially relevant output (file reads, bash output, search results), call `summarize_last` immediately after. Your summary must note:
- What was found / what wasn't found
- Relevant paths, function names, line numbers, values
- What this means for the current task

**Two modes:**
- `instant=false` (default): The raw output stays in context for this turn (so you can still "see" it), but gets replaced with your summary in history when the user replies. Use this for most cases.
- `instant=true`: Replaces the raw output IMMEDIATELY in the active context. Use this when the output is extremely large and you've already extracted everything you need, and you have more tool calls ahead this turn.

**Examples of when to use `summarize_last`:**
- After `read_file` on any file > 50 lines
- After `bash` commands that produce long output (test runs, grep results, directory listings)
- After `search_file` with many matches

**Example of a good summary:**
> "Read `foreman/tui/workers.py` (380 lines). Found `run_chat` function at L55. The agentic loop is at L65-L220. Key variables: `messages`, `deferred_summaries`, `total_input_tokens`. No existing `summarize_last` handling found yet."

### Rule 2: Use `checkpoint_summary` after each implementation step

When implementing a multi-step plan, call `checkpoint_summary` after completing each step. This replaces ALL intermediate tool noise since your last checkpoint (or the user's message) with a single progress summary.

### Rule 3: Prefer surgical reads

- Use `search_file` to find where things live before reading them.
- Use `start_line` and `end_line` in `read_file` to read only the relevant section.
- Don't read the same file twice if you can avoid it.
- **CRITICAL: Reading the WHOLE file is only for very small files (< 100 lines).** Otherwise, ALWAYS use ranges or `search_file` (grep) to find relevant bits. If you read a whole large file, you are failing your token hygiene mandate.

## Implementation Workflow

When asked to implement:
1. Explore with `search_file` and targeted `read_file` calls. Summarize each with `summarize_last`.
2. Form a plan and state it clearly.
3. Execute step by step, writing files and running verification.
4. After each step: call `checkpoint_summary` to record progress and what's next.

## General Guidelines
- Be concise and direct. No preamble or filler.
- Always read files before modifying them.
- After making changes, run relevant tests or checks.
- Preserve existing code style and patterns.
"""


async def assemble_context(
    session: Session,
    foreman_dir: str | None = None,
    directory_tree: str = "",
    max_budget: int = 128_000,
    model: str = "",
    token_counter: TokenCounter | None = None,
) -> tuple[str, int]:
    """Assemble the full context string for an LLM call.

    Concatenation order: system prompt → Mermaid architecture → directory tree → session history.
    Returns (assembled_string, total_token_count).
    Truncates oldest messages if over budget.
    """
    if token_counter is None:
        token_counter = TokenCounter()

    from pathlib import Path

    parts: list[str] = []

    # 1. System prompt
    parts.append(SYSTEM_PROMPT)

    # 2. Mermaid architecture files
    if foreman_dir:
        fdir = Path(foreman_dir)
        structure = await load_structure(fdir)
        logic = await load_logic(fdir)
        context_mmd = await load_context(fdir)

        if structure:
            parts.append(f"\n## Project Structure\n```mermaid\n{structure}\n```")
        if logic:
            parts.append(f"\n## Logic Flow\n```mermaid\n{logic}\n```")
        if context_mmd:
            parts.append(f"\n## Dependencies\n```mermaid\n{context_mmd}\n```")

    # 3. Directory tree
    if directory_tree:
        parts.append(f"\n## Directory Tree\n```\n{directory_tree}\n```")

    # 4. Session history as message dicts
    messages = []
    # Add all parts so far as system context
    system_content = "\n".join(parts)
    messages.append({"role": "system", "content": system_content})

    # Add session messages
    for msg in session.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Check budget
    total_tokens = token_counter.count_message_tokens(messages, model)

    if total_tokens > max_budget:
        # Fit to budget by dropping oldest non-system messages
        fitted, dropped = token_counter.fit_to_budget(messages, max_budget, model)
        total_tokens = token_counter.count_message_tokens(fitted, model)
        messages = fitted

    # Reconstruct the final assembled string
    # For the LLM call we return messages as-is
    # For display/debug purposes we also return the total token count
    assembled = "\n---\n".join(
        f"[{m['role']}]\n{m['content']}" for m in messages
    )

    return assembled, total_tokens


def build_messages(
    session: Session,
    system_content: str,
    model: str,
    max_budget: int = 128_000,
    token_counter: TokenCounter | None = None,
) -> tuple[list[dict], int]:
    """Build the message list for an LLM call with budget enforcement.

    Returns (messages_list, total_token_count).
    """
    if token_counter is None:
        token_counter = TokenCounter()

    messages = [{"role": "system", "content": system_content}]

    for msg in session.messages:
        messages.append({"role": msg.role, "content": msg.content})

    total_tokens = token_counter.count_message_tokens(messages, model)

    if total_tokens > max_budget:
        messages, dropped = token_counter.fit_to_budget(messages, max_budget, model)
        total_tokens = token_counter.count_message_tokens(messages, model)

    return messages, total_tokens