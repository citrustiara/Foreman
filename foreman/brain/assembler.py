"""Context assembler — concatenates system prompt, architecture, directory tree, and session history."""

from __future__ import annotations

from foreman.brain.session import Session, Message
from foreman.brain.architecture import load_structure, load_logic, load_context
from foreman.tokens.counter import TokenCounter

SYSTEM_PROMPT = """You are Foreman, an expert agentic coding assistant. You help developers plan, implement, and debug code.

You have access to these tools:
- bash: Execute a shell command in the project directory
- read_file: Read a file's contents
- write_file: Write content to a file (creates parent dirs, overwrites existing)

Guidelines:
- Be concise and direct. No preamble or filler.
- When exploring a project, use read_file and bash to understand the codebase before answering.
- When writing code, prefer write_file for new files or full rewrites.
- When asked to implement, explore the codebase first, then create a structured plan, then implement.
- Preserve existing code style and patterns.
- Always read files before modifying them.
- After making changes, run relevant tests or checks to verify correctness.
"""


async def assemble_context(
    session: Session,
    foreman_dir: str | None = None,
    directory_tree: str = "",
    max_budget: int = 128_000,
    model: str = "gpt-4o",
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
    model: str = "gpt-4o",
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