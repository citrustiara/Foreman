"""Async worker definitions for Foreman TUI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from foreman.brain.session import Session, Message, SessionStore
from foreman.compact.injector import compact_session, compact_session_self
from foreman.compact.summarizer import Summarizer
from foreman.compact.monitor import CompactMonitor
from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.models.router import ModelRouter
from foreman.tokens.counter import TokenCounter
from rich.markup import escape as rich_escape
from foreman.tokens.budget import TokenBudget
from foreman.tools.definitions import TOOL_DEFINITIONS, execute_tool
import subprocess
from foreman.implement.planner import Planner


if TYPE_CHECKING:
    from foreman.tui.app import ForemanApp

logger = logging.getLogger("foreman.tui.workers")

MAX_TOOL_ROUNDS = 25


def _extract_usage(response) -> tuple[int, int]:
    """Extract input/output token counts from a litellm response object."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage:
        if isinstance(usage, dict):
            return (
                int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            )
        return (
            int(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0),
            int(getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0),
        )
    return 0, 0


def _normalize_read_path(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_full_read(args: dict[str, Any]) -> bool:
    start_line = _to_int(args.get("start_line", args.get("offset", 1)), 1)
    max_bytes = _to_int(args.get("max_bytes", 2000), 2000)
    has_end_line = "end_line" in args and args.get("end_line") is not None
    has_legacy_limit = "limit" in args or "offset" in args
    return start_line <= 1 and max_bytes <= 0 and not has_end_line and not has_legacy_limit


def _resolve_tool_call_args(session: Session, tool_call_id: str) -> dict[str, Any] | None:
    for msg in reversed(session.messages):
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.get("id") != tool_call_id:
                continue
            fn = tc.get("function", {})
            if fn.get("name") != "read_file":
                return None
            try:
                return json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                return None
    return None


async def _replace_partial_read_with_note(
    app: "ForemanApp",
    session: Session,
    messages: list[dict],
    path: str,
) -> bool:
    path_key = _normalize_read_path(path)
    for msg in reversed(session.messages):
        if msg.role != "tool" or not msg.tool_call_id:
            continue
        args = _resolve_tool_call_args(session, msg.tool_call_id)
        if not args:
            continue
        prior_path = _normalize_read_path(str(args.get("path", "")))
        if prior_path != path_key:
            continue
        if _is_full_read(args):
            return False
        replacement = f"[Superseded partial read]\nA full read for {path} was fetched later."
        msg.content = replacement
        msg.token_count = app.token_counter.count_tokens(replacement, app.primary_profile.litellm_model)
        for local in messages:
            if local.get("role") == "tool" and local.get("tool_call_id") == msg.tool_call_id:
                local["content"] = replacement
                break
        session.total_tokens = sum(m.token_count for m in session.messages)
        await app.session_store.save(session)
        return True
    return False


async def run_chat(app: "ForemanApp", user_message: str) -> None:
    """Send a user message and run an agentic loop with tool calling."""
    session = app.current_session
    if session is None:
        return

    # Add user message to session
    msg = Message(role="user", content=user_message, token_count=0)
    msg.token_count = app.token_counter.count_tokens(user_message, app.primary_profile.litellm_model)
    await app.session_store.append_message(session, msg)

    app.chat_panel.add_user_message(user_message)
    app.update_status("\u25d0 Generating...")
    app.set_input_enabled(False)

    # Track cumulative token usage across all rounds
    total_input_tokens = 0
    total_output_tokens = 0
    # Deferred summaries: list of (tool_call_id, summary_text) to apply at turn end
    deferred_summaries: list[tuple[str, str]] = []

    try:
        # Build initial messages list
        messages = app.build_llm_messages(session)

        # Agentic loop: call LLM → handle tool calls → feed results back → repeat
        for round_num in range(MAX_TOOL_ROUNDS):
            if app._current_task and app._current_task.cancelled():
                app.chat_panel.add_system_message("[yellow]Interrupted.[/]")
                return

            current_assistant_widget = None
            current_assistant_text = ""

            def on_delta(delta: str):
                nonlocal current_assistant_widget, current_assistant_text
                current_assistant_text += delta
                if not current_assistant_widget:
                    current_assistant_widget = app.chat_panel.add_assistant_message(current_assistant_text)
                else:
                    app.chat_panel.update_assistant_message(current_assistant_widget, current_assistant_text)

            # Call LLM with tools
            response = await app.router.generate_with_tools(
                profile=app.primary_profile,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                delta_callback=on_delta,
                **app.llm_reasoning_kwargs(),
            )

            choice = response.choices[0]
            assistant_msg = choice.message
            tool_calls = getattr(assistant_msg, "tool_calls", None)

            # Use provider-reported usage only to avoid inflated cost accounting.
            round_input, round_output = _extract_usage(response)
            if round_input > 0 or round_output > 0:
                total_input_tokens += round_input
                total_output_tokens += round_output

            # Convert the assistant message to dict for the message list
            assistant_dict: dict = {"role": "assistant", "content": assistant_msg.content or ""}

            # Check if the model wants to call tools
            if not tool_calls:
                # No tool calls — model is done.
                text = assistant_msg.content or ""
                if text.strip() and not current_assistant_widget:
                    app.chat_panel.add_assistant_message(text)
                
                # Save assistant message to session
                output_tokens = app.token_counter.count_tokens(text, app.primary_profile.litellm_model)
                assistant_session_msg = Message(
                    role="assistant",
                    content=text,
                    token_count=output_tokens,
                    model=app.primary_profile.litellm_model,
                )
                await app.session_store.append_message(session, assistant_session_msg)
                break

            # Model wants to call tools
            # Add the assistant's tool_call message to the conversation
            assistant_dict["tool_calls"] = []
            for tc in tool_calls:
                assistant_dict["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            # Display which tools are being called
            tool_names = [tc.function.name for tc in tool_calls]
            app.chat_panel.add_system_message(f"\u2699 Calling: {', '.join(tool_names)}")

            # Save the assistant's tool_call message to the session
            # We estimate tokens for the content + tool calls
            tool_call_json = json.dumps(assistant_dict["tool_calls"])
            tc_tokens = app.token_counter.count_tokens(assistant_dict["content"] + tool_call_json, app.primary_profile.litellm_model)
            assistant_session_msg = Message(
                role="assistant",
                content=assistant_dict["content"],
                tool_calls=assistant_dict["tool_calls"],
                token_count=tc_tokens,
                model=app.primary_profile.litellm_model,
            )
            await app.session_store.append_message(session, assistant_session_msg)
            messages.append(assistant_dict)



            # Execute each tool call and add results
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    tool_result = f"Error: Invalid JSON arguments: {e}"
                    tool_args = {}

                # Display the tool call
                if app.expand_tool_output:
                    app.chat_panel.add_tool_call(f"⚙ {tool_name} {tc.function.arguments}")
                elif tool_name == "bash":
                    app.chat_panel.add_tool_call(f"$ {tool_args.get('command', '')}")
                elif tool_name == "read_file":
                    lines_hint = ""
                    sl = tool_args.get('start_line', 1)
                    el = tool_args.get('end_line')
                    mb = tool_args.get("max_bytes")
                    if el:
                        lines_hint = f" (L{sl}-{el})"
                    bytes_hint = f" [{mb}B]" if mb is not None else ""
                    app.chat_panel.add_tool_call(f"\u270e {tool_args.get('path', '')}{lines_hint}{bytes_hint}")
                elif tool_name == "write_file":
                    app.chat_panel.add_tool_call(f"\u270f {tool_args.get('path', '')}")
                elif tool_name == "search_file":
                    app.chat_panel.add_tool_call(f"\u2315 {tool_args.get('pattern', '')}")
                elif tool_name in ("summarize_last", "checkpoint_summary"):
                    pass  # handled below, no separate display

                # Execute the tool
                if tool_name == "checkpoint_summary":
                    summary = tool_args.get("summary", "")
                    await _handle_checkpoint_summary(app, session, messages, summary)
                    # MUST add tool result to history or LLM will error on next round
                    tr = "Context checkpointed."
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tr})
                    tr_tokens = app.token_counter.count_tokens(tr, app.primary_profile.litellm_model)
                    await app.session_store.append_message(session, Message(role="tool", content=tr, tool_call_id=tc.id, token_count=tr_tokens))
                    continue

                if tool_name == "summarize_last":
                    summary = tool_args.get("summary", "")
                    instant = tool_args.get("instant", False)
                    await _handle_summarize_last(
                        app, session, messages, summary, tc.id, instant, deferred_summaries
                    )
                    # MUST add tool result to history
                    tr = "Tool output summarized."
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tr})
                    tr_tokens = app.token_counter.count_tokens(tr, app.primary_profile.litellm_model)
                    await app.session_store.append_message(session, Message(role="tool", content=tr, tool_call_id=tc.id, token_count=tr_tokens))
                    continue

                if tool_name == "read_file" and _is_full_read(tool_args):
                    await _replace_partial_read_with_note(
                        app,
                        session,
                        messages,
                        str(tool_args.get("path", "")),
                    )

                tool_result = execute_tool(tool_name, tool_args, cwd=app.repo_root)
                if tool_name == "write_file":
                    app.register_file_modification(str(tool_args.get("path", "")), tool_result)

                # Display tool result with current expansion mode
                app.chat_panel.add_tool_result(app.format_tool_display(tool_result))

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

                # Save tool result to session
                tr_tokens = app.token_counter.count_tokens(tool_result, app.primary_profile.litellm_model)
                tool_session_msg = Message(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tc.id,
                    token_count=tr_tokens,
                )
                await app.session_store.append_message(session, tool_session_msg)


            app.update_status(f"\u25d0 Generating... (round {round_num + 2})")

            # Update stats and record cost incrementally
            if app.current_cost and (total_input_tokens > 0 or total_output_tokens > 0):
                app.current_cost.record(
                    model=app.primary_profile.litellm_model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost_per_1m_input=app.primary_profile.cost_per_1m_input,
                    cost_per_1m_output=app.primary_profile.cost_per_1m_output,
                )
                # Reset accumulators so we don't double-count in next round
                total_input_tokens = 0
                total_output_tokens = 0
            
            app.update_context_stats()

        else:
            app.chat_panel.add_system_message("[yellow]Reached maximum tool rounds. Stopping.[/]")

        # Apply any deferred summaries now that the turn is over
        if deferred_summaries:
            await _apply_deferred_summaries(app, session, messages, deferred_summaries)
            deferred_summaries.clear()


        # Check if compaction needed
        app.compact_monitor.check(session, app.primary_profile.litellm_model)
        if app.compact_monitor.should_compact:
            app.update_status("\u25d0 Compacting...")
            await run_compact(app)

        app.update_context_stats()
        app.update_status("\u25cb Ready")

    except asyncio.CancelledError:
        app.chat_panel.add_system_message("[yellow]Cancelled.[/]")
        app.update_status("\u25cb Cancelled")
    except Exception as e:
        logger.error("Chat failed: %s", e)
        app.chat_panel.add_error(str(e))
        app.update_status("\u2717 Error")
    finally:
        app.set_input_enabled(True)
        app._current_task = None


async def run_compact(app: "ForemanApp") -> None:
    """Run compaction using the summary model."""
    session = app.current_session
    if session is None:
        return

    try:
        event = await compact_session(
            session=session,
            store=app.session_store,
            summarizer=app.summarizer,
            token_counter=app.token_counter,
            model=app.primary_profile.litellm_model,
            recent_to_keep=app.config.recent_messages_to_keep,
            summary_model=app.secondary_profile,
            compactions_dir=app.foreman_dir / "compactions",
        )
        app.compact_monitor.reset()

        app.chat_panel.add_compact_event(
            event.tokens_before,
            event.tokens_after,
            event.messages_archived,
        )
        app.update_context_stats()
        app.update_status("\u25cb Ready (compacted)")

    except asyncio.CancelledError:
        app.chat_panel.add_system_message("[yellow]Compaction cancelled.[/]")
    except Exception as e:
        logger.error("Compact failed: %s", e)
        app.chat_panel.add_error(f"Compact failed: {e}")


async def run_compact_self(app: "ForemanApp") -> None:
    """Run self-compaction — primary model writes its own summary."""
    session = app.current_session
    if session is None:
        return

    try:
        event = await compact_session_self(
            session=session,
            store=app.session_store,
            summarizer=app.summarizer,
            token_counter=app.token_counter,
            primary_model=app.primary_profile,
            model=app.primary_profile.litellm_model,
            recent_to_keep=app.config.recent_messages_to_keep,
            compactions_dir=app.foreman_dir / "compactions",
        )
        app.compact_monitor.reset()

        app.chat_panel.add_compact_event(
            event.tokens_before,
            event.tokens_after,
            event.messages_archived,
            label="Self-Compacted",
        )
        app.update_context_stats()
        app.update_status("\u25cb Ready (self-compacted)")

    except asyncio.CancelledError:
        app.chat_panel.add_system_message("[yellow]Self-compaction cancelled.[/]")
    except Exception as e:
        logger.error("Self-compact failed: %s", e)
        app.chat_panel.add_error(f"Self-compact failed: {e}")


async def run_plan(app: "ForemanApp", feature_description: str) -> None:
    """Generate a plan and stage it for /approve."""
    session = app.current_session
    if session is None:
        return

    app.chat_panel.add_system_message(f"[bold]Planning:[/] {rich_escape(feature_description)}")
    app.update_status("⟐ Planning...")

    try:
        # 1. Load dense project context
        architecture = ""
        context_path = app.foreman_dir / "context.json"
        if context_path.exists():
            text = context_path.read_text(encoding="utf-8")
            if text.strip():
                architecture = f"### context.json\n```json\n{text}\n```\n"

        # 2. Generate directory tree (Windows-friendly version)
        try:
            # Try git ls-files first
            tree_output = subprocess.run(
                ["git", "ls-files"],
                capture_output=True, text=True, cwd=app.repo_root, timeout=5
            )
            if tree_output.returncode == 0:
                directory_tree = tree_output.stdout[:5000]
            else:
                # Fallback to simple listing
                directory_tree = "Git not found or not a repo. Skipping tree."
        except Exception:
            directory_tree = "Error generating directory tree."

        # 3. Generate plan
        planner = Planner(app.router)
        plan = await planner.generate_plan(
            feature_description=feature_description,
            architecture=architecture,
            directory_tree=directory_tree,
            primary_model=app.primary_profile,
            reasoning_effort=app.reasoning_effort_api(),
        )

        # 4. Display plan
        plan_md = planner.plan_to_markdown(plan)
        app.chat_panel.add_assistant_message(plan_md)
        app.chat_panel.add_system_message(
            "[bold]Type [/][cyan]/approve[/][bold] to execute, [/][cyan]/reject[/][bold] to cancel.[/]"
        )

        # 5. Wait for approval (stored in app state)
        app._pending_plan = plan
        app._pending_architecture = architecture
        app._pending_directory_tree = directory_tree
        app.update_status("⟐ Awaiting approval...")

    except Exception as e:
        logger.error("Plan generation failed: %s", e)
        app.chat_panel.add_error(f"Plan generation failed: {e}")
        app.update_status("✗ Error")


async def run_implement(app: "ForemanApp", feature_description: str) -> None:
    """Backward-compatible alias for older callers."""
    await run_plan(app, feature_description)

async def run_handoff(app: "ForemanApp") -> None:
    """Summarize session and start a new one."""
    session = app.current_session
    if session is None or not session.messages:
        app.chat_panel.add_system_message("Nothing to hand off.")
        return

    app.update_status("⟐ Creating handoff summary...")
    try:
        # Use Summarizer to compress all non-system messages
        non_system = [m for m in session.messages if m.role != "system"]
        messages_for_summary = [
            {"role": m.role, "content": m.content} for m in non_system
        ]

        # Use self_compress for better quality (primary model knows the context)
        summary = await app.summarizer.self_compress(
            messages_for_summary,
            primary_model=app.primary_profile,
        )

        # Display summary and start new session
        app.chat_panel.add_system_message("✅ Handoff summary created. Starting new session...")
        app.update_status("○ Ready (handoff complete)")

        # Trigger new session with this content
        await app.action_new_session_with_content(f"## HANDOFF SUMMARY\n\n{summary}")

    except Exception as e:
        logger.error("Handoff failed: %s", e)
        app.chat_panel.add_error(f"Handoff failed: {e}")
async def _handle_checkpoint_summary(app: ForemanApp, session: Session, messages: list[dict], summary: str) -> None:
    """Special tool handler for incremental context summarization."""
    # 1. Find last user message index in session
    last_user_idx = -1
    for i in range(len(session.messages) - 1, -1, -1):
        if session.messages[i].role == "user":
            last_user_idx = i
            break
            
    if last_user_idx == -1:
        return

    # 2. Truncate session messages to keep user message
    session.messages = session.messages[:last_user_idx + 1]
    
    # 3. Add the summary as a new assistant message
    tokens = app.token_counter.count_tokens(summary, app.primary_profile.litellm_model)
    summary_msg = Message(
        role="assistant",
        content=f"[Checkpoint Summary]\n{summary}",
        token_count=tokens,
        model=app.primary_profile.litellm_model
    )
    session.messages.append(summary_msg)
    
    # Recalculate total tokens
    session.total_tokens = sum(m.token_count for m in session.messages)
    await app.session_store.save(session)
    
    # 4. Update the local 'messages' list for the current LLM round
    local_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            local_user_idx = i
            break
            
    if local_user_idx != -1:
        new_messages = messages[:local_user_idx + 1]
        new_messages.append({"role": "assistant", "content": summary_msg.content})
        messages[:] = new_messages # In-place update
        
    app.chat_panel.add_system_message("[bold cyan]\u2728 Context Checkpointed:[/] History since last user message summarized.")


async def _handle_summarize_last(
    app: ForemanApp,
    session: Session,
    messages: list[dict],
    summary: str,
    this_tc_id: str,
    instant: bool,
    deferred_summaries: list[tuple[str, str]],
) -> None:
    """Handle the summarize_last tool call.

    Finds the most recent non-summarize_last tool result in history and either:
    - Instantly replaces it in both session and active messages (instant=True)
    - Queues it for deferred replacement at turn end (instant=False)
    """
    # Find the last actual tool result (not this summarize_last call itself)
    last_tool_call_id: str | None = None
    for msg in reversed(session.messages):
        if msg.role == "tool" and msg.tool_call_id != this_tc_id:
            last_tool_call_id = msg.tool_call_id
            break

    if not last_tool_call_id:
        # No prior tool result found; just show info
        app.chat_panel.add_system_message("[dim]summarize_last: no prior tool result to summarize.[/]")
        return

    formatted = f"[Summary]\n{summary}"
    app.chat_panel.add_system_message(
        f"[{'bold cyan' if instant else 'dim'}]\u270d {'Instant' if instant else 'Deferred'} Summary:[/] {summary[:120]}{'...' if len(summary) > 120 else ''}"
    )

    if instant:
        # Apply immediately to session
        tokens = app.token_counter.count_tokens(formatted, app.primary_profile.litellm_model)
        for msg in session.messages:
            if msg.role == "tool" and msg.tool_call_id == last_tool_call_id:
                msg.content = formatted
                msg.token_count = tokens
                break
        session.total_tokens = sum(m.token_count for m in session.messages)
        await app.session_store.save(session)

        # Apply immediately to active messages list
        for m in messages:
            if m.get("role") == "tool" and m.get("tool_call_id") == last_tool_call_id:
                m["content"] = formatted
                break
    else:
        # Queue for deferred application at turn end
        deferred_summaries.append((last_tool_call_id, formatted))


async def _apply_deferred_summaries(
    app: ForemanApp,
    session: Session,
    messages: list[dict],
    deferred_summaries: list[tuple[str, str]],
) -> None:
    """Apply all queued deferred summaries to the session at turn end."""
    if not deferred_summaries:
        return

    summary_map = dict(deferred_summaries)  # tool_call_id -> summary

    changed = False
    for msg in session.messages:
        if msg.role == "tool" and msg.tool_call_id in summary_map:
            new_content = summary_map[msg.tool_call_id]
            tokens = app.token_counter.count_tokens(new_content, app.primary_profile.litellm_model)
            msg.content = new_content
            msg.token_count = tokens
            changed = True

    if changed:
        session.total_tokens = sum(m.token_count for m in session.messages)
        await app.session_store.save(session)
        app.chat_panel.add_system_message(
            f"[dim]\u2713 Applied {len(deferred_summaries)} deferred tool summar{'y' if len(deferred_summaries) == 1 else 'ies'} to history.[/]"
        )
