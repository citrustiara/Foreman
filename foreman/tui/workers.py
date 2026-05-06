"""Async worker definitions for Foreman TUI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from foreman.brain.session import Session, Message, SessionStore
from foreman.compact.injector import compact_session, compact_session_self
from foreman.compact.summarizer import Summarizer
from foreman.compact.monitor import CompactMonitor
from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.models.router import ModelRouter
from foreman.tokens.counter import TokenCounter
from foreman.tokens.budget import TokenBudget
from foreman.tools.definitions import TOOL_DEFINITIONS, execute_tool

if TYPE_CHECKING:
    from foreman.tui.app import ForemanApp

logger = logging.getLogger("foreman.tui.workers")

MAX_TOOL_ROUNDS = 25


def _extract_usage(response) -> tuple[int, int]:
    """Extract input/output token counts from a litellm response object."""
    usage = getattr(response, "usage", None)
    if usage:
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    return 0, 0


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

    # Track cumulative token usage across all rounds
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        # Build initial messages list
        messages = app.build_llm_messages(session)

        # Agentic loop: call LLM → handle tool calls → feed results back → repeat
        for round_num in range(MAX_TOOL_ROUNDS):
            if app._current_task and app._current_task.cancelled():
                app.chat_panel.add_system_message("[yellow]Interrupted.[/]")
                return

            # Call LLM with tools
            response = await app.router.generate_with_tools(
                profile=app.primary_profile,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

            # Extract real token usage from the response
            round_input, round_output = _extract_usage(response)
            total_input_tokens += round_input
            total_output_tokens += round_output

            choice = response.choices[0]
            assistant_msg = choice.message

            # Convert the assistant message to dict for the message list
            assistant_dict: dict = {"role": "assistant", "content": assistant_msg.content or ""}

            # Check if the model wants to call tools
            tool_calls = getattr(assistant_msg, "tool_calls", None)

            if not tool_calls:
                # No tool calls — model is done. Display the text response and exit loop.
                text = assistant_msg.content or ""
                if text.strip():
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
            messages.append(assistant_dict)

            # Display which tools are being called
            tool_names = [tc.function.name for tc in tool_calls]
            app.chat_panel.add_system_message(f"\u2699 Calling: {', '.join(tool_names)}")

            # Execute each tool call and add results
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    tool_result = f"Error: Invalid JSON arguments: {e}"
                    tool_args = {}

                # Display the tool call
                if tool_name == "bash":
                    app.chat_panel.add_system_message(f"$ {tool_args.get('command', '')}")
                elif tool_name == "read_file":
                    app.chat_panel.add_system_message(f"\u270e {tool_args.get('path', '')}")
                elif tool_name == "write_file":
                    app.chat_panel.add_system_message(f"\u270f {tool_args.get('path', '')}")

                # Execute the tool
                tool_result = execute_tool(tool_name, tool_args, cwd=app.repo_root)

                # Display truncated result
                display_result = tool_result
                if len(display_result) > 500:
                    display_result = display_result[:500] + "\n... (truncated in display)"
                app.chat_panel.add_system_message(display_result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            app.update_status(f"\u25d0 Generating... (round {round_num + 2})")

        else:
            app.chat_panel.add_system_message("[yellow]Reached maximum tool rounds. Stopping.[/]")

        # Track cost using real usage numbers
        if app.current_cost and (total_input_tokens > 0 or total_output_tokens > 0):
            app.current_cost.record(
                model=app.primary_profile.litellm_model,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost_per_1m_input=app.primary_profile.cost_per_1m_input,
                cost_per_1m_output=app.primary_profile.cost_per_1m_output,
            )

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


async def run_implement(app: "ForemanApp", feature_description: str) -> None:
    ...

async def run_handoff(app: "ForemanApp") -> None:
    """Summarize session and start a new one."""
    session = app.current_session
    if session is None or not session.messages:
        app.chat_panel.add_system_message("Nothing to hand off.")
        return

    app.update_status("\u25d0 Creating handoff summary...")
    try:
        # 1. Summarize entire history
        from foreman.compact.summarizer import SUMMARIZE_PROMPT
        history = "\n".join([f"{m.role}: {m.content}" for m in session.messages])
        prompt = f"TRANSCRIPT FOR HANDOFF:\n\n{history}\n\nTask: Summarize the current development state, architectural decisions made, and pending tasks into a single concise handoff message for a fresh session."

        summary = await app.router.generate(
            profile=app.secondary_profile,
            messages=[
                {"role": "system", "content": "You are a senior architect preparing a handoff for another AI agent."},
                {"role": "user", "content": prompt}
            ]
        )

        # 2. Start new session
        app.chat_panel.add_system_message("\u2705 Handoff summary created. Starting new session...")
        await app.action_new_session_with_content(f"## HANDOFF SUMMARY\n\n{summary}")
        app.update_status("\u25cb New session (handoff)")

    except Exception as e:
        logger.error("Handoff failed: %s", e)
        app.chat_panel.add_error(f"Handoff failed: {e}")
