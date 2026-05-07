"""Foreman TUI — main Textual application."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DirectoryTree, Footer, Header, Static

from foreman.brain.session import Session, SessionStore, Message
from foreman.brain.assembler import SYSTEM_PROMPT

from foreman.brain.architecture import load_structure, load_logic, load_context
from foreman.compact.monitor import CompactMonitor
from foreman.compact.summarizer import Summarizer
from foreman.config import ForemanConfig
from foreman.models.keys import load_env, key_status
from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.models.router import ModelRouter
from foreman.models.fetcher import fetch_openrouter_models, save_models, load_models
from foreman.tokens.budget import TokenBudget
from foreman.tokens.cost import CostTracker
from foreman.tokens.counter import TokenCounter
from foreman.tui.widgets import (
    ChatPanel,
    ContextStatsPanel,
    InputBar,
    ModelSelectorScreen,
    HelpScreen,
    SessionSelectorScreen,
)

from foreman.implement.planner import Planner, ImplementationPlan
from foreman.implement.executor import Executor
from foreman.implement.patcher import apply_all_patches


logger = logging.getLogger("foreman.tui.app")


def load_openrouter_models() -> list[dict]:
    """Load OpenRouter model list from global cache."""
    return load_models()



class ForemanApp(App):
    """The Foreman agentic coding assistant TUI."""

    TITLE = "Foreman"
    CSS_PATH = "foreman.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "model_selector", "Models", show=True),
        Binding("ctrl+n", "new_session", "New Session", show=True),
        Binding("escape", "interrupt", "Cancel", show=False),
    ]

    def __init__(self, repo_root: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.repo_root = repo_root or Path.cwd()
        self.foreman_dir = self.repo_root / ".foreman"

        # Load config
        self.config = ForemanConfig.load(self.repo_root)

        # Initialize model profiles
        self._primary_model_str = self.config.primary_model
        self._secondary_model_str = self.config.secondary_model
        self.primary_profile = resolve_profile(self._primary_model_str)
        self.secondary_profile = resolve_profile(self._secondary_model_str)

        # Core components
        self.token_counter = TokenCounter()
        self.budget = TokenBudget.from_model(
            self._primary_model_str,
            output_reserve=self.config.output_reserve,
        )

        self.router = ModelRouter()
        self.summarizer = Summarizer(self.router, self.config.max_summary_tokens)
        self.compact_monitor = CompactMonitor(
            self.token_counter,
            self.budget,
            threshold=self.config.compact_threshold,
        )

        # Session state
        self.session_store = SessionStore(self.foreman_dir / "sessions")
        self.current_session: Session | None = None

        # Cost tracking
        self.cost_tracker = CostTracker(self.foreman_dir / "costs")
        self.current_cost = None

        # OpenRouter models
        self.openrouter_models = load_openrouter_models()


        # Widget references
        self.chat_panel: ChatPanel | None = None
        self.context_panel: ContextStatsPanel | None = None
        self.input_bar: InputBar | None = None

        # Current running task (for ESC cancellation)
        self._current_task: asyncio.Task | None = None

        # Step 1 State
        self._pending_plan: ImplementationPlan | None = None
        self._pending_architecture: str = ""
        self._pending_directory_tree: str = ""


    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(classes="sidebar"):
                yield DirectoryTree(self.repo_root, id="dir-tree")
            with Vertical(classes="main"):
                yield ChatPanel(id="chat-panel")
                yield InputBar(id="input-bar")
            with Vertical(classes="sidebar sidebar-right"):
                yield ContextStatsPanel(id="context-stats")
        yield Footer()

    async def on_mount(self) -> None:
        load_env(self.repo_root)

        self.chat_panel = self.query_one("#chat-panel", ChatPanel)
        self.context_panel = self.query_one("#context-stats", ContextStatsPanel)
        self.input_bar = self.query_one("#input-bar", InputBar)

        # Apply theme
        try:
            self.theme = self.config.theme
        except Exception:
            pass


        # Fetch OpenRouter models in background
        asyncio.create_task(self._refresh_models())
        asyncio.create_task(self._init_session())

        self.chat_panel.add_system_message(
            f"[bold]Foreman v0.1.0[/] \u2014 Agentic Coding Assistant\n\n"
            f"Primary: [cyan]{self.primary_profile.name}[/]\n"
            f"Summary: [cyan]{self.secondary_profile.name}[/]\n\n"
            f"Type [cyan]/help[/] for commands \u2022 [cyan]Ctrl+L[/] model selector \u2022 [cyan]Escape[/] cancel"
        )
        self._update_model_display()


    async def _refresh_models(self) -> None:
        """Fetch latest OpenRouter models if cache is stale or missing."""
        try:
            # Check if global cache exists
            from foreman.config import get_global_foreman_dir
            global_dir = get_global_foreman_dir()
            cache_path = global_dir / "openrouter_models.json"

            if not cache_path.exists():
                models = await fetch_openrouter_models()
                if models:
                    save_models(models)  # Saves to global by default
                    self.openrouter_models = models
                    
                    # RE-RESOLVE PROFILES to pick up new costs/context
                    self.primary_profile = resolve_profile(self._primary_model_str)
                    self.secondary_profile = resolve_profile(self._secondary_model_str)
                    self.update_context_stats()
                    self._update_model_display()

                    self.chat_panel.add_system_message(
                        f"[dim]Loaded {len(models)} OpenRouter models[/]"
                    )
        except Exception as e:
            logger.warning("Failed to refresh OpenRouter models: %s", e)

    async def _init_session(self) -> None:
        self.current_session = await self.session_store.create()
        self.current_cost = self.cost_tracker.get_or_create(self.current_session.session_id)
        self.update_context_stats()

    # ─── Model Switching ───────────────────────────────────────────

    def switch_primary_model(self, model_string: str) -> None:
        self._primary_model_str = model_string
        self.primary_profile = resolve_profile(model_string)
        self.config.primary_model = model_string
        self.budget = TokenBudget.from_model(model_string, output_reserve=self.config.output_reserve)
        self.compact_monitor = CompactMonitor(self.token_counter, self.budget, threshold=self.config.compact_threshold)
        self._update_model_display()
        self.chat_panel.add_system_message(f"Primary model: [cyan]{self.primary_profile.name}[/] ({model_string})")

    def switch_secondary_model(self, model_string: str) -> None:
        self._secondary_model_str = model_string
        self.secondary_profile = resolve_profile(model_string)
        self.config.secondary_model = model_string
        self._update_model_display()
        self.chat_panel.add_system_message(f"Summary model: [cyan]{self.secondary_profile.name}[/] ({model_string})")

    def _update_model_display(self) -> None:
        self.title = f"Foreman | {self.primary_profile.name}"

    # ─── Command Handling ──────────────────────────────────────────

    def on_input_bar_command_submitted(self, event: InputBar.CommandSubmitted) -> None:
        cmd = event.command
        args = event.args.strip()

        handlers = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "compact": self._cmd_compact,
            "compact-self": self._cmd_compact_self,
            "implement": self._cmd_implement,
            "clear": self._cmd_clear,
            "new": self._cmd_new,
            "status": self._cmd_status,
            "cost": self._cmd_cost,
            "keys": self._cmd_keys,
            "handoff": self._cmd_handoff,
            "config": self._cmd_config,
            "fetch": self._cmd_fetch,
            "approve": self._cmd_approve,
            "reject": self._cmd_reject,
            "resume": self._cmd_resume,
            "quit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            handler(args)
        else:
            self.chat_panel.add_error(f"Unknown command: /{cmd}. Type /help for available commands.")

    def on_input_bar_user_message(self, event: InputBar.UserMessage) -> None:
        if self.current_session is None:
            self.chat_panel.add_error("No active session. Use /new to create one.")
            return
        from foreman.tui.workers import run_chat
        self._current_task = asyncio.create_task(run_chat(self, event.content))

    # ─── Command Implementations ───────────────────────────────────

    def _cmd_help(self, args: str) -> None:
        self.push_screen(HelpScreen())

    def _cmd_model(self, args: str) -> None:
        if not args:
            self.chat_panel.add_system_message(
                f"[bold]Current Models[/]\n"
                f"  Primary: [cyan]{self.primary_profile.name}[/] ({self.primary_profile.litellm_model})\n"
                f"  Summary: [cyan]{self.secondary_profile.name}[/] ({self.secondary_profile.litellm_model})"
            )
            return

        parts = args.split()
        if parts[0] == "list":
            lines = []
            if self.openrouter_models:
                lines.append("[bold]Available Models (OpenRouter)[/]")
                for m in self.openrouter_models:
                    ctx = f"{m['context_window'] // 1000}K"
                    lines.append(f"  [cyan]openrouter/{m['id']}[/] \u2014 {m['name']} ({ctx})")
            else:
                lines.append("[yellow]No OpenRouter models cached. Run /fetch to load them.[/]")
            self.chat_panel.add_system_message("\n".join(lines))
            return

        if len(parts) < 2:
            self.chat_panel.add_error("Usage: /model <primary|summary> <model_id>")
            return

        target = parts[0].lower()
        model_string = " ".join(parts[1:])

        if target == "primary":
            self.switch_primary_model(model_string)
        elif target in ("summary", "secondary"):
            self.switch_secondary_model(model_string)
        else:
            self.chat_panel.add_error(f"Unknown target: {target}. Use 'primary' or 'summary'.")

    def _cmd_compact(self, args: str) -> None:
        from foreman.tui.workers import run_compact
        self._current_task = asyncio.create_task(run_compact(self))

    def _cmd_compact_self(self, args: str) -> None:
        from foreman.tui.workers import run_compact_self
        self._current_task = asyncio.create_task(run_compact_self(self))

    def _cmd_implement(self, args: str) -> None:
        if not args:
            # Fallback: Use last assistant message if available
            if self.current_session and self.current_session.messages:
                last_assistant = next((m for m in reversed(self.current_session.messages) if m.role == "assistant"), None)
                if last_assistant:
                    args = last_assistant.content
            
            if not args:
                self.chat_panel.add_error("/implement requires a feature description (or a prior AI recommendation)")
                return
        from foreman.tui.workers import run_implement
        self._current_task = asyncio.create_task(run_implement(self, args))

    def _cmd_approve(self, args: str) -> None:
        if not hasattr(self, '_pending_plan') or self._pending_plan is None:
            self.chat_panel.add_error("No pending plan to approve.")
            return
        self._current_task = asyncio.create_task(self._execute_approved_plan())

    def _cmd_reject(self, args: str) -> None:
        if not hasattr(self, '_pending_plan') or self._pending_plan is None:
            self.chat_panel.add_error("No pending plan to reject.")
            return
        self._pending_plan = None
        self._pending_architecture = ""
        self._pending_directory_tree = ""
        self.chat_panel.add_system_message("[yellow]Plan rejected.[/]")
        self.update_status("\u25cb Ready")

    async def _execute_approved_plan(self) -> None:
        """Execute the pending implementation plan."""
        plan = self._pending_plan
        architecture = self._pending_architecture
        
        # Clear pending state
        self._pending_plan = None
        self._pending_architecture = ""
        self._pending_directory_tree = ""

        self.chat_panel.add_system_message("[bold]Cleaning context and executing plan...[/]")
        self.update_status("\u25d0 Executing...")

        try:
            # 0. Clean context: Remove intermediate tool noise since the last user message
            if self.current_session:
                # Find last user message
                last_user_idx = -1
                for i in range(len(self.current_session.messages) -1, -1, -1):
                    if self.current_session.messages[i].role == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx != -1:
                    # Keep everything up to the last user message + the plan we just displayed
                    # Actually, we want to keep the session but maybe just "summarize" or truncate the noise.
                    # For now, let's just truncate the noise between the last user message and the plan.
                    self.current_session.messages = self.current_session.messages[:last_user_idx + 1]
                    
                    # Add the plan as a clean starting point for the executor
                    plan_md = Planner.plan_to_markdown(plan)
                    tokens = self.token_counter.count_tokens(plan_md, self.primary_profile.litellm_model)
                    plan_msg = Message(
                        role="assistant",
                        content=f"## APPROVED PLAN\n\n{plan_md}",
                        token_count=tokens,
                        model=self.primary_profile.litellm_model
                    )
                    self.current_session.messages.append(plan_msg)
                    self.current_session.total_tokens = sum(m.token_count for m in self.current_session.messages)
                    await self.session_store.save(self.current_session)

            # 1. Collect relevant file contents
            files = set()
            for step in plan.steps:
                files.update(step.files_to_modify)
                files.update(step.files_to_create)
            
            relevant_files = {}
            for file_path in files:
                full_path = self.repo_root / file_path
                if full_path.exists():
                    relevant_files[file_path] = full_path.read_text(encoding="utf-8")
                else:
                    relevant_files[file_path] = ""

            # 2. Execute plan via model
            executor = Executor(self.router)
            plan_md = Planner.plan_to_markdown(plan)
            model_output = await executor.execute_plan(
                plan_markdown=plan_md,
                relevant_files=relevant_files,
                architecture=architecture,
                primary_model=self.primary_profile,
            )

            # 3. Parse and apply patches
            results = apply_all_patches(model_output, self.repo_root)

            # 4. Display results
            lines = ["[bold]Execution Results:[/]\n"]
            success_count = 0
            for res in results:
                icon = "✅" if res.success else "❌"
                lines.append(f"{icon} {res.file_path}: {res.message}")
                if res.success:
                    success_count += 1

            self.chat_panel.add_system_message("\n".join(lines))

            if success_count < len(results):
                self.chat_panel.add_error("Some patches failed to apply. See below for raw output.")
                self.chat_panel.add_assistant_message("## RAW MODEL OUTPUT (for manual review)\n\n" + model_output)

            self.update_status("\u25cb Ready")

            # 5. Optional: Self-compress if session is active
            if self.current_session:
                 from foreman.tui.workers import run_compact_self
                 await run_compact_self(self)

        except Exception as e:
            logger.error("Execution failed: %s", e)
            self.chat_panel.add_error(f"Execution failed: {e}")
            self.update_status("\u2717 Error")

    def _cmd_handoff(self, args: str) -> None:
        from foreman.tui.workers import run_handoff
        self._current_task = asyncio.create_task(run_handoff(self))

    def _cmd_clear(self, args: str) -> None:
        if self.chat_panel:
            self.chat_panel.clear_chat()

    def _cmd_new(self, args: str) -> None:
        asyncio.create_task(self._init_session())
        if self.chat_panel:
            self.chat_panel.clear_chat()
            self.chat_panel.add_system_message("New session started.")

    def _cmd_resume(self, args: str) -> None:
        async def _do_resume():
            if not args:
                # Show session selector
                sessions = await self.session_store.list_sessions()
                if not sessions:
                    self.chat_panel.add_system_message("No past sessions found.")
                    return
                
                def on_session_selected(session_id: str | None) -> None:
                    if session_id:
                        asyncio.create_task(self.action_resume_session(session_id))

                self.push_screen(SessionSelectorScreen(sessions), on_session_selected)
            else:
                await self.action_resume_session(args)

        asyncio.create_task(_do_resume())

    async def action_resume_session(self, session_id: str) -> None:
        """Load and switch to a past session."""
        self.update_status("\u25d0 Resuming...")
        try:
            session = await self.session_store.load(session_id)
            self.current_session = session
            self.current_cost = self.cost_tracker.get_or_create(session_id)
            
            if self.chat_panel:
                self.chat_panel.clear_chat()
                self.chat_panel.add_system_message(f"Resumed session: [cyan]{session_id}[/]")
                
                # Re-populate chat display
                for msg in session.messages:
                    if msg.role == "user":
                        self.chat_panel.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        if msg.content.strip():
                            self.chat_panel.add_assistant_message(msg.content)
                        if msg.tool_calls:
                            names = [tc["function"]["name"] for tc in msg.tool_calls]
                            self.chat_panel.add_system_message(f"\u2699 Called: {', '.join(names)}")
                    elif msg.role == "tool":
                        display_result = msg.content
                        if len(display_result) > 500:
                            display_result = display_result[:500] + "\n... (truncated in display)"
                        self.chat_panel.add_tool_result(display_result)
            
            self.update_context_stats()
            self.update_status("\u25cb Ready (resumed)")
            
        except Exception as e:
            logger.error("Resume failed: %s", e)
            self.chat_panel.add_error(f"Failed to resume session {session_id}: {e}")
            self.update_status("\u2717 Error")


    def _cmd_status(self, args: str) -> None:
        if self.current_session is None:
            self.chat_panel.add_system_message("No active session.")
            return
        session = self.current_session
        status = (
            f"[bold]Session[/]\n"
            f"  ID: {session.session_id[:12]}...\n"
            f"  Messages: {len(session.messages)}\n"
            f"  Tokens: {session.total_tokens:,}\n"
            f"  Compactions: {session.compaction_count}\n\n"
            f"[bold]Models[/]\n"
            f"  Primary: [cyan]{self.primary_profile.name}[/]\n"
            f"  Summary: [cyan]{self.secondary_profile.name}[/]\n\n"
            f"{self.budget.format_bar()}"
        )
        self.chat_panel.add_system_message(status)

    def _cmd_cost(self, args: str) -> None:
        if self.current_cost:
            self.chat_panel.add_system_message(self.current_cost.format_detailed())
        else:
            self.chat_panel.add_system_message("No cost data yet.")

    def _cmd_keys(self, args: str) -> None:
        load_env(self.repo_root)
        statuses = key_status()
        lines = ["[bold]API Keys[/]"]
        for provider, status in statuses.items():
            icon = "\u2713" if status != "not set" else "\u2717"
            lines.append(f"  {icon} {provider}: {status}")
        self.chat_panel.add_system_message("\n".join(lines))

    def _cmd_config(self, args: str) -> None:
        if not args:
            from dataclasses import asdict
            lines = ["[bold]Configuration[/]"]
            for k, v in asdict(self.config).items():
                lines.append(f"  {k}: {v}")
            self.chat_panel.add_system_message("\n".join(lines))
            return

        parts = args.split(None, 1)
        if len(parts) == 1:
            key = parts[0]
            if hasattr(self.config, key):
                self.chat_panel.add_system_message(f"  {key}: {getattr(self.config, key)}")
            else:
                self.chat_panel.add_error(f"Unknown config key: {key}")
            return

        key, value = parts
        try:
            self.config.update(key, value)
            self.config.save(self.repo_root)
            if key == "primary_model":
                self.switch_primary_model(value)
            elif key in ("secondary_model",):
                self.switch_secondary_model(value)
            elif key == "compact_threshold":
                self.compact_monitor.threshold = float(value)
            elif key == "output_reserve":
                self.budget.reserved_output_tokens = int(value)
            elif key == "theme":
                try:
                    self.theme = value
                except Exception:
                    self.chat_panel.add_error(f"Failed to set theme: {value}")
            self.chat_panel.add_system_message(f"  Set {key} = {value}")
        except ValueError as e:
            self.chat_panel.add_error(str(e))

    def _cmd_fetch(self, args: str) -> None:
        """Manually fetch OpenRouter models."""
        async def _do_fetch():
            self.chat_panel.add_system_message("[dim]Fetching OpenRouter models...[/]")
            try:
                models = await fetch_openrouter_models()
                if models:
                    save_models(models)  # Use global by default
                    self.openrouter_models = models
                    # Re-resolve
                    self.primary_profile = resolve_profile(self._primary_model_str)
                    self.secondary_profile = resolve_profile(self._secondary_model_str)
                    self.update_context_stats()
                    self._update_model_display()

                    self.chat_panel.add_system_message(
                        f"[green]Loaded {len(models)} OpenRouter models[/]"
                    )
                else:
                    self.chat_panel.add_error("No models returned from OpenRouter")
            except Exception as e:
                self.chat_panel.add_error(f"Fetch failed: {e}")
        asyncio.create_task(_do_fetch())

    def _cmd_quit(self, args: str) -> None:
        self.exit()

    # ─── Model Selector ────────────────────────────────────────────

    def action_model_selector(self) -> None:
        """Open the model selector modal."""
        # Build model list from openrouter
        all_models = []
        for m in self.openrouter_models:
            prefixed_id = f"openrouter/{m['id']}"
            all_models.append({
                "id": prefixed_id,
                "name": m["name"],
                "context_window": m["context_window"],
                "cost_per_1m_input": m.get("cost_per_1m_input", 0.0),
                "cost_per_1m_output": m.get("cost_per_1m_output", 0.0),
            })

        def on_model_selected(model_id: str | None) -> None:
            if model_id:
                self.switch_primary_model(str(model_id))
                self._update_model_display()

        self.push_screen(
            ModelSelectorScreen(all_models, self._primary_model_str, target="primary"),
            on_model_selected,
        )

    # ─── Helper Methods ────────────────────────────────────────────

    def build_llm_messages(self, session: Session) -> list[dict]:
        """Reconstruct the full message list for an LLM call from the persisted session.

        Critically, this must preserve:
          - ``tool_calls`` on assistant messages (so the provider knows which tool was invoked)
          - ``tool_call_id`` on tool messages (so results are matched back to their call)
        Dropping either field causes providers to return a 400 error on every subsequent turn.
        """
        from foreman.brain.assembler import SYSTEM_PROMPT
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        foreman_dir = self.foreman_dir
        for mmd_file, heading in [("structure.mmd", "Project Structure"), ("logic.mmd", "Logic Flow")]:
            path = foreman_dir / mmd_file
            if path.exists():
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    messages[0]["content"] += f"\n\n## {heading}\n```mermaid\n{text}\n```"

        for msg in session.messages:
            d: dict = {"role": msg.role, "content": msg.content or ""}
            # Preserve tool_calls on assistant messages
            if msg.tool_calls:
                d["tool_calls"] = msg.tool_calls
            # Preserve tool_call_id on tool result messages
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            messages.append(d)

        total = self.token_counter.count_message_tokens(messages, self.primary_profile.litellm_model)
        max_budget = self.budget.available_for_session

        if total > max_budget:
            fitted, dropped = self.token_counter.fit_to_budget(messages, max_budget, self.primary_profile.litellm_model)
            if dropped > 0:
                self.chat_panel.add_system_message(f"[dim]Dropped {dropped} older messages to fit context window[/]")
            messages = fitted

        return messages

    def update_status(self, text: str) -> None:
        self.sub_title = text

    def update_context_stats(self) -> None:
        if self.current_session and self.context_panel:
            # Count system prompt + architecture tokens
            system_content = SYSTEM_PROMPT
            mermaid_content = ""
            foreman_dir = self.foreman_dir
            for mmd_file, heading in [("structure.mmd", "Project Structure"), ("logic.mmd", "Logic Flow")]:
                path = foreman_dir / mmd_file
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    if text.strip():
                        mermaid_content += f"\n\n## {heading}\n```mermaid\n{text}\n```"

            self.budget.system_prompt_tokens = self.token_counter.count_tokens(system_content, self.primary_profile.litellm_model)
            self.budget.mermaid_tokens = self.token_counter.count_tokens(mermaid_content, self.primary_profile.litellm_model)

            # Live-count session tokens from the actual message list (includes
            # tool_calls / tool_call_id fields that the stale accumulator misses)
            msgs = self.build_llm_messages(self.current_session)
            # Exclude the system message — budget already accounts for it separately
            non_system = [m for m in msgs if m.get("role") != "system"]
            self.budget.session_tokens = self.token_counter.count_message_tokens(non_system, self.primary_profile.litellm_model)

            self.context_panel.update_stats(self.budget, self.current_cost)

    # ─── Actions ────────────────────────────────────────────────────

    def action_interrupt(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self.chat_panel.add_system_message("[yellow]Interrupted.[/]")
            self.update_status("* Interrupted")

    def action_clear_chat(self) -> None:
        if self.chat_panel:
            self.chat_panel.clear_chat()

    def action_new_session(self) -> None:
        self._cmd_new("")

    async def action_new_session_with_content(self, initial_content: str) -> None:
        """Create a new session and inject a handoff message."""
        await self._init_session()
        if self.chat_panel:
            self.chat_panel.clear_chat()
            self.chat_panel.add_system_message("New session started via handoff.")
            self.chat_panel.add_assistant_message(initial_content)
        
        # Add to the session history so the model sees it
        if self.current_session:
            tokens = self.token_counter.count_tokens(initial_content, self.primary_profile.litellm_model)
            msg = Message(role="assistant", content=initial_content, token_count=tokens)
            await self.session_store.append_message(self.current_session, msg)
            self.update_context_stats()

    async def action_quit(self) -> None:
        if self.current_session:
            await self.session_store.save(self.current_session)
        if self.current_cost:
            await self.cost_tracker.save(self.current_session.session_id)
        self.exit()
