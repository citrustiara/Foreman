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
from textual.widgets import Button, DirectoryTree, Footer, Header, Static

from foreman.brain.session import Session, SessionStore, Message
from foreman.brain.assembler import SYSTEM_PROMPT

from foreman.brain.architecture import refresh_context_json
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
    FilePreviewPanel,
    InputBar,
    StatusFooter,
    SubmitTextArea,
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
        Binding("ctrl+l", "model_selector", "Models", show=True),
        Binding("shift+tab", "cycle_thinking_effort", "Thinking", show=True),
        Binding("ctrl+o", "toggle_tool_output", "Tool Output", show=True),
        Binding("ctrl+d", "toggle_file_diff", "File Diff", show=True),
        Binding("ctrl+b", "toggle_context_panel", "Stats Panel", show=True),
        Binding("ctrl+e", "toggle_code_focus", "Code Focus", show=True),
        Binding("ctrl+s", "save_open_file", "Save File", show=False),
        Binding("ctrl+n", "new_session", "New Session", show=True),
        Binding("escape", "interrupt", "Cancel", show=False),
    ]

    def __init__(
        self,
        repo_root: Path | None = None,
        hydrate_context_on_start: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.repo_root = repo_root or Path.cwd()
        self.foreman_dir = self.repo_root / ".foreman"
        self.hydrate_context_on_start = hydrate_context_on_start

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
        self.file_preview: FilePreviewPanel | None = None
        self.footer_stats: StatusFooter | None = None
        self.right_sidebar: Vertical | None = None

        # UI state
        self._thinking_levels = ("none", "low", "med", "high")
        self._thinking_idx = 1
        self.expand_tool_output = False
        self._open_file_path: Path | None = None
        self._file_preview_show_diff = False
        self._context_panel_minimized = True
        self._code_focus = False
        self.input_locked = False
        self.session_modified_files: set[str] = set()
        self.session_file_diffs: dict[str, str] = {}

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
                with Horizontal(id="work-area"):
                    yield FilePreviewPanel(id="file-preview")
                    yield ChatPanel(id="chat-panel")
                yield InputBar(id="input-bar")
            with Vertical(classes="sidebar sidebar-right", id="right-sidebar"):
                yield Button("▕◀", id="toggle-context-btn")
                yield ContextStatsPanel(id="context-stats")
        yield StatusFooter(id="stats-footer")
        yield Footer()

    async def on_mount(self) -> None:
        load_env(self.repo_root)

        self.chat_panel = self.query_one("#chat-panel", ChatPanel)
        self.context_panel = self.query_one("#context-stats", ContextStatsPanel)
        self.input_bar = self.query_one("#input-bar", InputBar)
        self.file_preview = self.query_one("#file-preview", FilePreviewPanel)
        self.footer_stats = self.query_one("#stats-footer", StatusFooter)
        self.right_sidebar = self.query_one("#right-sidebar", Vertical)

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
        self._apply_panel_layout_state()
        self._update_model_display()
        self.update_status("○ Ready")


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
        self.session_modified_files.clear()
        self.session_file_diffs.clear()
        self._open_file_path = None
        self._file_preview_show_diff = False
        self._code_focus = False
        if self.file_preview:
            self.file_preview.hide_file()
        self._apply_panel_layout_state()
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

    def _apply_panel_layout_state(self) -> None:
        if self.right_sidebar:
            button = self.query_one("#toggle-context-btn", Button)
            if self._context_panel_minimized:
                self.right_sidebar.add_class("minimized")
                button.label = "▕▶"
            else:
                self.right_sidebar.remove_class("minimized")
                button.label = "▕◀"

        work_area = self.query_one("#work-area", Horizontal)
        if self._code_focus:
            work_area.add_class("code-focused")
        else:
            work_area.remove_class("code-focused")

    def _build_dynamic_system_suffix(self, session: Session) -> str:
        total_used = self.budget.used
        context_window = max(1, self.budget.context_window)
        pct = (total_used / context_window) * 100.0
        total_cost = self.current_cost.total_cost if self.current_cost else 0.0
        total_input = self.current_cost.total_input_tokens if self.current_cost else 0
        total_output = self.current_cost.total_output_tokens if self.current_cost else 0
        return (
            "\n\n## Runtime Token/Cost Snapshot (dynamic; appended for cache locality)\n"
            f"- Context used: {total_used:,}/{context_window:,} ({pct:.1f}%)\n"
            f"- Session tokens: {session.total_tokens:,}\n"
            f"- Remaining budget estimate: {self.budget.available_for_session:,}\n"
            f"- Session cost total: ${total_cost:.6f}\n"
            f"- Session input/output tokens: {total_input:,}/{total_output:,}\n"
            "- Use these values when deciding to use token-saving tools "
            "(especially summarize_last and checkpoint_summary for large outputs)."
        )

    def _build_system_content(self, session: Session) -> str:
        system_content = SYSTEM_PROMPT
        context_path = self.foreman_dir / "context.json"
        if context_path.exists():
            text = context_path.read_text(encoding="utf-8")
            if text.strip():
                system_content += f"\n\n## Project Context JSON\n```json\n{text}\n```"
        system_content += self._build_dynamic_system_suffix(session)
        return system_content

    # ─── Command Handling ──────────────────────────────────────────

    def on_input_bar_command_submitted(self, event: InputBar.CommandSubmitted) -> None:
        cmd = event.command
        args = event.args.strip()

        handlers = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "compact": self._cmd_compact,
            "compact-self": self._cmd_compact_self,
            "plan": self._cmd_plan,
            "implement": self._cmd_plan,
            "clear": self._cmd_clear,
            "new": self._cmd_new,
            "status": self._cmd_status,
            "cost": self._cmd_cost,
            "keys": self._cmd_keys,
            "handoff": self._cmd_handoff,
            "config": self._cmd_config,
            "fetch": self._cmd_fetch,
            "context": self._cmd_context,
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

    def _cmd_plan(self, args: str) -> None:
        from foreman.tui.workers import run_plan

        if args:
            self._current_task = asyncio.create_task(run_plan(self, args))
            return

        self._current_task = asyncio.create_task(self._plan_from_last_assistant())

    async def _plan_from_last_assistant(self) -> None:
        from foreman.tui.workers import run_plan

        if not self.current_session:
            self.chat_panel.add_error("No active session.")
            return

        last_assistant = next(
            (m for m in reversed(self.current_session.messages) if m.role == "assistant" and m.content.strip()),
            None,
        )
        if not last_assistant:
            self.chat_panel.add_error("/plan requires a feature description or a prior assistant message.")
            return

        self.chat_panel.add_system_message("[dim]Summarizing last assistant message for planning...[/]")
        try:
            summary = await self.router.generate(
                profile=self.secondary_profile,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following assistant output into a concise implementation brief. "
                            "Preserve explicit requirements, constraints, and acceptance criteria. "
                            "Write only the brief."
                        ),
                    },
                    {"role": "user", "content": last_assistant.content},
                ],
                temperature=0.2,
                max_tokens=min(600, self.config.max_summary_tokens),
                **self.llm_reasoning_kwargs(),
            )
            await run_plan(self, summary.strip() or last_assistant.content[:1000])
        except Exception as e:
            logger.error("Failed to summarize last assistant message for /plan: %s", e)
            self.chat_panel.add_error(f"/plan failed: {e}")

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
                reasoning_effort=self.reasoning_effort_api(),
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
            self._rebuild_modified_files_from_session()
            
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
                            if self.expand_tool_output:
                                for tc in msg.tool_calls:
                                    fn = tc.get("function", {})
                                    self.chat_panel.add_tool_call(
                                        f"⚙ {fn.get('name', 'tool')} {fn.get('arguments', '')}"
                                    )
                            else:
                                names = [tc["function"]["name"] for tc in msg.tool_calls]
                                self.chat_panel.add_system_message(f"\u2699 Called: {', '.join(names)}")
                    elif msg.role == "tool":
                        self.chat_panel.add_tool_result(self.format_tool_display(msg.content))
            
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
            f"  Summary: [cyan]{self.secondary_profile.name}[/]\n"
            f"  Thinking: [cyan]{self.thinking_level}[/]\n"
            f"  Tool output: [cyan]{'expanded' if self.expand_tool_output else 'collapsed'}[/]\n\n"
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

    def _cmd_context(self, args: str) -> None:
        """Refresh dense .foreman/context.json using the summary model."""
        async def _do_context_refresh():
            self.chat_panel.add_system_message("[dim]Refreshing .foreman/context.json...[/]")
            try:
                await refresh_context_json(
                    repo_root=self.repo_root,
                    foreman_dir=self.foreman_dir,
                    router=self.router,
                    summary_model=self.secondary_profile,
                )
                self.update_context_stats()
                self.chat_panel.add_system_message("[green]Updated .foreman/context.json[/]")
            except Exception as e:
                self.chat_panel.add_error(f"Context refresh failed: {e}")

        asyncio.create_task(_do_context_refresh())

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

    def action_cycle_thinking_effort(self) -> None:
        self._thinking_idx = (self._thinking_idx + 1) % len(self._thinking_levels)
        self.update_context_stats()
        self.chat_panel.add_system_message(f"[dim]Thinking effort: {self.thinking_level}[/]")
        self.update_status("○ Ready")

    def action_toggle_tool_output(self) -> None:
        self.expand_tool_output = not self.expand_tool_output
        self._rerender_current_session()
        self.update_context_stats()
        self.chat_panel.add_system_message(
            f"[dim]Tool output {'expanded' if self.expand_tool_output else 'collapsed'}.[/]"
        )
        self.update_status("○ Ready")

    def action_toggle_file_diff(self) -> None:
        if not self._open_file_path or not self.file_preview:
            self.chat_panel.add_system_message("[dim]No file open in preview.[/]")
            return
        self._file_preview_show_diff = not self._file_preview_show_diff
        path_rel = self._to_repo_path(self._open_file_path)
        if self._file_preview_show_diff:
            self.file_preview.show_diff(path_rel, self.session_file_diffs.get(path_rel, ""))
            self.update_status(f"○ Preview diff: {self._open_file_path.name}")
        else:
            self.file_preview.show_file(str(self._open_file_path))
            self.update_status(f"○ Preview source: {self._open_file_path.name}")
        self.update_context_stats()

    def action_toggle_context_panel(self) -> None:
        self._context_panel_minimized = not self._context_panel_minimized
        self._apply_panel_layout_state()
        self.update_context_stats()
        self.update_status("○ Ready")

    def action_toggle_code_focus(self) -> None:
        if not self._open_file_path:
            self.chat_panel.add_system_message("[dim]Open a file first to focus code view.[/]")
            return
        self._code_focus = not self._code_focus
        self._apply_panel_layout_state()
        self.update_status(
            f"○ Code focus {'on' if self._code_focus else 'off'}: {self._open_file_path.name}"
        )

    def action_save_open_file(self) -> None:
        if self.file_preview:
            self.file_preview.action_save_file()

    # ─── Helper Methods ────────────────────────────────────────────

    def build_llm_messages(self, session: Session) -> list[dict]:
        """Reconstruct the full message list for an LLM call from the persisted session.

        Critically, this must preserve:
          - ``tool_calls`` on assistant messages (so the provider knows which tool was invoked)
          - ``tool_call_id`` on tool messages (so results are matched back to their call)
        Dropping either field causes providers to return a 400 error on every subsequent turn.
        """
        messages: list[dict] = [{"role": "system", "content": self._build_system_content(session)}]

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
        self.sub_title = (
            f"{text} • think:{self.thinking_level} • "
            f"tools:{'full args' if self.expand_tool_output else 'names only'}"
        )

    def update_context_stats(self) -> None:
        if self.current_session and self.context_panel:
            system_content = self._build_system_content(self.current_session)
            self.budget.system_prompt_tokens = self.token_counter.count_tokens(
                SYSTEM_PROMPT, self.primary_profile.litellm_model
            )
            dynamic_tail = system_content.removeprefix(SYSTEM_PROMPT)
            self.budget.mermaid_tokens = self.token_counter.count_tokens(
                dynamic_tail, self.primary_profile.litellm_model
            )

            # Live-count session tokens from the actual message list (includes
            # tool_calls / tool_call_id fields that the stale accumulator misses)
            msgs = self.build_llm_messages(self.current_session)
            # Exclude the system message — budget already accounts for it separately
            non_system = [m for m in msgs if m.get("role") != "system"]
            self.budget.session_tokens = self.token_counter.count_message_tokens(non_system, self.primary_profile.litellm_model)

            self.context_panel.update_stats(
                self.budget,
                self.current_cost,
                thinking_level=self.thinking_level,
                expanded_tools=self.expand_tool_output,
                modified_files=sorted(self.session_modified_files),
                open_file=self._to_repo_path(self._open_file_path) if self._open_file_path else None,
            )
            if self.footer_stats:
                self.footer_stats.update_stats(
                    context_used=self.budget.used,
                    context_window=self.budget.context_window,
                    available=self.budget.available_for_session,
                    total_cost=self.current_cost.total_cost if self.current_cost else 0.0,
                    input_tokens=self.current_cost.total_input_tokens if self.current_cost else 0,
                    output_tokens=self.current_cost.total_output_tokens if self.current_cost else 0,
                )

    @property
    def thinking_level(self) -> str:
        return self._thinking_levels[self._thinking_idx]

    def reasoning_effort_api(self) -> str | None:
        mapping = {"none": None, "low": "low", "med": "medium", "high": "high"}
        return mapping[self.thinking_level]

    def llm_reasoning_kwargs(self) -> dict[str, str]:
        effort = self.reasoning_effort_api()
        if effort is None:
            return {}
        return {"reasoning_effort": effort}

    def format_tool_display(self, content: str) -> str:
        if self.expand_tool_output:
            return content
        if len(content) > 500:
            return content[:500] + "\n... (truncated in display)"
        return content

    def _rerender_current_session(self) -> None:
        if not self.current_session or not self.chat_panel:
            return
        self.chat_panel.clear_chat()
        for msg in self.current_session.messages:
            if msg.role == "user":
                self.chat_panel.add_user_message(msg.content)
            elif msg.role == "assistant":
                if msg.content.strip():
                    self.chat_panel.add_assistant_message(msg.content)
                if msg.tool_calls:
                    if self.expand_tool_output:
                        for tc in msg.tool_calls:
                            fn = tc.get("function", {})
                            self.chat_panel.add_tool_call(
                                f"⚙ {fn.get('name', 'tool')} {fn.get('arguments', '')}"
                            )
                    else:
                        names = [tc["function"]["name"] for tc in msg.tool_calls]
                        self.chat_panel.add_system_message(f"⚙ Called: {', '.join(names)}")
            elif msg.role == "tool":
                self.chat_panel.add_tool_result(self.format_tool_display(msg.content))

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        selected = Path(event.path)
        if not selected.is_file():
            return
        if self._open_file_path and selected.resolve() == self._open_file_path.resolve():
            self._open_file_path = None
            self._file_preview_show_diff = False
            self._code_focus = False
            if self.file_preview:
                self.file_preview.hide_file()
            self._apply_panel_layout_state()
            self.update_status("○ File preview closed")
            self.update_context_stats()
            return
        self._open_file_path = selected
        self._file_preview_show_diff = False
        if self.file_preview:
            self.file_preview.show_file(str(selected))
        self._apply_panel_layout_state()
        self.update_status(f"○ Preview: {selected.name}")
        self.update_context_stats()

    def on_file_preview_panel_file_saved(self, event: FilePreviewPanel.FileSaved) -> None:
        self.register_file_modification(event.path, event.diff_text)
        self.chat_panel.add_system_message(f"[green]Saved[/] {self._to_repo_path(Path(event.path))}")
        self.update_status(f"○ Saved: {Path(event.path).name}")

    # ─── Actions ────────────────────────────────────────────────────

    def action_interrupt(self) -> None:
        if self.input_locked:
            self.set_input_enabled(True)
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

    def set_input_enabled(self, enabled: bool) -> None:
        """Lock/unlock message submission while keeping UI navigation responsive."""
        if not self.input_bar:
            return
        try:
            ta = self.input_bar.query_one("#input-area", SubmitTextArea)
            self.input_locked = not enabled
            ta.disabled = False
            ta.read_only = self.input_locked
            if enabled:
                ta.focus()
            elif self.chat_panel:
                self.chat_panel.focus()
        except Exception:
            pass

    def scroll_workspace(self, key: str) -> None:
        """Scroll a useful panel when input is locked and nav keys are pressed."""
        target = self.chat_panel
        if self.file_preview and self.file_preview.has_focus_within:
            target = self.file_preview
        if not target:
            return
        mapping = {
            "up": "action_scroll_up",
            "down": "action_scroll_down",
            "left": "action_scroll_left",
            "right": "action_scroll_right",
            "pageup": "action_page_up",
            "pagedown": "action_page_down",
            "home": "action_scroll_home",
            "end": "action_scroll_end",
        }
        action_name = mapping.get(key)
        if not action_name:
            return
        action = getattr(target, action_name, None)
        if callable(action):
            action()

    def _to_repo_path(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except Exception:
            return path.as_posix()

    def register_file_modification(self, path: str, diff_text: str) -> None:
        rel = self._to_repo_path((self.repo_root / path) if not Path(path).is_absolute() else Path(path))
        self.session_modified_files.add(rel)
        self.session_file_diffs[rel] = diff_text
        self.update_context_stats()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-context-btn":
            self.action_toggle_context_panel()

    def _rebuild_modified_files_from_session(self) -> None:
        self.session_modified_files.clear()
        self.session_file_diffs.clear()
        if not self.current_session:
            return
        for msg in self.current_session.messages:
            if msg.role != "tool":
                continue
            if not msg.content.startswith("Updated "):
                continue
            first_line = msg.content.splitlines()[0]
            path = first_line.removeprefix("Updated ").strip()
            if path:
                self.register_file_modification(path, msg.content)
