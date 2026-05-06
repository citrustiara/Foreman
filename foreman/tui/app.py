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

from foreman.brain.session import Session, SessionStore
from foreman.brain.architecture import load_structure, load_logic, load_context
from foreman.compact.monitor import CompactMonitor
from foreman.compact.summarizer import Summarizer
from foreman.config import ForemanConfig
from foreman.models.keys import load_env, key_status
from foreman.models.profiles import ModelProfile, PRESETS, resolve_profile
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
)

logger = logging.getLogger("foreman.tui.app")


def load_openrouter_models(foreman_dir: Path) -> list[dict]:
    """Load OpenRouter model list from .foreman/openrouter_models.json."""
    return load_models(foreman_dir)


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
        self.openrouter_models = load_openrouter_models(self.foreman_dir)

        # Widget references
        self.chat_panel: ChatPanel | None = None
        self.context_panel: ContextStatsPanel | None = None
        self.input_bar: InputBar | None = None

        # Current running task (for ESC cancellation)
        self._current_task: asyncio.Task | None = None

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
            cache_path = self.foreman_dir / "openrouter_models.json"
            # Only fetch if cache doesn't exist
            if not cache_path.exists():
                models = await fetch_openrouter_models()
                if models:
                    save_models(models, self.foreman_dir)
                    self.openrouter_models = models
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
            lines = ["[bold]Available Models[/]"]
            for key, profile in PRESETS.items():
                ctx = f"{profile.max_context // 1000}K"
                lines.append(f"  [cyan]{key}[/] \u2014 {profile.name} ({ctx})")
            if self.openrouter_models:
                lines.append("")
                lines.append("[bold]OpenRouter Models[/]")
                for m in self.openrouter_models:
                    ctx = f"{m['context_window'] // 1000}K"
                    lines.append(f"  [cyan]openrouter/{m['id']}[/] \u2014 {m['name']} ({ctx})")
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
            self.chat_panel.add_error("/implement requires a feature description")
            return
        from foreman.tui.workers import run_implement
        self._current_task = asyncio.create_task(run_implement(self, args))

    def _cmd_handoff(self, args: str) -> None:
        from foreman.tui.workers import run_handoff
        self._current_task = asyncio.create_task(run_handoff(self))

    def _cmd_clear(self, args: str) -> None:
        if self.chat_panel:
            self.chat_panel.clear_chat()

    def _cmd_new(self, args: str) -> None:
        asyncio.create_task(self._init_session())
        if self.chat_panel:
            self.chat_panel.add_system_message("New session started.")

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
                    save_models(models, self.foreman_dir)
                    self.openrouter_models = models
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
        # Build combined model list: presets + openrouter
        all_models = []
        for key, profile in PRESETS.items():
            all_models.append({
                "id": key,
                "name": profile.name,
                "context_window": profile.max_context,
                "cost_per_1m_input": profile.cost_per_1m_input,
                "cost_per_1m_output": profile.cost_per_1m_output,
            })
        for m in self.openrouter_models:
            prefixed_id = f"openrouter/{m['id']}"
            if not any(am["id"] == prefixed_id for am in all_models):
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

    def build_llm_messages(self, session: Session) -> list[dict[str, str]]:
        from foreman.brain.assembler import SYSTEM_PROMPT
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        foreman_dir = self.foreman_dir
        for mmd_file, heading in [("structure.mmd", "Project Structure"), ("logic.mmd", "Logic Flow")]:
            path = foreman_dir / mmd_file
            if path.exists():
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    messages[0]["content"] += f"\n\n## {heading}\n```mermaid\n{text}\n```"

        for msg in session.messages:
            messages.append({"role": msg.role, "content": msg.content})

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
            from foreman.brain.assembler import SYSTEM_PROMPT
            system_content = SYSTEM_PROMPT
            foreman_dir = self.foreman_dir
            for mmd_file, heading in [("structure.mmd", "Project Structure"), ("logic.mmd", "Logic Flow")]:
                path = foreman_dir / mmd_file
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    if text.strip():
                        system_content += f"\n\n## {heading}\n```mermaid\n{text}\n```"
            self.budget.system_prompt_tokens = self.token_counter.count_tokens(system_content, self.primary_profile.litellm_model)
            self.budget.session_tokens = self.current_session.total_tokens
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
            from foreman.brain.session import Message
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
