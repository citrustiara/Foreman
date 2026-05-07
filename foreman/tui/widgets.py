"""Custom Textual widgets for Foreman TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, ListItem, ListView, Static, TextArea
from textual.containers import Horizontal
from rich.markdown import Markdown

if TYPE_CHECKING:
    from foreman.tokens.budget import TokenBudget
    from foreman.tokens.cost import SessionCost


# ─── Command Definitions ──────────────────────────────────────────

COMMANDS = {
    "help": "Show available commands",
    "model": "Switch or show models",
    "model list": "Show available model presets",
    "compact": "Force context compaction (summary model)",
    "compact-self": "Force self-compaction (primary model)",
    "implement": "Start implement pipeline",
    "status": "Show session and token stats",
    "cost": "Show cost breakdown",
    "keys": "Show API key status",
    "config": "View or set configuration",
    "fetch": "Fetch latest OpenRouter models",
    "handoff": "Summarize and start a new session (resets tokens)",
    "clear": "Clear chat display",
    "new": "Start a new session",
    "resume": "Resume a past session",
    "approve": "Execute the pending implementation plan",
    "reject": "Cancel the pending implementation plan",
    "quit": "Exit Foreman",
}



# ─── SubmitTextArea ───────────────────────────────────────────────

class SubmitTextArea(TextArea):
    """TextArea that submits on Enter."""

    @dataclass
    class Submitted(Message):
        """Fired when Enter is pressed."""
        text: str

    async def _on_key(self, event: Key) -> None:
        """Override to intercept Enter before TextArea processes it."""
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.text = ""
            return
        # Let TextArea handle everything else
        await super()._on_key(event)


# ─── Command Palette ──────────────────────────────────────────────

class CommandPalette(Static):
    """Floating command list that filters as you type."""

    DEFAULT_CSS = """
    CommandPalette {
        display: none;
        height: auto;
        max-height: 10;
        background: $surface;
        border: tall $primary;
        padding: 0 1;
        margin: 0 0 0 1;
    }
    CommandPalette.visible {
        display: block;
    }
    .cmd-item {
        height: 1;
        padding: 0 1;
    }
    .cmd-item.selected {
        background: $primary 40%;
    }
    .cmd-name {
        color: $accent;
        width: 20;
    }
    .cmd-desc {
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_idx = 0
        self._filtered: list[tuple[str, str]] = []
        self._all_commands = list(COMMANDS.items())

    def filter_commands(self, text: str) -> None:
        """Filter commands based on typed text."""
        if not text.startswith("/"):
            self.remove_class("visible")
            return

        query = text[1:].lower().strip()
        if not query:
            self._filtered = self._all_commands[:]
        else:
            self._filtered = [
                (cmd, desc) for cmd, desc in self._all_commands
                if cmd.startswith(query) or query in cmd
            ]

        if not self._filtered:
            self.remove_class("visible")
            return

        self._selected_idx = min(self._selected_idx, len(self._filtered) - 1)
        self.add_class("visible")
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Render the filtered command list."""
        lines = []
        for i, (cmd, desc) in enumerate(self._filtered):
            marker = "\u25b6 " if i == self._selected_idx else "  "
            lines.append(f"{marker}[cyan]{cmd}[/] [dim]{desc}[/]")
        self.update("\n".join(lines))

    def move_up(self) -> None:
        if self._selected_idx > 0:
            self._selected_idx -= 1
            self._refresh_display()

    def move_down(self) -> None:
        if self._selected_idx < len(self._filtered) - 1:
            self._selected_idx += 1
            self._refresh_display()

    def get_selected(self) -> str | None:
        """Return the selected command, or None."""
        if 0 <= self._selected_idx < len(self._filtered):
            return self._filtered[self._selected_idx][0]
        return None

    def hide(self) -> None:
        self.remove_class("visible")


# ─── Input Bar ────────────────────────────────────────────────────

class InputBar(Vertical):
    """Text input bar with command palette."""

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        max-height: 15;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    @dataclass
    class UserMessage(Message):
        content: str

    @dataclass
    class CommandSubmitted(Message):
        command: str
        args: str

    def compose(self) -> ComposeResult:
        yield CommandPalette(id="cmd-palette")
        yield SubmitTextArea("", id="input-area")

    def on_mount(self) -> None:
        self.query_one("#input-area", SubmitTextArea).focus()

    def on_submit_text_area_submitted(self, event: SubmitTextArea.Submitted) -> None:
        """Handle submit from the text area."""
        palette = self.query_one("#cmd-palette", CommandPalette)
        text = event.text

        if text.startswith("/"):
            # If palette has a selection, use it
            selected = palette.get_selected()
            if selected and not text[1:].strip().startswith(selected.split()[0]):
                # User typed / and selected a command
                text = "/" + selected

            palette.hide()

            parts = text[1:].split(None, 1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""
            self.post_message(self.CommandSubmitted(command=command, args=args))
        else:
            palette.hide()
            self.post_message(self.UserMessage(content=text))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update palette whenever text changes."""
        self._update_palette()


    def on_key(self, event: Key) -> None:
        """Handle arrow keys for palette navigation."""
        palette = self.query_one("#cmd-palette", CommandPalette)
        ta = self.query_one("#input-area", SubmitTextArea)

        if palette.has_class("visible"):
            if event.key == "up":
                event.prevent_default()
                event.stop()
                palette.move_up()
                return
            elif event.key == "down":
                event.prevent_default()
                event.stop()
                palette.move_down()
                return
            elif event.key == "tab":
                event.prevent_default()
                event.stop()
                selected = palette.get_selected()
                if selected:
                    ta.text = "/" + selected
                    palette.hide()
                return

        # Update palette on navigation keys if visible
        # (characters are handled by on_text_area_changed)
        pass


    def _update_palette(self) -> None:
        try:
            ta = self.query_one("#input-area", SubmitTextArea)
            palette = self.query_one("#cmd-palette", CommandPalette)
            palette.filter_commands(ta.text)
        except Exception:
            pass


# ─── Chat Panel ───────────────────────────────────────────────────

class ChatPanel(VerticalScroll):
    """Scrollable chat/log area."""

    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }
    """

    def add_user_message(self, content: str) -> None:
        self.mount(Static(f"[bold blue]\u25b6 You[/]\n{content}", classes="msg msg-user"))
        self.scroll_end()

    def add_assistant_message(self, content: str) -> Static:
        widget = Static(Markdown(content), classes="msg msg-assistant", markup=False)
        self.mount(widget)
        self.scroll_end()
        return widget

    def update_assistant_message(self, widget: Static, content: str) -> None:
        """Update an existing assistant message with new content."""
        widget.update(Markdown(content))
        self.scroll_end()

    def add_system_message(self, content: str) -> None:
        self.mount(Static(content, classes="msg msg-system"))
        self.scroll_end()

    def add_compact_event(self, tokens_before: int, tokens_after: int, messages_archived: int, label: str = "Context Compacted") -> None:
        reduction = (1.0 - tokens_after / tokens_before) * 100 if tokens_before else 0
        text = (
            f"[bold yellow]\u258c {label}[/]\n"
            f"  Tokens: {tokens_before:,} \u2192 {tokens_after:,} ({reduction:.0f}% reduction)\n"
            f"  Archived: {messages_archived} messages"
        )
        self.mount(Static(text, classes="msg msg-compact"))
        self.scroll_end()

    def add_error(self, content: str) -> None:
        self.mount(Static(f"[bold red]\u2717 Error:[/bold red] {content}", classes="msg msg-error"))
        self.scroll_end()

    def add_tool_call(self, content: str) -> None:
        self.mount(Static(content, classes="msg-tool-call"))
        self.scroll_end()

    def add_tool_result(self, content: str) -> None:
        self.mount(Static(content, classes="msg-tool-result"))
        self.scroll_end()

    def clear_chat(self) -> None:
        self.remove_children()


# ─── Context Stats Panel ─────────────────────────────────────────

class ContextStatsPanel(Static):
    """Live token usage and cost display."""

    DEFAULT_CSS = """
    ContextStatsPanel {
        height: auto;
        width: 100%;
        padding: 0 1;
        border-top: solid $primary;
    }
    """

    def update_stats(self, budget: "TokenBudget", cost: "SessionCost | None" = None) -> None:
        bar = budget.format_bar(width=20)
        total_used = budget.used  # system_reserve + session_tokens
        lines = [
            f"[bold]Token Budget[/]",
            f"  {bar}",
            f"  Context: {budget.context_window:,}",
            f"  System: {budget.system_prompt_tokens:,}",
            f"  Arch: {budget.mermaid_tokens:,}",
            f"  Session: {budget.session_tokens:,}",
            f"  Total used: {total_used:,}",
            f"  Available: {budget.available_for_session:,}",
        ]
        if cost and (cost.total_cost > 0 or cost.entries):
            lines.append("")
            lines.append(f"[bold]Cost[/]")
            lines.append(f"  Total: ${cost.total_cost:.4f}")
            lines.append(f"  Input: {cost.total_input_tokens:,} tokens")
            lines.append(f"  Output: {cost.total_output_tokens:,} tokens")
            lines.append(f"  Calls: {len(cost.entries)}")
        self.update("\n".join(lines))


# ─── Model Selector Screen ───────────────────────────────────────

class ModelSelectorScreen(ModalScreen[str | None]):
    """Modal screen for selecting a model."""

    DEFAULT_CSS = """
    ModelSelectorScreen {
        background: $surface 80%;
        align: center middle;
    }
    #model-selector {
        width: 120;
        height: 35;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }
    #model-title {
        text-align: center;
        margin-bottom: 1;
    }
    #model-search {
        margin-bottom: 1;
        height: 3;
        border: tall $primary;
    }
    #model-list {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    def __init__(self, models: list[dict], current_model: str, target: str = "primary", **kwargs):
        super().__init__(**kwargs)
        self._models = models
        self._current = current_model
        self._target = target
        self._filtered = models[:]

    def compose(self) -> ComposeResult:
        with Vertical(id="model-selector"):
            yield Label(f"Select {self._target.title()} Model", id="model-title")
            yield Input(placeholder="Type to filter models...", id="model-search")
            yield DataTable(id="model-list")

    def on_mount(self) -> None:
        table = self.query_one("#model-list", DataTable)
        table.add_columns("", "Model", "Context", "$/M tokens in/out")
        table.cursor_type = "row"
        self._render_table(self._models)
        self.query_one("#model-search", Input).focus()

    def _render_table(self, models: list[dict]) -> None:
        table = self.query_one("#model-list", DataTable)
        table.clear()
        for m in models:
            ctx = f"{m['context_window'] // 1000:,}K"
            cost = f"${m['cost_per_1m_input']:.2f} / ${m['cost_per_1m_output']:.2f}"
            marker = "\u25cf" if m["id"] == self._current else ""
            table.add_row(marker, m["name"], ctx, cost, key=m["id"])

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter models as user types."""
        query = event.value.lower().strip()
        if not query:
            self._filtered = self._models[:]
        else:
            self._filtered = [
                m for m in self._models
                if query in m["id"].lower() or query in m["name"].lower()
            ]
        self._render_table(self._filtered)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Select the highlighted model on Enter."""
        self._select_current()

    def on_key(self, event: Key) -> None:
        """Handle enter key for selection."""
        if event.key == "enter":
            # Only handle if input is focused (on_input_submitted handles this)
            # but also handle if table is focused
            self._select_current()
            event.prevent_default()
            event.stop()

    def _select_current(self) -> None:
        """Select the currently highlighted model."""
        table = self.query_one("#model-list", DataTable)
        try:
            cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            self.dismiss(str(cell_key.row_key.value))
        except Exception:
            pass

    def action_cursor_up(self) -> None:
        """Move cursor up in the table."""
        table = self.query_one("#model-list", DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the table."""
        table = self.query_one("#model-list", DataTable)
        table.action_cursor_down()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(str(event.row_key.value))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


# ─── Help Screen ──────────────────────────────────────────────────

class HelpScreen(ModalScreen[None]):
    """Modal help overlay."""

    DEFAULT_CSS = """
    HelpScreen {
        background: $surface 80%;
        align: center middle;
    }
    #help-box {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            help_text = "\n".join([
                "[bold]Foreman Commands[/]",
                "",
                "[cyan]/help[/]        Show this help",
                "[cyan]/model[/]       Show current models",
                "[cyan]/model list[/]  List available presets",
                "[cyan]/model primary <id>[/]  Switch primary model",
                "[cyan]/model summary <id>[/]  Switch summary model",
                "[cyan]/compact[/]     Force compaction (summary model)",
                "[cyan]/compact-self[/] Force self-compaction (primary model)",
                "[cyan]/implement[/]   Start implement pipeline",
                "[cyan]/approve[/]     Approve pending plan",
                "[cyan]/reject[/]      Reject pending plan",
                "[cyan]/status[/]      Session and token stats",
                "[cyan]/cost[/]        Cost breakdown",
                "[cyan]/keys[/]        API key status",
                "[cyan]/config[/]      View/set configuration",
                "[cyan]/fetch[/]       Fetch latest OpenRouter models",
                "[cyan]/clear[/]       Clear chat",
                "[cyan]/new[/]         New session",
                "[cyan]/resume[/]      Resume a past session",
                "[cyan]/quit[/]        Exit Foreman",
                "",
                "[bold]Keybindings[/]",
                "[cyan]Ctrl+L[/]       Model selector",
                "[cyan]Ctrl+N[/]       New session",
                "[cyan]Ctrl+Q[/]       Quit",
                "[cyan]Escape[/]       Cancel running task",
            ])
            yield Static(help_text)

    def on_key(self, event: Key) -> None:
        if event.key in ("escape", "enter", "q"):
            self.dismiss()

# ─── Session Selector Screen ─────────────────────────────────────

class SessionSelectorScreen(ModalScreen[str | None]):
    """Modal screen for selecting a past session to resume."""

    DEFAULT_CSS = """
    SessionSelectorScreen {
        background: $surface 80%;
        align: center middle;
    }
    #session-selector {
        width: 100;
        height: 25;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }
    #session-title {
        text-align: center;
        margin-bottom: 1;
    }
    #session-list {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=False),
    ]

    def __init__(self, sessions: list[dict], **kwargs):
        super().__init__(**kwargs)
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="session-selector"):
            yield Label("Resume Past Session", id="session-title")
            yield DataTable(id="session-list")

    def on_mount(self) -> None:
        table = self.query_one("#session-list", DataTable)
        table.add_columns("Date", "ID", "Msgs", "Tokens")
        table.cursor_type = "row"
        
        # Sort sessions by updated_at descending
        sorted_sessions = sorted(self._sessions, key=lambda s: s["updated_at"], reverse=True)
        
        for s in sorted_sessions:
            # Format date
            try:
                dt = s["updated_at"].split(".")[0].replace("T", " ")
            except Exception:
                dt = s["updated_at"]
            
            table.add_row(
                dt,
                s["session_id"][:12] + "...",
                str(s["message_count"]),
                f"{s['total_tokens']:,}",
                key=s["session_id"]
            )
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(str(event.row_key.value))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
