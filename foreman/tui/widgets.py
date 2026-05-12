"""Custom Textual widgets for Foreman TUI."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, ListItem, ListView, Static, TextArea
from rich.markdown import Markdown

# File extension → Pygments/Textual language mapping for syntax highlighting
_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".jsonc": "json",
    ".json5": "json5",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".xml": "xml",
    ".svg": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".bat": "batch",
    ".cmd": "batch",
    ".ps1": "powershell",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".clj": "clojure",
    ".groovy": "groovy",
    ".dockerfile": "docker",
    ".makefile": "make",
    ".cmake": "cmake",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".vue": "vue",
    ".svelte": "svelte",
    ".tcl": "tcl",
    ".v": "verilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
    ".zig": "zig",
    ".nim": "nim",
    ".v": "v",
}

# Filenames (no extension) that map to a language
_FILENAME_LANG: dict[str, str] = {
    "Dockerfile": "docker",
    "Makefile": "make",
    "makefile": "make",
    "CMakeLists.txt": "cmake",
    "Vagrantfile": "ruby",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
    "Justfile": "just",
    "justfile": "just",
    "Tiltfile": "starlark",
    "PKGBUILD": "bash",
    ".env": "ini",
    ".gitignore": "gitignore",
    ".dockerignore": "gitignore",
    ".editorconfig": "ini",
    ".eslintrc": "json",
    ".prettierrc": "json",
    "Cargo.toml": "toml",
    "pyproject.toml": "toml",
    "Pipfile": "toml",
}


def detect_language(file_path: str) -> str | None:
    """Detect the syntax highlighting language from a file path.

    Returns a Pygments language name or None if unknown.
    """
    p = Path(file_path)
    # Check by exact filename first
    if p.name in _FILENAME_LANG:
        return _FILENAME_LANG[p.name]
    # Check by extension
    ext = p.suffix.lower()
    if ext in _EXT_LANG:
        return _EXT_LANG[ext]
    # Handle double extensions like .t.css → css
    if len(p.suffixes) >= 2:
        double = "".join(p.suffixes[-2:]).lower()
        if double in _EXT_LANG:
            return _EXT_LANG[double]
    return None

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
    "plan": "Generate and stage an implementation plan",
    "status": "Show session and token stats",
    "cost": "Show cost breakdown",
    "keys": "Show API key status",
    "config": "View or set configuration",
    "fetch": "Fetch latest OpenRouter models",
    "context": "Refresh dense .foreman/context.json",
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

    @staticmethod
    def _read_clipboard_text() -> str | None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            text = root.clipboard_get()
            root.destroy()
            return text
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None

    async def _on_key(self, event: Key) -> None:
        """Override to intercept Enter before TextArea processes it."""
        if getattr(self.app, "input_locked", False):
            if event.key in {"up", "down", "left", "right", "pageup", "pagedown", "home", "end"}:
                event.prevent_default()
                event.stop()
                self.app.scroll_workspace(event.key)
                return
            if event.key in {"ctrl+b", "ctrl+e", "ctrl+o", "ctrl+d", "ctrl+s"}:
                # Still allow global app shortcuts while generation is running.
                pass
            else:
                # Ignore regular typing while locked.
                event.prevent_default()
                event.stop()
                return
        if event.key == "shift+tab":
            event.prevent_default()
            event.stop()
            self.app.action_cycle_thinking_effort()
            return
        if event.key == "ctrl+o":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_tool_output()
            return
        if event.key == "ctrl+d":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_file_diff()
            return
        if event.key == "ctrl+b":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_context_panel()
            return
        if event.key == "ctrl+e":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_code_focus()
            return
        if event.key == "ctrl+s":
            event.prevent_default()
            event.stop()
            self.app.action_save_open_file()
            return
        if event.key == "ctrl+z":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_undo"):
                self.action_undo()
            return
        if event.key == "ctrl+y":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_redo"):
                self.action_redo()
            return
        if event.key == "ctrl+a":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_select_all"):
                self.action_select_all()
            return
        if event.key == "ctrl+v":
            event.prevent_default()
            event.stop()
            pasted = self._read_clipboard_text()
            if pasted:
                try:
                    self.insert(pasted)
                except Exception:
                    self.text += pasted
            return
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


class PreviewTextArea(TextArea):
    """Editor textarea used in the file preview panel."""

    async def _on_key(self, event: Key) -> None:
        if event.key == "ctrl+b":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_context_panel()
            return
        if event.key == "ctrl+e":
            event.prevent_default()
            event.stop()
            self.app.action_toggle_code_focus()
            return
        if event.key == "ctrl+s":
            event.prevent_default()
            event.stop()
            self.app.action_save_open_file()
            return
        if event.key == "ctrl+z":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_undo"):
                self.action_undo()
            return
        if event.key == "ctrl+y":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_redo"):
                self.action_redo()
            return
        if event.key == "ctrl+a":
            event.prevent_default()
            event.stop()
            if hasattr(self, "action_select_all"):
                self.action_select_all()
            return
        if event.key == "ctrl+v":
            event.prevent_default()
            event.stop()
            pasted = SubmitTextArea._read_clipboard_text()
            if pasted:
                try:
                    self.insert(pasted)
                except Exception:
                    self.text += pasted
            return
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
        ta = self.query_one("#input-area", SubmitTextArea)
        ta.soft_wrap = True
        ta.focus()
        self._autosize_input()

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
        self._autosize_input()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update palette whenever text changes."""
        self._update_palette()
        self._autosize_input()


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

    def _autosize_input(self) -> None:
        try:
            ta = self.query_one("#input-area", SubmitTextArea)
            lines = max(1, ta.text.count("\n") + 1)
            ta.styles.height = min(5, max(3, lines + 1))
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
        self.mount(Static(f"[bold blue]\u25b6 You[/]\n{escape(content)}", classes="msg msg-user"))
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
        self.mount(Static(f"[bold red]\u2717 Error:[/bold red] {escape(content)}", classes="msg msg-error"))
        self.scroll_end()

    def add_tool_call(self, content: str) -> None:
        self.mount(Static(escape(content), classes="msg-tool-call"))
        self.scroll_end()

    def add_tool_result(self, content: str) -> None:
        self.mount(Static(self._colorize_diff(content), classes="msg-tool-result"))
        self.scroll_end()

    def clear_chat(self) -> None:
        self.remove_children()

    @staticmethod
    def _colorize_diff(content: str) -> str:
        lines = content.splitlines()
        if not any(line.startswith(("--- ", "+++ ", "@@", "+", "-")) for line in lines):
            # Not a diff — escape all markup so brackets in tool output don't break rendering
            return escape(content)
        colored: list[str] = []
        for line in lines:
            safe = escape(line)
            if line.startswith("+") and not line.startswith("+++"):
                colored.append(f"[green]{safe}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                colored.append(f"[red]{safe}[/]")
            elif line.startswith(("@@", "---", "+++")):
                colored.append(f"[cyan]{safe}[/]")
            else:
                colored.append(safe)
        return "\n".join(colored)


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

    def update_stats(
        self,
        budget: "TokenBudget",
        cost: "SessionCost | None" = None,
        thinking_level: str = "low",
        expanded_tools: bool = False,
        modified_files: list[str] | None = None,
        open_file: str | None = None,
    ) -> None:
        bar = budget.format_bar(width=20)
        total_used = budget.used  # system_reserve + session_tokens
        lines = [
            f"[bold]Token Budget[/]",
            f"  Thinking: {thinking_level}",
            f"  Tool view: {'expanded' if expanded_tools else 'collapsed'}",
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
        if modified_files:
            lines.append("")
            lines.append("[bold]Modified (session)[/]")
            for path in modified_files[:12]:
                is_open = open_file is not None and path == open_file
                lines.append(f"  {'[bold green]●[/] ' if is_open else '[green]•[/] '}{path}")
            if len(modified_files) > 12:
                lines.append(f"  [dim]+{len(modified_files) - 12} more[/]")
        self.update("\n".join(lines))


class FilePreviewPanel(VerticalScroll):
    """Editable source preview panel with optional diff view."""

    DEFAULT_CSS = """
    FilePreviewPanel {
        display: none;
        width: 1fr;
        border-right: solid $primary;
        padding: 0 1;
    }
    FilePreviewPanel.visible {
        display: block;
    }
    """

    @dataclass
    class FileSaved(Message):
        path: str
        diff_text: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_path: str | None = None
        self._original_text: str = ""
        self._showing_diff: bool = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="file-preview-toolbar"):
            yield Static("[dim]No file open[/]", id="file-preview-title")
            yield Button("💾", id="save-file-btn")
            yield Button("⤢", id="toggle-code-focus-btn")
        yield PreviewTextArea("", id="file-editor")
        yield Static("", id="file-diff-view")

    def on_mount(self) -> None:
        editor = self.query_one("#file-editor", PreviewTextArea)
        editor.soft_wrap = False
        try:
            editor.show_line_numbers = True
        except Exception:
            pass
        self.query_one("#file-diff-view", Static).add_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-file-btn":
            self.action_save_file()
        elif event.button.id == "toggle-code-focus-btn":
            self.app.action_toggle_code_focus()

    def show_file(self, path: str) -> None:
        self.add_class("visible")
        editor = self.query_one("#file-editor", PreviewTextArea)
        diff_view = self.query_one("#file-diff-view", Static)
        title = self.query_one("#file-preview-title", Static)
        self._showing_diff = False
        self._current_path = path
        try:
            text = Path(path).read_text(encoding="utf-8")
            self._original_text = text
            # Detect and apply syntax highlighting language
            lang = detect_language(path)
            if lang is not None:
                try:
                    editor.language = lang
                except Exception:
                    pass
            else:
                try:
                    editor.language = None
                except Exception:
                    pass
            editor.text = text
            editor.remove_class("hidden")
            diff_view.add_class("hidden")
            lang_label = f" [dim]({lang})[/]" if lang else ""
            title.update(f"[bold]{Path(path).name}[/]{lang_label} [dim]{path}[/]")
            editor.focus()
        except Exception as e:
            editor.add_class("hidden")
            diff_view.remove_class("hidden")
            diff_view.update(f"[bold red]Failed to open {path}:[/] {e}")
            title.update(f"[bold red]Open failed[/] [dim]{path}[/]")

    def hide_file(self) -> None:
        self.remove_class("visible")
        self._current_path = None
        self._original_text = ""
        self._showing_diff = False
        editor = self.query_one("#file-editor", PreviewTextArea)
        diff_view = self.query_one("#file-diff-view", Static)
        title = self.query_one("#file-preview-title", Static)
        editor.text = ""
        editor.remove_class("hidden")
        diff_view.update("")
        diff_view.add_class("hidden")
        title.update("[dim]No file open[/]")

    def show_diff(self, path: str, diff_text: str) -> None:
        self.add_class("visible")
        editor = self.query_one("#file-editor", PreviewTextArea)
        diff_view = self.query_one("#file-diff-view", Static)
        title = self.query_one("#file-preview-title", Static)
        self._showing_diff = True
        self._current_path = path
        editor.add_class("hidden")
        diff_view.remove_class("hidden")
        diff_view.update(ChatPanel._colorize_diff(diff_text or "(no diff available for this file in this session)"))
        title.update(f"[bold]Diff[/] [dim]{path}[/]")

    def action_save_file(self) -> None:
        if not self._current_path or self._showing_diff:
            return
        editor = self.query_one("#file-editor", PreviewTextArea)
        updated_text = editor.text
        if updated_text == self._original_text:
            return
        path_obj = Path(self._current_path)
        path_obj.write_text(updated_text, encoding="utf-8")
        diff_text = "\n".join(
            unified_diff(
                self._original_text.splitlines(),
                updated_text.splitlines(),
                fromfile=f"a/{path_obj.as_posix()}",
                tofile=f"b/{path_obj.as_posix()}",
                lineterm="",
            )
        )
        self._original_text = updated_text
        self.post_message(self.FileSaved(path=self._current_path, diff_text=diff_text))
        self.query_one("#file-preview-title", Static).update(
            f"[bold]{path_obj.name}[/] [green](saved)[/] [dim]{self._current_path}[/]"
        )


class StatusFooter(Static):
    """Compact always-visible session stats at the bottom."""

    def update_stats(
        self,
        *,
        context_used: int,
        context_window: int,
        available: int,
        total_cost: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        pct = (context_used / context_window * 100.0) if context_window else 0.0
        self.update(
            f"[bold]Ctx[/] {context_used:,}/{context_window:,} ({pct:.0f}%) • "
            f"[bold]Avail[/] {available:,} • "
            f"[bold]Cost[/] ${total_cost:.4f} • "
            f"[bold]I/O[/] {input_tokens:,}/{output_tokens:,}"
        )


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
                "[cyan]/plan[/]        Generate and stage implementation plan",
                "[cyan]/approve[/]     Approve pending plan",
                "[cyan]/reject[/]      Reject pending plan",
                "[cyan]/status[/]      Session and token stats",
                "[cyan]/cost[/]        Cost breakdown",
                "[cyan]/keys[/]        API key status",
                "[cyan]/config[/]      View/set configuration",
                "[cyan]/fetch[/]       Fetch latest OpenRouter models",
                "[cyan]/context[/]     Refresh .foreman/context.json",
                "[cyan]/clear[/]       Clear chat",
                "[cyan]/new[/]         New session",
                "[cyan]/resume[/]      Resume a past session",
                "[cyan]/quit[/]        Exit Foreman",
                "",
                "[bold]Keybindings[/]",
                "[cyan]Ctrl+L[/]       Model selector",
                "[cyan]Shift+Tab[/]    Cycle thinking effort",
                "[cyan]Ctrl+O[/]       Toggle full tool output",
                "[cyan]Ctrl+D[/]       Toggle diff/source in file preview",
                "[cyan]Ctrl+B[/]       Minimize/expand token-cost panel",
                "[cyan]Ctrl+E[/]       Toggle code focus size",
                "[cyan]Ctrl+S[/]       Save open code file",
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
