"""Tool definitions for the agentic loop."""

from __future__ import annotations

import difflib
import platform
import re
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("foreman.tools")
READ_FILE_DEFAULT_MAX_BYTES = 2000
READ_FILE_MAX_OUTPUT_CHARS = 30_000
WRITE_DIFF_MAX_CHARS = 8_000

# Find bash executable for Windows (Git Bash)
# IMPORTANT: On Windows, `shutil.which("bash")` often returns
# C:\Windows\System32\bash.exe — the WSL launcher, NOT Git Bash.
# We must check real Git Bash paths FIRST and skip the WSL shim.

def _is_wsl_shim(path: str) -> bool:
    """Return True if the path is the Windows WSL bash launcher."""
    normalized = path.replace("/", "\\").lower()
    return "\\system32\\bash.exe" in normalized or "\\sysnative\\bash.exe" in normalized


def _find_bash() -> str | None:
    """Find a real bash executable on Windows, skipping the WSL shim."""
    if platform.system() != "Windows":
        return None  # Let subprocess use default shell

    # 1. Check well-known Git Bash install locations FIRST
    git_bash_candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
        r"C:\Git\bin\bash.exe",
        r"C:\Git\usr\bin\bash.exe",
    ]
    # Also check scoop installs: ~/scoop/apps/git/current/usr/bin/bash.exe
    import os
    scoop_base = Path(os.environ.get("USERPROFILE", "")) / "scoop" / "apps"
    if scoop_base.exists():
        for git_dir in (scoop_base / "git").glob("*/usr/bin/bash.exe"):
            git_bash_candidates.append(str(git_dir))
        # Scoop 'current' symlink
        git_bash_candidates.append(str(scoop_base / "git" / "current" / "usr" / "bin" / "bash.exe"))

    for candidate in git_bash_candidates:
        if Path(candidate).exists():
            logger.info("Found Git Bash at: %s", candidate)
            return candidate

    # 2. Fall back to shutil.which, but SKIP the WSL shim
    bash = shutil.which("bash")
    if bash and not _is_wsl_shim(bash):
        logger.info("Found bash via PATH: %s", bash)
        return bash

    logger.warning("No Git Bash found. Install Git for Windows to use the bash tool.")
    return None

_BASH_PATH = _find_bash()

# Tool schemas compatible with litellm/OpenAI function calling format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the project directory. "
                "Returns stdout and stderr (truncated to last 50KB). "
                "Use for running tests, git commands, find/grep, etc. "
                "Commands run in a bash shell."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read file contents safely. ALWAYS prefer byte-bounded reads with max_bytes=2000. "
                "Set max_bytes to 0 only when you explicitly need an unbounded/full read. "
                "You may set functions_only=true to return only defined/called function names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to project root)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed, default 1)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Line number to end reading at (inclusive, default start_line + 500)",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            "Maximum UTF-8 bytes to return from the selected range. "
                            "Default is 2000. Set to 0 for no byte limit."
                        ),
                    },
                    "functions_only": {
                        "type": "boolean",
                        "description": (
                            "If true, returns only defined and called function names "
                            "detected in the selected content."
                        ),
                    },
                },
                "required": ["path", "max_bytes"],
            },

        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories as needed and returns a unified diff snippet of the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to project root or absolute)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search for a pattern in all project files (grep). Returns matching lines with line numbers. Use this to find where functions or classes are defined.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern (regex supported)",
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob pattern for files to include (e.g. '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_last",
            "description": (
                "Summarize the output of the LAST tool call and replace it in history with your summary. "
                "Use this immediately after any tool call whose output is large but only partially relevant — "
                "for example after read_file, bash, or search_file. "
                "If instant=false (default), the replacement happens when the user's turn begins (deferred). "
                "If instant=true, the replacement happens NOW in the active context, freeing tokens for the rest of this turn. "
                "The summary should capture what was relevant, what was found/not found, and any key values or paths noted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of the last tool output. Include relevant findings.",
                    },
                    "instant": {
                        "type": "boolean",
                        "description": "If true, replace the last tool result in the active context immediately (saves tokens this turn). If false or omitted, deferred until user replies.",
                    },
                },
                "required": ["summary"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "checkpoint_summary",
            "description": (
                "Replace all messages since the LAST user message with this summary. "
                "Use this to 'clean up' your context after completing a major step, "
                "keeping only the high-level progress while preserving the user's original request. "
                "The summary will be the ONLY message from you since the user's last message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The concise summary of progress made so far.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


_DEF_PATTERNS = [
    re.compile(r"^\s*async\s+def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*function\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", re.MULTILINE),
    re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*lambda\b", re.MULTILINE),
]
_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_NON_CALL_NAMES = {
    "if",
    "for",
    "while",
    "with",
    "switch",
    "catch",
    "return",
    "print",
    "typeof",
    "sizeof",
    "new",
    "super",
}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _function_names_view(text: str, path: str) -> str:
    defined: list[str] = []
    for pattern in _DEF_PATTERNS:
        defined.extend(pattern.findall(text))
    defined = _ordered_unique(defined)

    called = _ordered_unique(
        [
            name
            for name in _CALL_PATTERN.findall(text)
            if name not in _NON_CALL_NAMES and not name.startswith("__")
        ]
    )

    lines = [f"Function summary for {path}", "Defined:"]
    lines.extend(f"- {name}" for name in defined) if defined else lines.append("- (none found)")
    lines.append("Called:")
    lines.extend(f"- {name}" for name in called) if called else lines.append("- (none found)")
    return "\n".join(lines)


def _render_write_diff(path: str, old_content: str, new_content: str, existed: bool) -> str:
    from_file = f"a/{path}" if existed else "/dev/null"
    to_file = f"b/{path}"
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=from_file,
            tofile=to_file,
            n=3,
        )
    )
    if not diff:
        return f"No textual diff for {path} (file content unchanged)."

    if len(diff) > WRITE_DIFF_MAX_CHARS:
        diff = diff[:WRITE_DIFF_MAX_CHARS] + "\n... (diff truncated)"
    return f"Updated {path}\n{diff}"




def execute_tool(
    name: str,
    arguments: dict[str, Any],
    cwd: str | Path,
    timeout: int = 120,
) -> str:
    """Execute a tool call and return the result as a string.

    Args:
        name: Tool name (bash, read_file, write_file).
        arguments: Tool arguments dict.
        cwd: Working directory for bash commands.
        timeout: Timeout in seconds for bash commands.

    Returns:
        Result string (stdout for bash, file contents for read, confirmation for write).
    """
    cwd = Path(cwd)

    if name == "bash":
        command = arguments["command"]
        logger.info("Tool bash: %s", command)
        try:
            # On Windows, use Git Bash explicitly so Unix commands work
            if platform.system() == "Windows" and _BASH_PATH:
                result = subprocess.run(
                    [_BASH_PATH, "-c", command],
                    capture_output=True,
                    text=True,
                    cwd=str(cwd),
                    timeout=timeout,
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(cwd),
                    timeout=timeout,
                )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += f"STDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            if not output:
                output = "(no output)"
            # Truncate to prevent blowing up the context
            if len(output) > 50_000:
                output = output[:50_000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

    elif name == "read_file":
        path = arguments["path"]
        start_line = arguments.get("start_line", 1)
        # Compatibility for old offset/limit if model still uses them
        if "offset" in arguments:
            start_line = arguments["offset"]
        
        limit = arguments.get("limit", 1000)
        end_line = arguments.get("end_line", start_line + limit - 1)
        raw_max_bytes = arguments.get("max_bytes", READ_FILE_DEFAULT_MAX_BYTES)
        max_bytes = READ_FILE_DEFAULT_MAX_BYTES if raw_max_bytes is None else int(raw_max_bytes)
        functions_only = bool(arguments.get("functions_only", False))
        
        logger.info(
            "Tool read_file: %s (lines %d-%d, max_bytes=%d, functions_only=%s)",
            path,
            start_line,
            end_line,
            max_bytes,
            functions_only,
        )

        # Resolve relative paths against cwd
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = cwd / file_path

        try:
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            # Skip binary files
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: Binary file, cannot read as text: {path}"

            lines = content.splitlines()
            # Apply range (1-indexed)
            # Ensure indices are within bounds
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            selected = lines[start_idx:end_idx]
            if not selected:
                return f"(empty or beyond file end, file has {len(lines)} lines)"

            selected_text = "\n".join(selected)
            truncated_by_bytes = False
            if max_bytes > 0:
                selected_raw = selected_text.encode("utf-8")
                if len(selected_raw) > max_bytes:
                    selected_text = selected_raw[:max_bytes].decode("utf-8", errors="ignore")
                    truncated_by_bytes = True

            result = _function_names_view(selected_text, path) if functions_only else selected_text
            if truncated_by_bytes and not functions_only:
                result += f"\n... (byte-truncated to {max_bytes} bytes)"

            # Final output cap
            if len(result) > READ_FILE_MAX_OUTPUT_CHARS:
                result = result[:READ_FILE_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            return result
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "search_file":
        pattern = arguments["pattern"]
        include = arguments.get("include", "")
        logger.info("Tool search_file: %s (include=%s)", pattern, include)

        try:
            # Prefer ripgrep if available, otherwise grep
            grep_cmd = ["rg", "--vimgrep", "--no-heading", "--smart-case"]
            if not shutil.which("rg"):
                grep_cmd = ["grep", "-rnEI"]
            
            cmd = [*grep_cmd, pattern]
            if include:
                if "rg" in grep_cmd[0]:
                    cmd.extend(["-g", include])
                else:
                    cmd.append(include)
            else:
                cmd.append(".")

            # On Windows, use Git Bash for grep if rg not found
            if platform.system() == "Windows" and _BASH_PATH and not shutil.which("rg"):
                bash_cmd = f"grep -rnEI '{pattern}' ."
                if include:
                    bash_cmd = f"grep -rnEI --include='{include}' '{pattern}' ."
                result = subprocess.run(
                    [_BASH_PATH, "-c", bash_cmd],
                    capture_output=True, text=True, cwd=str(cwd), timeout=30
                )
            else:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=str(cwd), timeout=30
                )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            if not output.strip():
                return f"No matches found for pattern: {pattern}"
            
            # Truncate if too large
            if len(output) > 20_000:
                output = output[:20_000] + "\n... (truncated)"
            return output
        except Exception as e:
            return f"Error searching files: {e}"


    elif name == "write_file":
        path = arguments["path"]
        content = arguments["content"]
        logger.info("Tool write_file: %s (%d chars)", path, len(content))

        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = cwd / file_path

        try:
            existed = file_path.exists()
            old_content = ""
            if existed:
                if not file_path.is_file():
                    return f"Error: Not a file: {path}"
                try:
                    old_content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    old_content = ""

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return _render_write_diff(path, old_content, content, existed)
        except Exception as e:
            return f"Error writing file: {e}"

    else:
        return f"Unknown tool: {name}"
