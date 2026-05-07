"""Tool definitions for the agentic loop."""

from __future__ import annotations

import platform
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("foreman.tools")

# Find bash executable for Windows (Git Bash)
def _find_bash() -> str | None:
    """Find a bash executable, preferring Git Bash on Windows."""
    if platform.system() != "Windows":
        return None  # Let subprocess use default shell
    bash = shutil.which("bash")
    if bash:
        return bash
    # Check common Git Bash locations
    for candidate in [
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
        r"C:\Git\usr\bin\bash.exe",
    ]:
        if Path(candidate).exists():
            return candidate
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
            "description": "Read the contents of a file. Returns text (truncated to 30KB). For files > 100 lines, you MUST use start_line and end_line. Reading entire large files is a waste of tokens and context space.",
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
                },
                "required": ["path"],
            },

        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing content.",
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
        
        logger.info("Tool read_file: %s (lines %d-%d)", path, start_line, end_line)

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
            result = "\n".join(selected)

            # Truncate if too large (reduced limit to 30KB)
            if len(result) > 30_000:
                result = result[:30_000] + "\n... (truncated)"

            if not selected:
                return f"(empty or beyond file end, file has {len(lines)} lines)"
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
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    else:
        return f"Unknown tool: {name}"
