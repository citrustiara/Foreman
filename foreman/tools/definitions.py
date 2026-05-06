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
            "description": "Read the contents of a file. Returns the text content (truncated to 2000 lines or 50KB). Use offset/limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to project root or absolute)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
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
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit", 2000)
        logger.info("Tool read_file: %s (offset=%d, limit=%d)", path, offset, limit)

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
            # Apply offset (1-indexed) and limit
            selected = lines[offset - 1 : offset - 1 + limit]
            result = "\n".join(selected)

            # Truncate if too large
            if len(result) > 50_000:
                result = result[:50_000] + "\n... (truncated)"

            if not selected:
                return f"(empty or beyond file end, file has {len(lines)} lines)"
            return result
        except Exception as e:
            return f"Error reading file: {e}"

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
