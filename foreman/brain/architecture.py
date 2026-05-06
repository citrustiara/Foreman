"""Architecture ingestion — loading Mermaid brain files."""

from __future__ import annotations

import logging
from pathlib import Path

import aiofiles

logger = logging.getLogger("foreman.brain.architecture")


async def load_mermaid(path: Path) -> str:
    """Load a Mermaid file, returning empty string with warning if missing."""
    if not path.exists():
        logger.warning("Mermaid file not found: %s", path)
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def load_structure(foreman_dir: Path) -> str:
    """Load .foreman/structure.mmd."""
    return await load_mermaid(foreman_dir / "structure.mmd")


async def load_logic(foreman_dir: Path) -> str:
    """Load .foreman/logic.mmd."""
    return await load_mermaid(foreman_dir / "logic.mmd")


async def load_context(foreman_dir: Path) -> str:
    """Load .foreman/context.mmd if it exists."""
    return await load_mermaid(foreman_dir / "context.mmd")


def generate_context_mmd(repo_root: Path) -> str:
    """Walk the repo tree and produce a simple Mermaid directory dependency graph.

    This is a basic implementation that maps the directory structure.
    A future version could parse import statements for real dependency edges.
    """
    lines = ["graph TD"]
    node_id = 0
    path_to_id: dict[str, str] = {}

    # Skip hidden dirs and common non-source dirs
    skip_dirs = {".git", ".foreman", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}

    def _visit(directory: Path, parent_id: str | None = None) -> None:
        nonlocal node_id
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name not in (".foreman",):
                continue
            if entry.is_dir() and entry.name in skip_dirs:
                continue
            if entry.is_dir():
                node_id += 1
                nid = f"D{node_id}"
                path_to_id[str(entry)] = nid
                lines.append(f"    {nid}[{entry.name}/]")
                if parent_id:
                    lines.append(f"    {parent_id} --> {nid}")
                _visit(entry, nid)
            elif entry.is_file() and entry.suffix in (".py", ".js", ".ts", ".go", ".rs"):
                node_id += 1
                fid = f"F{node_id}"
                lines.append(f"    {fid}[{entry.name}]")
                if parent_id:
                    lines.append(f"    {parent_id} --> {fid}")

    _visit(repo_root)
    return "\n".join(lines) if len(lines) > 1 else ""
