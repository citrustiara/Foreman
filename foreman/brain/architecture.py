"""Project context ingestion/building with dense JSON brain files."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger("foreman.brain.architecture")

CONTEXT_FILE = "context.json"
MAX_SNAPSHOT_FILES = 60
MAX_SNAPSHOT_CHARS_PER_FILE = 700
_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}

_DEF_PATTERNS = [
    re.compile(r"^\s*async\s+def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*function\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", re.MULTILINE),
]

EMPTY_CONTEXT: dict[str, Any] = {
    "version": 1,
    "generated_at": "",
    "summary": "",
    "entrypoints": [],
    "modules": [],
    "flows": [],
    "dependencies": [],
    "commands": [],
    "notes": [],
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_defs(text: str) -> list[str]:
    defs: list[str] = []
    for pattern in _DEF_PATTERNS:
        defs.extend(pattern.findall(text))
    return _ordered_unique(defs)


def _extract_json(response: str) -> dict[str, Any]:
    fenced = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{", response):
        try:
            return json.loads(response[match.start():])
        except json.JSONDecodeError:
            continue
    return {}


def _collect_candidate_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode == 0:
            files = [(repo_root / line.strip()) for line in result.stdout.splitlines() if line.strip()]
            return [p for p in files if p.suffix.lower() in _TEXT_EXTENSIONS and p.exists() and p.is_file()]
    except Exception:
        pass

    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in _TEXT_EXTENSIONS
            and ".git" not in path.parts
            and ".foreman" not in path.parts
            and "node_modules" not in path.parts
            and "__pycache__" not in path.parts
        ):
            files.append(path)
    return files


def _build_snapshot(repo_root: Path) -> dict[str, Any]:
    files = _collect_candidate_files(repo_root)[:MAX_SNAPSHOT_FILES]
    sampled: list[dict[str, Any]] = []

    for file_path in files:
        rel = file_path.relative_to(repo_root).as_posix()
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        snippet = text[:MAX_SNAPSHOT_CHARS_PER_FILE]
        sampled.append(
            {
                "path": rel,
                "defs": _extract_defs(snippet)[:20],
                "snippet": snippet,
            }
        )

    root_files = []
    for name in ("README.md", "pyproject.toml", "package.json", "go.mod"):
        p = repo_root / name
        if p.exists():
            try:
                root_files.append({"path": name, "content": p.read_text(encoding="utf-8")[:2000]})
            except UnicodeDecodeError:
                continue

    return {"sampled_files": sampled, "root_files": root_files}


async def load_context_json(foreman_dir: Path) -> dict[str, Any]:
    """Load .foreman/context.json, returning an empty template if missing/invalid."""
    path = foreman_dir / CONTEXT_FILE
    if not path.exists():
        return dict(EMPTY_CONTEXT)
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        if not isinstance(data, dict):
            return dict(EMPTY_CONTEXT)
        return {**EMPTY_CONTEXT, **data}
    except Exception:
        logger.warning("Failed to parse context JSON at %s; resetting to template", path)
        return dict(EMPTY_CONTEXT)


async def save_context_json(foreman_dir: Path, context: dict[str, Any]) -> None:
    """Persist .foreman/context.json atomically."""
    path = foreman_dir / CONTEXT_FILE
    tmp_path = foreman_dir / f"{CONTEXT_FILE}.tmp"
    payload = {**EMPTY_CONTEXT, **context, "generated_at": context.get("generated_at") or _utcnow()}
    async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp_path.replace(path)


async def load_structure(foreman_dir: Path) -> str:
    """Compatibility loader: return a dense JSON section for structure/entrypoints."""
    data = await load_context_json(foreman_dir)
    section = {
        "summary": data.get("summary", ""),
        "entrypoints": data.get("entrypoints", []),
        "modules": data.get("modules", []),
    }
    return json.dumps(section, indent=2, ensure_ascii=False)


async def load_logic(foreman_dir: Path) -> str:
    """Compatibility loader: return a dense JSON section for flow."""
    data = await load_context_json(foreman_dir)
    section = {"flows": data.get("flows", []), "notes": data.get("notes", [])}
    return json.dumps(section, indent=2, ensure_ascii=False)


async def load_context(foreman_dir: Path) -> str:
    """Compatibility loader: return a dense JSON section for deps/commands."""
    data = await load_context_json(foreman_dir)
    section = {
        "dependencies": data.get("dependencies", []),
        "commands": data.get("commands", []),
        "generated_at": data.get("generated_at", ""),
    }
    return json.dumps(section, indent=2, ensure_ascii=False)


async def refresh_context_json(
    repo_root: Path,
    foreman_dir: Path,
    router: Any,
    summary_model: Any,
) -> dict[str, Any]:
    """Generate dense project context JSON using the summary model."""
    snapshot = _build_snapshot(repo_root)

    prompt = (
        "You are generating compact, high-signal project context for an agentic coding assistant.\n"
        "Return JSON only. Keep it concise and factual.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "summary": "short project summary",\n'
        '  "entrypoints": ["path: role"],\n'
        '  "modules": [{"path":"", "purpose":"", "key_symbols":["..."]}],\n'
        '  "flows": ["step-by-step runtime/data flow bullets"],\n'
        '  "dependencies": ["dependency and why it matters"],\n'
        '  "commands": ["important dev commands"],\n'
        '  "notes": ["constraints, conventions, gotchas"]\n'
        "}\n"
        "Rules:\n"
        "- Do not include markdown.\n"
        "- Avoid long prose.\n"
        "- Prefer exact paths and symbols from provided data."
    )

    response = await router.generate(
        profile=summary_model,
        messages=[
            {"role": "system", "content": "You produce concise machine-readable JSON context."},
            {"role": "user", "content": f"{prompt}\n\nProject snapshot:\n{json.dumps(snapshot, ensure_ascii=False)}"},
        ],
        temperature=0.2,
        max_tokens=1800,
    )

    parsed = _extract_json(response)
    if not parsed:
        parsed = {"summary": "Context generation fallback: model returned non-JSON output.", "notes": [response[:1000]]}

    context = {**EMPTY_CONTEXT, **parsed, "generated_at": _utcnow()}
    await save_context_json(foreman_dir, context)
    return context
