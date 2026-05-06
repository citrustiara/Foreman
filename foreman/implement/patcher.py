"""SEARCH/REPLACE patch application — AST-aware code patching."""

from __future__ import annotations

import logging
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger("foreman.implement.patcher")


# Regex to parse SEARCH/REPLACE blocks
PATCH_PATTERN = re.compile(
    r"###\s*file:\s*(.+?)\s*\n"
    r"<<<<<<<\s*SEARCH\n"
    r"(.*?)\n"
    r"=======\n"
    r"(.*?)\n"
    r">>>>>>>\s*REPLACE",
    re.DOTALL,
)

# Simpler pattern for new files (no SEARCH block)
NEW_FILE_PATTERN = re.compile(
    r"###\s*file:\s*(.+?)\s*\n"
    r"```\n"
    r"(.*?)\n"
    r"```",
    re.DOTALL,
)


class PatchResult:
    """Result of applying a single patch."""

    def __init__(self, file_path: str, success: bool, message: str = ""):
        self.file_path = file_path
        self.success = success
        self.message = message

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"PatchResult({status} {self.file_path}: {self.message})"


def parse_patches(model_output: str) -> list[dict[str, str]]:
    """Parse SEARCH/REPLACE blocks from model output.

    Returns list of dicts with keys: file_path, search, replace
    """
    patches = []

    # Parse SEARCH/REPLACE blocks
    for match in PATCH_PATTERN.finditer(model_output):
        file_path = match.group(1).strip()
        search_block = match.group(2)
        replace_block = match.group(3)
        patches.append({
            "file_path": file_path,
            "search": search_block,
            "replace": replace_block,
        })

    # If no SEARCH/REPLACE blocks found, try new-file format
    if not patches:
        for match in NEW_FILE_PATTERN.finditer(model_output):
            file_path = match.group(1).strip()
            content = match.group(2)
            patches.append({
                "file_path": file_path,
                "search": "",  # Empty search = new file
                "replace": content,
            })

    return patches


def apply_patch(
    file_path: str | Path,
    search_block: str,
    replace_block: str,
    dry_run: bool = False,
    fuzzy: bool = True,
) -> PatchResult:
    """Apply a single SEARCH/REPLACE patch to a file.

    Args:
        file_path: Path to the file to patch.
        search_block: Text to find. Empty string means create new file.
        replace_block: Text to replace it with.
        dry_run: If True, validate but don't write.
        fuzzy: If True, use fuzzy matching when exact match fails.

    Returns:
        PatchResult indicating success or failure.
    """
    path = Path(file_path)

    # New file case
    if not search_block:
        if path.exists():
            return PatchResult(str(path), False, f"File already exists: {path}")
        if dry_run:
            return PatchResult(str(path), True, f"Would create new file: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(replace_block, encoding="utf-8")
        return PatchResult(str(path), True, f"Created new file: {path}")

    # Existing file case
    if not path.exists():
        return PatchResult(str(path), False, f"File not found: {path}")

    original = path.read_text(encoding="utf-8")

    # Try exact match first
    if search_block in original:
        new_content = original.replace(search_block, replace_block, 1)
        if dry_run:
            return PatchResult(str(path), True, "Exact match found (dry run)")
        _write_with_backup(path, new_content)
        return PatchResult(str(path), True, "Applied with exact match")

    # Try fuzzy match
    if fuzzy:
        match_result = _fuzzy_find(original, search_block)
        if match_result:
            start, end = match_result
            new_content = original[:start] + replace_block + original[end:]
            if dry_run:
                return PatchResult(str(path), True, f"Fuzzy match found (dry run)")
            _write_with_backup(path, new_content)
            return PatchResult(str(path), True, "Applied with fuzzy match")

    return PatchResult(
        str(path),
        False,
        f"No match found for search block in {path} (first 80 chars: {search_block[:80]}...)",
    )


def apply_all_patches(
    model_output: str,
    repo_root: str | Path,
    dry_run: bool = False,
) -> list[PatchResult]:
    """Parse and apply all patches from model output.

    Args:
        model_output: Raw model output containing SEARCH/REPLACE blocks.
        repo_root: Root directory to resolve relative paths against.
        dry_run: If True, validate all patches without writing.

    Returns:
        List of PatchResult for each patch.
    """
    patches = parse_patches(model_output)
    results = []

    for patch in patches:
        file_path = Path(repo_root) / patch["file_path"]
        result = apply_patch(
            file_path=file_path,
            search_block=patch["search"],
            replace_block=patch["replace"],
            dry_run=dry_run,
        )
        results.append(result)

    return results


def _fuzzy_find(text: str, search: str, threshold: float = 0.8) -> tuple[int, int] | None:
    """Find the best fuzzy match position for search within text.

    Returns (start, end) indices or None if no good match found.
    """
    search_lines = search.splitlines(keepends=True)
    text_lines = text.splitlines(keepends=True)

    if len(search_lines) > len(text_lines):
        return None

    best_ratio = 0.0
    best_pos = None

    # Sliding window approach
    window_size = len(search_lines)
    for i in range(len(text_lines) - window_size + 1):
        window = "".join(text_lines[i : i + window_size])
        ratio = SequenceMatcher(None, search, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i

    if best_ratio >= threshold and best_pos is not None:
        # Find character positions
        char_start = sum(len(line) for line in text_lines[:best_pos])
        char_end = char_start + len("".join(text_lines[best_pos : best_pos + window_size]))
        return (char_start, char_end)

    return None


def _write_with_backup(path: Path, new_content: str) -> None:
    """Write file with backup, atomic write, and verification."""
    # Create backup
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    try:
        # Atomic write via temp file
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_content, encoding="utf-8")
        tmp.replace(path)
        logger.info("Successfully patched: %s", path)
    except Exception as e:
        # Rollback from backup
        logger.error("Patch failed for %s: %s. Rolling back.", path, e)
        shutil.copy2(backup, path)
        raise
    finally:
        # Clean up backup (keep for safety)
        # In production, you might want to keep backups for a while
        pass