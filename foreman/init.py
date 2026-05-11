"""Project initialization — creates .foreman/ scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from foreman.config import ForemanConfig
from foreman.brain.architecture import EMPTY_CONTEXT


FOREMAN_DIR = ".foreman"


def init_project(repo_root: Path, config: ForemanConfig | None = None) -> Path:
    """Create the .foreman/ directory structure with template files.

    Args:
        repo_root: Root directory of the project.
        config: Optional configuration to write. Uses defaults if None.

    Returns:
        Path to the created .foreman/ directory.
    """
    foreman_dir = repo_root / FOREMAN_DIR

    # Create directories
    (foreman_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (foreman_dir / "compactions").mkdir(parents=True, exist_ok=True)

    # Create config.json
    if config is None:
        config = ForemanConfig()
    config.save(repo_root)

    # Create dense JSON context file
    context_path = foreman_dir / "context.json"
    if not context_path.exists():
        context_path.write_text(json.dumps(EMPTY_CONTEXT, indent=2), encoding="utf-8")

    # Create .gitkeep files in empty dirs
    for subdir in ("sessions", "compactions"):
        gitkeep = foreman_dir / subdir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")

    return foreman_dir
