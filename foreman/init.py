"""Project initialization — creates .foreman/ scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from foreman.config import ForemanConfig


FOREMAN_DIR = ".foreman"

TEMPLATE_STRUCTURE = """graph TD
    A[Your Module] --> B[Submodule]
    A --> C[Utilities]
    B --> D[Helpers]
"""

TEMPLATE_LOGIC = """sequenceDiagram
    participant User
    participant System
    User->>System: Request
    System->>System: Process
    System->>User: Response
"""


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

    # Create template Mermaid files
    structure_path = foreman_dir / "structure.mmd"
    if not structure_path.exists():
        structure_path.write_text(TEMPLATE_STRUCTURE.strip(), encoding="utf-8")

    logic_path = foreman_dir / "logic.mmd"
    if not logic_path.exists():
        logic_path.write_text(TEMPLATE_LOGIC.strip(), encoding="utf-8")

    # Create .gitkeep files in empty dirs
    for subdir in ("sessions", "compactions"):
        gitkeep = foreman_dir / subdir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")

    return foreman_dir
