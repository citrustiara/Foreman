"""Foreman configuration loading and defaults."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "primary_model": "openrouter/xiaomi/mimo-v2.5-pro",
    "secondary_model": "openrouter/xiaomi/mimo-v2.5",
    "compact_threshold": 0.70,
    "output_reserve": 16_384,
    "max_summary_tokens": 2_000,
    "recent_messages_to_keep": 6,  # 3 exchanges
    "theme": "dark",
}


def get_global_foreman_dir() -> Path:
    """Get the global Foreman data directory (platform-specific)."""
    if platform.system() == "Windows":
        # Check for %APPDATA%
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Foreman"
    return Path.home() / ".foreman"



@dataclass
class ForemanConfig:
    """Runtime configuration for Foreman."""
    primary_model: str = DEFAULT_CONFIG["primary_model"]
    secondary_model: str = DEFAULT_CONFIG["secondary_model"]
    compact_threshold: float = DEFAULT_CONFIG["compact_threshold"]
    output_reserve: int = DEFAULT_CONFIG["output_reserve"]
    max_summary_tokens: int = DEFAULT_CONFIG["max_summary_tokens"]
    recent_messages_to_keep: int = DEFAULT_CONFIG["recent_messages_to_keep"]
    theme: str = DEFAULT_CONFIG["theme"]

    @classmethod
    def load(cls, repo_root: Path) -> "ForemanConfig":
        """Load config from .foreman/config.json, falling back to defaults."""
        config_path = repo_root / ".foreman" / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: saved values override defaults
            merged = {**DEFAULT_CONFIG, **saved}
            return cls(**{k: v for k, v in merged.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, repo_root: Path) -> None:
        """Persist current config to .foreman/config.json."""
        config_dir = repo_root / ".foreman"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def update(self, key: str, value: str) -> None:
        """Update a config field by name with string-to-type coercion."""
        if key not in self.__dataclass_fields__:
            raise ValueError(f"Unknown config key: {key!r}")
        field_type = self.__dataclass_fields__[key].type
        # Coerce string value to the appropriate type
        if field_type in ("int", int):
            setattr(self, key, int(value))
        elif field_type in ("float", float):
            setattr(self, key, float(value))
        elif field_type in ("bool", bool):
            setattr(self, key, value.lower() in ("true", "1", "yes"))
        else:
            setattr(self, key, value)