"""Foreman CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from foreman.config import ForemanConfig
from foreman.init import init_project


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="foreman",
        description="Foreman — Agentic CLI coding assistant",
    )
    parser.add_argument("--version", action="version", version="foreman 0.1.0")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start — launch TUI
    start_parser = subparsers.add_parser("start", help="Launch the Foreman TUI")
    start_parser.add_argument(
        "--repo", type=str, default=None, help="Repository root directory (default: cwd)"
    )

    # init — create .foreman scaffold
    init_parser = subparsers.add_parser("init", help="Initialize .foreman/ directory")
    init_parser.add_argument(
        "--repo", type=str, default=None, help="Repository root directory (default: cwd)"
    )

    # config — view/set config
    config_parser = subparsers.add_parser("config", help="View or set configuration")
    config_parser.add_argument("key", nargs="?", help="Config key to view/set")
    config_parser.add_argument("value", nargs="?", help="Config value to set")
    config_parser.add_argument(
        "--repo", type=str, default=None, help="Repository root directory (default: cwd)"
    )

    # sessions — list sessions
    sessions_parser = subparsers.add_parser("sessions", help="Manage sessions")
    sessions_parser.add_argument("action", choices=["list"], help="Session action")
    sessions_parser.add_argument(
        "--repo", type=str, default=None, help="Repository root directory (default: cwd)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_root = Path(args.repo) if getattr(args, "repo", None) else Path.cwd()

    if args.command == "start":
        _run_tui(repo_root)
    elif args.command == "init":
        _run_init(repo_root)
    elif args.command == "config":
        _run_config(repo_root, args.key, args.value)
    elif args.command == "sessions":
        _run_sessions(repo_root, args.action)
    else:
        # Default: launch TUI
        _run_tui(repo_root)


def _run_tui(repo_root: Path) -> None:
    """Launch the Textual TUI."""
    from foreman.tui.app import ForemanApp

    # Check if .foreman exists, offer to init
    foreman_dir = repo_root / ".foreman"
    if not foreman_dir.exists():
        print(f"Initializing .foreman/ in {repo_root}...")
        init_project(repo_root)

    app = ForemanApp(repo_root=repo_root)
    app.run()


def _run_init(repo_root: Path) -> None:
    """Initialize the .foreman/ directory."""
    foreman_dir = init_project(repo_root)
    print(f"Created {foreman_dir}/")
    print("  - config.json")
    print("  - sessions/")
    print("  - compactions/")
    print("  - structure.mmd")
    print("  - logic.mmd")
    print("\nRun 'foreman start' to launch the TUI.")


def _run_config(repo_root: Path, key: str | None, value: str | None) -> None:
    """View or set configuration."""
    config = ForemanConfig.load(repo_root)

    if key is None:
        # Show all config
        from dataclasses import asdict
        print(json.dumps(asdict(config), indent=2))
    elif value is None:
        # Show single key
        if hasattr(config, key):
            print(f"{key}: {getattr(config, key)}")
        else:
            print(f"Unknown config key: {key}", file=sys.stderr)
            sys.exit(1)
    else:
        # Set config value
        try:
            config.update(key, value)
            config.save(repo_root)
            print(f"Set {key} = {getattr(config, key)}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def _run_sessions(repo_root: Path, action: str) -> None:
    """Manage sessions."""
    if action == "list":
        from foreman.brain.session import SessionStore

        store = SessionStore(repo_root / ".foreman" / "sessions")
        sessions = asyncio.run(store.list_sessions())
        if not sessions:
            print("No sessions found.")
            return

        for s in sessions:
            print(
                f"  {s['session_id'][:12]}...  "
                f"messages={s['message_count']}  "
                f"tokens={s['total_tokens']:,}  "
                f"updated={s.get('updated_at', 'N/A')}"
            )


if __name__ == "__main__":
    main()