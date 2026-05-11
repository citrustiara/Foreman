"""Foreman CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import sys
from pathlib import Path

from foreman.config import ForemanConfig
from foreman.init import init_project
from foreman.models.keys import load_env
from foreman.models.profiles import resolve_profile
from foreman.models.router import ModelRouter
from foreman.brain.architecture import refresh_context_json


def _configure_event_loop_policy() -> None:
    """Use selector policy on Windows to avoid Proactor SSL teardown crashes."""
    if platform.system() == "Windows" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _run_async(coro):
    """Run a coroutine in a dedicated loop with explicit cleanup."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        # Give transports a chance to flush/close before loop teardown (Windows SSL).
        loop.run_until_complete(asyncio.sleep(0.2))
        return result
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)


def main() -> None:
    """Main CLI entry point."""
    _configure_event_loop_policy()
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

    # context — build dense project context JSON
    context_parser = subparsers.add_parser("context", help="Project context operations")
    context_parser.add_argument("action", choices=["build"], help="Context action")
    context_parser.add_argument(
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
    if not args.debug:
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    repo_root = Path(args.repo) if getattr(args, "repo", None) else Path.cwd()

    if args.command == "start":
        _run_tui(repo_root, hydrate_context_on_start=True)
    elif args.command == "init":
        _run_init(repo_root)
    elif args.command == "config":
        _run_config(repo_root, args.key, args.value)
    elif args.command == "sessions":
        _run_sessions(repo_root, args.action)
    elif args.command == "context":
        _run_context(repo_root, args.action)
    else:
        # Default launch: no automatic background hydration
        _run_tui(repo_root, hydrate_context_on_start=False)


def _run_tui(repo_root: Path, hydrate_context_on_start: bool) -> None:
    """Launch the Textual TUI."""
    from foreman.tui.app import ForemanApp

    # Check if .foreman exists, offer to init
    foreman_dir = repo_root / ".foreman"
    if not foreman_dir.exists():
        print(f"Initializing .foreman/ in {repo_root}...")
        init_project(repo_root)

    app = ForemanApp(repo_root=repo_root, hydrate_context_on_start=hydrate_context_on_start)
    app.run()


def _run_init(repo_root: Path) -> None:
    """Initialize the .foreman/ directory."""
    foreman_dir = init_project(repo_root)
    context_status = _build_context(repo_root)
    print(f"Created {foreman_dir}/")
    print("  - config.json")
    print("  - sessions/")
    print("  - compactions/")
    print("  - context.json")
    print(f"  - context build: {context_status}")
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
        sessions = _run_async(store.list_sessions())
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


def _build_context(repo_root: Path) -> str:
    """Generate .foreman/context.json using the configured summary model."""
    try:
        load_env(repo_root)
        config = ForemanConfig.load(repo_root)
        router = ModelRouter()
        profile = resolve_profile(config.secondary_model)
        _run_async(refresh_context_json(repo_root, repo_root / ".foreman", router, profile))
        return f"ok ({config.secondary_model})"
    except Exception as e:
        return f"failed ({e})"


def _run_context(repo_root: Path, action: str) -> None:
    """Manage dense project context files."""
    init_project(repo_root)
    if action == "build":
        status = _build_context(repo_root)
        print(f"context.json build {status}")


if __name__ == "__main__":
    main()
