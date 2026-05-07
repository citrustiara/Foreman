"""Fetch available models from OpenRouter API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from foreman.config import get_global_foreman_dir


logger = logging.getLogger("foreman.models.fetcher")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"


async def fetch_openrouter_models() -> list[dict[str, Any]]:
    """Fetch model list from OpenRouter API.

    Returns a list of dicts with keys: id, name, context_length, pricing, etc.
    """
    headers = {
        "HTTP-Referer": "https://github.com/foreman",
        "X-Title": "Foreman CLI",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(OPENROUTER_API_URL, headers=headers)
            response.raise_for_status()
            data = response.json()
            raw_models = data.get("data", [])

            # Normalize to our format
            models = []
            for m in raw_models:
                pricing = m.get("pricing", {})
                # OpenRouter pricing is per token, convert to per-million
                prompt_price = float(pricing.get("prompt", 0)) * 1_000_000
                completion_price = float(pricing.get("completion", 0)) * 1_000_000

                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("name", m.get("id", "")),
                    "context_window": m.get("context_length", 128000),
                    "cost_per_1m_input": round(prompt_price, 4),
                    "cost_per_1m_output": round(completion_price, 4),
                })

            # Sort by name
            models.sort(key=lambda x: x["name"].lower())
            logger.info("Fetched %d models from OpenRouter", len(models))
            return models

    except httpx.HTTPError as e:
        logger.error("Failed to fetch OpenRouter models: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error fetching models: %s", e)
        return []


def save_models(models: list[dict], foreman_dir: Path | None = None) -> None:
    """Save models to openrouter_models.json (defaults to global dir)."""
    if foreman_dir is None:
        foreman_dir = get_global_foreman_dir()
    
    foreman_dir.mkdir(parents=True, exist_ok=True)

    path = foreman_dir / "openrouter_models.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"models": models, "fetched_count": len(models)}, f, indent=2)
    logger.info("Saved %d models to %s", len(models), path)


def load_models(foreman_dir: Path | None = None) -> list[dict]:
    """Load cached models from disk (defaults to global dir)."""
    if foreman_dir is None:
        foreman_dir = get_global_foreman_dir()
    
    path = foreman_dir / "openrouter_models.json"

    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", [])
    except (json.JSONDecodeError, KeyError):
        return []
