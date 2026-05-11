"""Clean sub-session execution for the plan/approve pipeline."""

from __future__ import annotations

import logging
from typing import Any

from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.models.router import ModelRouter

logger = logging.getLogger("foreman.implement.executor")

EXECUTOR_SYSTEM_PROMPT = """You are an expert code implementation agent. You will be given:
1. An approved implementation plan
2. The relevant source files
3. The project's dense context metadata

Your task is to implement the plan by producing code changes.

Output format for each file change:
### file: path/to/file.py
<<<<<<< SEARCH
exact text to find
=======
replacement text
>>>>>>> REPLACE

Rules:
- Use EXACT string matching for SEARCH blocks — copy the original text precisely
- One SEARCH/REPLACE block per logical change
- If creating a new file, omit the SEARCH block and just provide the full content
- Preserve indentation exactly
- Do not add unrelated changes
"""


class Executor:
    """Creates a clean sub-session and executes an implementation plan."""

    def __init__(self, router: ModelRouter):
        self.router = router

    async def execute_plan(
        self,
        plan_markdown: str,
        relevant_files: dict[str, str],
        architecture: str = "",
        primary_model: ModelProfile | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Execute an implementation plan in a clean sub-session.

        Args:
            plan_markdown: The approved plan in Markdown format.
            relevant_files: Dict mapping file paths to their current contents.
            architecture: Dense project context JSON / metadata.
            primary_model: Model to use for generation.

        Returns:
            The raw model output containing SEARCH/REPLACE blocks.
        """
        if primary_model is None:
            primary_model = resolve_profile("gemini/gemini-2.5-pro")

        # Build a clean message list with ONLY:
        # 1. System prompt
        # 2. Architecture
        # 3. The plan
        # 4. Current file contents
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        ]

        if architecture:
            messages.append({
                "role": "user",
                "content": f"## Project Context\n{architecture}",
            })

        # Build the implementation request
        file_contents = ""
        for path, content in relevant_files.items():
            file_contents += f"\n### {path}\n```\n{content}\n```\n"

        implementation_request = f"""## Implementation Plan
{plan_markdown}

## Current File Contents
{file_contents if file_contents else "(no files provided)"}

Implement this plan. Output all changes as SEARCH/REPLACE blocks."""

        messages.append({"role": "user", "content": implementation_request})

        logger.info(
            "Executing plan with %s (%d files, %d messages)",
            primary_model.litellm_model,
            len(relevant_files),
            len(messages),
        )

        generation_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        response = await self.router.generate(
            profile=primary_model,
            messages=messages,
            temperature=0.2,  # Low temperature for precise code generation
            **generation_kwargs,
        )

        return response
