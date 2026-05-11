"""Plan generation — uses pydantic-ai for structured plan output."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from foreman.models.profiles import ModelProfile
from foreman.models.router import ModelRouter

logger = logging.getLogger("foreman.implement.planner")


class PlanStep(BaseModel):
    """A single step in an implementation plan."""
    step_number: int
    description: str
    files_to_modify: list[str]
    files_to_create: list[str]
    dependencies: list[int]  # step numbers this depends on


class ImplementationPlan(BaseModel):
    """Structured output from the planner agent."""
    feature_name: str
    overview: str
    steps: list[PlanStep]
    risks: list[str]
    estimated_complexity: Literal["small", "medium", "large"]


PLANNER_SYSTEM_PROMPT = """You are a planning agent for a coding assistant called Foreman.
Given a feature description and the project's architecture, create a structured implementation plan.

Rules:
- Each step should be atomic and testable
- List all files that need to be modified or created
- Identify dependencies between steps
- Assess risks honestly
- Be specific about what changes are needed, not vague

Output your plan as a structured JSON matching the ImplementationPlan schema."""


PLANNER_USER_TEMPLATE = """## Feature Request
{feature_description}

## Project Context
{architecture}

## Directory Structure
{directory_tree}

Create a detailed implementation plan for this feature."""


class Planner:
    """Generates implementation plans using the primary model."""

    def __init__(self, router: ModelRouter):
        self.router = router

    async def generate_plan(
        self,
        feature_description: str,
        architecture: str = "",
        directory_tree: str = "",
        primary_model: ModelProfile | None = None,
        reasoning_effort: str | None = None,
    ) -> ImplementationPlan:
        """Generate a structured implementation plan.

        Args:
            feature_description: What to implement.
            architecture: Dense project context JSON / metadata.
            directory_tree: Project directory structure.
            primary_model: Model to use. Defaults to gemini/gemini-2.5-pro.

        Returns:
            Structured ImplementationPlan.
        """
        from foreman.models.profiles import resolve_profile

        if primary_model is None:
            primary_model = resolve_profile("gemini/gemini-2.5-pro")

        user_content = PLANNER_USER_TEMPLATE.format(
            feature_description=feature_description,
            architecture=architecture or "(no architecture files loaded)",
            directory_tree=directory_tree or "(no directory tree available)",
        )

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Request JSON output
        generation_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        response = await self.router.generate(
            profile=primary_model,
            messages=messages,
            temperature=0.3,
            **generation_kwargs,
        )

        # Parse the response into an ImplementationPlan
        plan = self._parse_plan_response(response)
        plan.feature_name = feature_description
        return plan

    @staticmethod
    def _parse_plan_response(response: str) -> ImplementationPlan:
        """Parse the LLM response into an ImplementationPlan.

        Tries JSON parsing first, falls back to a minimal plan from the raw text.
        """
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return ImplementationPlan(**data)
            except Exception:
                pass

        # Try parsing the whole response as JSON
        try:
            data = json.loads(response)
            return ImplementationPlan(**data)
        except Exception:
            pass

        # Fallback: create a minimal plan from the raw text
        return ImplementationPlan(
            feature_name="",
            overview=response[:500],
            steps=[
                PlanStep(
                    step_number=1,
                    description=response[:200],
                    files_to_modify=[],
                    files_to_create=[],
                    dependencies=[],
                )
            ],
            risks=["Plan was auto-parsed from unstructured response"],
            estimated_complexity="medium",
        )

    @staticmethod
    def plan_to_markdown(plan: ImplementationPlan) -> str:
        """Convert an ImplementationPlan to Markdown for display/storage."""
        lines = [
            f"# Implementation Plan: {plan.feature_name}",
            "",
            f"**Overview:** {plan.overview}",
            f"**Complexity:** {plan.estimated_complexity}",
            "",
            "## Steps",
            "",
        ]
        for step in plan.steps:
            lines.append(f"### Step {step.step_number}: {step.description}")
            if step.files_to_modify:
                lines.append(f"- **Modify:** {', '.join(step.files_to_modify)}")
            if step.files_to_create:
                lines.append(f"- **Create:** {', '.join(step.files_to_create)}")
            if step.dependencies:
                lines.append(f"- **Depends on:** Step(s) {', '.join(map(str, step.dependencies))}")
            lines.append("")

        if plan.risks:
            lines.append("## Risks")
            for risk in plan.risks:
                lines.append(f"- {risk}")
            lines.append("")

        return "\n".join(lines)
