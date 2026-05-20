"""Prompt registry.

Prompt templates live under the repo-root ``prompts/`` directory, grouped
by lifecycle phase:

    prompts/
    ├── phase1_initialization/   # *.system.j2 — agent personas
    ├── phase2_directive/        # expert.directive, trend.interpret*
    ├── phase3_reasoning/        # item.describe, user.profile, user.evaluate
    └── phase4_validation/       # expert.validate, expert.refine

The registry adds all four phase folders to Jinja's ``FileSystemLoader``
search path so callers can keep using flat dotted names — e.g.
``reg.render("user.evaluate", ...)`` resolves to
``prompts/phase3_reasoning/user.evaluate.j2``.

Usage
-----
>>> reg = PromptRegistry()
>>> reg.render("user.system", user_profile=..., directive=...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMPT_DIR = _REPO_ROOT / "prompts"

_PHASE_DIRS = (
    "phase1_initialization",
    "phase2_directive",
    "phase3_reasoning",
    "phase4_validation",
)


class PromptRegistry:
    """Loads & renders Jinja2 prompt templates with strict variable binding.

    Parameters
    ----------
    prompt_dir:
        Root directory containing the ``phase*`` template folders. Defaults
        to ``<repo>/prompts``.
    """

    def __init__(self, prompt_dir: Path | str = _DEFAULT_PROMPT_DIR) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.env = Environment(
            loader=FileSystemLoader(self._search_paths(self.prompt_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,
        )

    @staticmethod
    def _search_paths(root: Path) -> list[str]:
        paths: list[str] = []
        for phase in _PHASE_DIRS:
            p = root / phase
            if p.is_dir():
                paths.append(str(p))
        if not paths:
            # Fallback: flat layout (older checkouts / unit tests).
            paths.append(str(root))
        return paths

    def render(self, name: str, **vars: Any) -> str:
        """Render ``{name}.j2`` (the ``.j2`` suffix is added automatically)."""
        template = self.env.get_template(f"{name}.j2")
        return template.render(**vars)

    def has(self, name: str) -> bool:
        for base in self.env.loader.searchpath:  # type: ignore[union-attr]
            if (Path(base) / f"{name}.j2").exists():
                return True
        return False
