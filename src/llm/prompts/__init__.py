"""Prompt registry.

System prompts live as ``.j2`` files next to this module. A ``PromptRegistry``
loads them lazily and renders them through Jinja2.

Usage
-----
>>> reg = PromptRegistry()
>>> reg.render("user_agent.system", user_profile=..., directive=...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROMPT_DIR = Path(__file__).parent


class PromptRegistry:
    """Loads & renders Jinja2 prompt templates with strict variable binding."""

    def __init__(self, prompt_dir: Path | str = _PROMPT_DIR) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def render(self, name: str, **vars: Any) -> str:
        """Render ``{name}.j2`` (the ``.j2`` suffix is added automatically)."""
        template = self.env.get_template(f"{name}.j2")
        return template.render(**vars)

    def has(self, name: str) -> bool:
        return (self.prompt_dir / f"{name}.j2").exists()
