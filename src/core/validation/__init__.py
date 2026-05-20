"""Runtime validation of agent outputs.

Two failure modes are tracked here, each with a matching offline metric in
:mod:`eval.grounding`:

* **VDR** — Vocabulary Drift Rate (token-level)
* **SVR** — Schema Violation Rate (Pydantic structural)
"""

from core.validation.schema_validator import SchemaValidator
from core.validation.vocabulary import Vocabulary

__all__ = ["SchemaValidator", "Vocabulary"]
