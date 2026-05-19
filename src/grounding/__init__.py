"""Grounding layer — keeps the LLM agents within catalog & vocabulary bounds.

Three failure modes are tracked here, each of which has a corresponding
evaluation metric in :mod:`sage.evaluation.grounding`:

* **IHR** — Item Hallucination Rate (catalog containment)
* **VDR** — Vocabulary Drift Rate (token-level grounding)
* **SVR** — Schema Violation Rate (pydantic structural grounding)

The fourth governance-side metric, **CADR**, lives in the lifecycle module
since it counts cross-agent disagreements, not raw hallucinations.
"""

from grounding.schema_validator import SchemaValidator
from grounding.snapshot import TrendSnapshotStore
from grounding.vocabulary import Vocabulary

__all__ = ["SchemaValidator", "TrendSnapshotStore", "Vocabulary"]
