"""Baseline retrievers / rankers reproduced for ablation studies.

Heavyweight baselines (LightGCN, AgentCF, MACF) live here as stub modules
to be filled in during Step 3 of the plan; their interfaces are pinned now
so the rest of the pipeline (and the eval harness) can be wired against
them today.
"""

from baselines.lightgcn import LightGCNRetriever, train_lightgcn

__all__ = ["LightGCNRetriever", "train_lightgcn"]
