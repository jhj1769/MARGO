"""Candidate retrievers shared by MARGO & baselines."""

from adapters.retrieval.base import BaseRetriever, RetrievalHit
from adapters.retrieval.bm25_retriever import BM25Retriever

__all__ = ["BaseRetriever", "RetrievalHit", "BM25Retriever", "BGERetriever"]


def __getattr__(name: str):  # pragma: no cover — lazy heavy import
    if name == "BGERetriever":
        from adapters.retrieval.bge_retriever import BGERetriever

        return BGERetriever
    raise AttributeError(name)
