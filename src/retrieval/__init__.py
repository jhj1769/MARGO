"""Candidate retrievers shared by MARGO & baselines."""

from retrieval.base import BaseRetriever, RetrievalHit
from retrieval.bm25_retriever import BM25Retriever

__all__ = ["BaseRetriever", "RetrievalHit", "BM25Retriever", "BGERetriever"]


def __getattr__(name: str):  # pragma: no cover — lazy heavy import
    if name == "BGERetriever":
        from retrieval.bge_retriever import BGERetriever

        return BGERetriever
    raise AttributeError(name)
