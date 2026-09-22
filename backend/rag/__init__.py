# =======================================================================
# DR. MIGI — Phase 2: RAG Pipeline
# =======================================================================
# This package contains the full Retrieval-Augmented Generation layer
# that sits between the user query and the LLM inference engine.
#
# Pipeline order:
#   Patient Records (JSON)
#       → Chunker      (splits visits into searchable units)
#       → Embedder     (converts text to vectors via bge-small-en-v1.5)
#       → VectorStore  (stores/searches vectors via ChromaDB)
#       → Retriever    (query → top-k relevant chunks)
#       → Pipeline     (retriever + DrMigiEngine = grounded response)

from .chunker import PatientRecordChunker
from .embedder import MigiEmbedder
from .vector_store import MigiVectorStore
from .retriever import MigiRetriever
from .pipeline import DrMigiRAGPipeline

__all__ = [
    "PatientRecordChunker",
    "MigiEmbedder",
    "MigiVectorStore",
    "MigiRetriever",
    "DrMigiRAGPipeline",
]
