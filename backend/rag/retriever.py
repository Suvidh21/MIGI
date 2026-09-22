from __future__ import annotations  # Python 3.9 compatibility

import json
from backend.rag.vector_store import MigiVectorStore


class MigiRetriever:
    """
    The retrieval interface for the DR. MIGI RAG pipeline.

    What it does:
        Given a clinical query and an optional patient ID, retrieves the most
        semantically relevant patient visit records from the vector store and
        assembles them into a formatted context string ready to be injected
        into the LLM prompt.

    Why it is needed:
        The vector store returns raw search results (text chunks + metadata).
        The retriever's job is to post-process those results into a coherent,
        structured context block that the LLM can reason over effectively.
        It acts as the bridge between raw retrieval and prompt construction.

    Best practices:
        Always filter by patient_id when the clinical question is about a
        specific patient — this prevents records from other patients being
        injected into the context, which would be a serious clinical error.
    """

    def __init__(self, vector_store: MigiVectorStore, config_path: str = "configs/rag_config.json"):
        """
        Arguments:
            vector_store: An initialised and populated MigiVectorStore.
            config_path: Path to rag_config.json.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        self.vector_store = vector_store
        self.top_k = config.get("top_k_retrieval", 5)

    def retrieve(self, query: str, patient_id: str | None = None) -> str:
        """
        Retrieves relevant patient history and formats it as a context string.

        What it does:
            1. Calls semantic similarity search on the vector store
            2. Sorts results by visit date (chronological order)
            3. Formats retrieved chunks into a numbered, labelled context block

        Arguments:
            query: The clinical question or concern.
            patient_id: If provided, restricts search to this patient only.

        Returns:
            A formatted string containing the retrieved patient context,
            ready to be injected directly into the LLM prompt.
        """
        results = self.vector_store.search(
            query=query,
            top_k=self.top_k,
            patient_id=patient_id
        )

        if not results:
            return "No relevant patient history found in the database."

        # Sort by date for chronological clinical narrative
        results.sort(key=lambda x: x["metadata"].get("date", ""))

        context_blocks = []
        for i, result in enumerate(results, start=1):
            date = result["metadata"].get("date", "Unknown date")
            visit_id = result["metadata"].get("visit_id", "")
            relevance = 1 - result["distance"]  # Convert distance to similarity score

            header = f"--- Retrieved Record {i} | {date} | {visit_id} | Relevance: {relevance:.2f} ---"
            context_blocks.append(f"{header}\n{result['text']}")

        return "\n\n".join(context_blocks)

    def retrieve_raw(self, query: str, patient_id: str | None = None) -> list[dict]:
        """
        Returns raw search results without formatting — for inspection or debugging.

        Arguments:
            query: The clinical question.
            patient_id: Optional patient filter.

        Returns:
            List of raw result dicts with 'text', 'metadata', and 'distance' keys.
        """
        return self.vector_store.search(query=query, top_k=self.top_k, patient_id=patient_id)

    def retrieve_full_timeline(self, patient_id: str) -> str:
        """
        Retrieves the complete chronological history for a patient — no query needed.

        Why it is needed:
            Some clinical questions are about overall trends rather than a specific concern.
            For example: 'Give me a full summary of this patient's history.'
            In this case, retrieving everything chronologically is more appropriate
            than semantic search against a specific query.

        Arguments:
            patient_id: The patient to retrieve a full timeline for.

        Returns:
            Formatted string of all chunks for this patient in visit order.
        """
        chunks = self.vector_store.get_patient_chunks(patient_id)

        if not chunks:
            return f"No records found for patient {patient_id}."

        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            date = chunk["metadata"].get("date", "Unknown date")
            header = f"--- Visit {i} | {date} ---"
            context_blocks.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(context_blocks)
