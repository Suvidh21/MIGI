import os
import json
import chromadb
from chromadb.config import Settings
from src.rag.embedder import MigiEmbedder
from src.rag.chunker import PatientRecordChunker


class MigiVectorStore:
    """
    ChromaDB vector database wrapper for the DR. MIGI RAG pipeline.
    """

    def __init__(self, embedder: MigiEmbedder, config_path: str = "configs/rag_config.json"):
        """
        Initialises the ChromaDB client and retrieves or creates the patient collection.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        db_path = config.get("vector_db_path", "./outputs/chroma_db")
        collection_name = config.get("collection_name", "migi_patients")

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedder = embedder
        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            self._auto_ingest_default_patients()

        print(f"VectorStore ready. Collection: '{collection_name}' | Documents: {self.collection.count()}")

    def _auto_ingest_default_patients(self):
        """Auto-ingests patient JSON files from datasets/patients if collection is empty."""
        patients_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets",
            "patients"
        )
        if not os.path.exists(patients_dir):
            return

        patient_files = [
            os.path.join(patients_dir, f)
            for f in os.listdir(patients_dir)
            if f.endswith(".json")
        ]
        if not patient_files:
            return

        print(f"Empty VectorStore detected. Auto-ingesting {len(patient_files)} default patient profiles...")
        chunker = PatientRecordChunker()
        for filepath in patient_files:
            try:
                chunks = chunker.chunk_from_file(filepath)
                self.add_chunks(chunks)
            except Exception as e:
                print(f"Warning: Failed to auto-ingest {filepath}: {e}")

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Embeds and stores a list of patient visit chunks in ChromaDB.

        What it does:
            1. Extracts the text from each chunk
            2. Batch-embeds all texts using the MigiEmbedder
            3. Stores (chunk_id, embedding, text, metadata) in ChromaDB

        Arguments:
            chunks: List of chunk dicts from PatientRecordChunker.
                    Each must have 'chunk_id', 'text', and 'metadata' keys.
        """
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts)

        # ChromaDB upsert: inserts new records, updates existing ones with the same ID
        # This makes re-ingestion safe — running ingest_patients.py twice won't duplicate data
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Stored {len(chunks)} chunks. Total in collection: {self.collection.count()}")

    def search(self, query: str, top_k: int = 5, patient_id: str | None = None) -> list[dict]:
        """
        Performs semantic similarity search against the stored patient records.

        What it does:
            Embeds the query, then asks ChromaDB for the top_k most similar chunks.
            If patient_id is specified, search is filtered to that patient only.

        Why semantic search:
            Unlike keyword search (which requires exact word matches), semantic search
            finds conceptually related records. For example, querying "glucose control"
            will retrieve chunks mentioning "blood sugar", "HbA1c", and "Metformin"
            even if the word "glucose" doesn't appear in the chunk.

        Arguments:
            query: The clinical question or concern to search for.
            top_k: Maximum number of chunks to retrieve.
            patient_id: If provided, filters results to this patient only.

        Returns:
            List of result dicts, each containing 'text', 'metadata', and 'distance'.
        """
        query_embedding = self.embedder.embed_text(query)

        where_filter = {"patient_id": patient_id} if patient_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                formatted_results.append({
                    "text": text,
                    "metadata": metadata,
                    "distance": round(distance, 4)
                })

        return formatted_results

    def get_patient_chunks(self, patient_id: str) -> list[dict]:
        """
        Retrieves all stored chunks for a specific patient, ordered by visit date.

        Why it is needed:
            Useful for generating a full patient summary or timeline
            without a specific query — just retrieve everything for this patient.

        Arguments:
            patient_id: The patient identifier (e.g., 'P001').
        """
        results = self.collection.get(
            where={"patient_id": patient_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results["documents"]:
            for text, metadata in zip(results["documents"], results["metadatas"]):
                chunks.append({"text": text, "metadata": metadata})

        # Sort by visit date for chronological order
        chunks.sort(key=lambda x: x["metadata"].get("date", ""))
        return chunks

    def collection_count(self) -> int:
        """Returns the total number of chunks currently stored in the collection."""
        return self.collection.count()
