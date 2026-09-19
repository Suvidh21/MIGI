"""
DR. MIGI — Patient Ingestion Script
=====================================
Run this script once to load all patient records into the ChromaDB vector database.

Usage:
    cd d:\\Suvidh\\Suvidh\\DR.MIGI
    python scripts/ingest_patients.py

What it does:
    1. Scans datasets/patients/ for all .json patient record files
    2. Chunks each patient record by visit (PatientRecordChunker)
    3. Embeds all chunks using BAAI/bge-small-en-v1.5
    4. Stores all embeddings in ChromaDB at outputs/chroma_db/

Run this once before using the RAG pipeline or the demo notebook.
Re-running is safe — ChromaDB upserts will update existing records without duplication.
"""

import os
import sys

# Add project root to path so src.* imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.chunker import PatientRecordChunker
from src.rag.embedder import MigiEmbedder
from src.rag.vector_store import MigiVectorStore

PATIENTS_DIR = "datasets/patients"
CONFIG_PATH  = "configs/rag_config.json"


def main():
    print("=" * 60)
    print("DR. MIGI — Patient Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Discover all patient JSON files
    patient_files = [
        os.path.join(PATIENTS_DIR, f)
        for f in os.listdir(PATIENTS_DIR)
        if f.endswith(".json")
    ]

    if not patient_files:
        print(f"No patient JSON files found in {PATIENTS_DIR}/")
        sys.exit(1)

    print(f"\nFound {len(patient_files)} patient file(s):")
    for pf in patient_files:
        print(f"  * {pf}")

    # Step 2: Load embedding model
    print("\nLoading embedding model...")
    embedder = MigiEmbedder(config_path=CONFIG_PATH)

    # Step 3: Connect to vector store
    print("\nConnecting to ChromaDB vector store...")
    vector_store = MigiVectorStore(embedder=embedder, config_path=CONFIG_PATH)

    # Step 4: Chunk and ingest each patient
    chunker = PatientRecordChunker()
    total_chunks = 0

    for filepath in sorted(patient_files):
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        try:
            chunks = chunker.chunk_from_file(filepath)
            print(f"  -> {len(chunks)} visit chunks generated")

            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)

            for chunk in chunks:
                print(f"     Stored: {chunk['chunk_id']} | {chunk['metadata']['date']}")

        except Exception as e:
            print(f"  [X] Error processing {filename}: {e}")
            raise

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Ingestion complete.")
    print(f"   Patients processed : {len(patient_files)}")
    print(f"   Total chunks stored: {total_chunks}")
    print(f"   Total in ChromaDB  : {vector_store.collection_count()}")
    print("=" * 60)
    print("\nYou can now run the RAG pipeline or the demo notebook.")


if __name__ == "__main__":
    main()
