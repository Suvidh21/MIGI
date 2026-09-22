"""
DR. MIGI — Phase 2 RAG Demo
==============================
End-to-end demonstration of the RAG pipeline.

Prerequisites:
    1. Run: pip install sentence-transformers chromadb
    2. Run: python scripts/ingest_patients.py   (loads patient data into ChromaDB)
    3. Then run this notebook.

Usage:
    cd d:\\Suvidh\\Suvidh\\DR.MIGI
    python notebooks/03_rag_demo.py

What this demonstrates:
    - Semantic retrieval of patient-specific records from ChromaDB
    - Grounded LLM response using retrieved patient history
    - Comparison between Phase 1 (generic) and Phase 2 (grounded RAG) responses
    - Token-level metrics for the full RAG pipeline
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.chunker import PatientRecordChunker
from backend.rag.embedder import MigiEmbedder
from backend.rag.vector_store import MigiVectorStore
from backend.rag.retriever import MigiRetriever
from backend.rag.pipeline import DrMigiRAGPipeline

CONFIG_PATH = "configs/rag_config.json"
SEPARATOR = "\n" + "=" * 70 + "\n"


def demo_retrieval_only():
    """
    Demo Part 1: Show the retrieval layer working independently.
    Demonstrates what records ChromaDB finds for a given query.
    No LLM involved — pure vector search.
    """
    print(SEPARATOR)
    print("DEMO 1: Retrieval Layer — Semantic Search Without LLM")
    print(SEPARATOR)

    embedder = MigiEmbedder(config_path=CONFIG_PATH)
    vector_store = MigiVectorStore(embedder=embedder, config_path=CONFIG_PATH)
    retriever = MigiRetriever(vector_store=vector_store, config_path=CONFIG_PATH)

    print(f"Total records in ChromaDB: {vector_store.collection_count()}")

    # Query 1: Trend detection for a specific patient
    query = "Is there a concerning trend in blood sugar and blood pressure?"
    patient_id = "P001"

    print(f"\nQuery    : '{query}'")
    print(f"Patient  : {patient_id}")
    print(f"\nRetrieved context:\n")

    context = retriever.retrieve(query=query, patient_id=patient_id)
    print(context)

    # Query 2: Cardiac risk across all patients
    print(SEPARATOR)
    query2 = "cardiac risk, chest pain, ECG changes"
    print(f"Query (all patients): '{query2}'")
    print(f"\nRetrieved context:\n")
    context2 = retriever.retrieve(query=query2, patient_id=None)
    print(context2)

    return embedder, vector_store, retriever


def demo_full_rag_pipeline(embedder, vector_store, retriever):
    """
    Demo Part 2: Full RAG pipeline — retrieval + LLM generation.
    Asks clinical questions and shows grounded responses.
    """
    print(SEPARATOR)
    print("DEMO 2: Full RAG Pipeline — Grounded Clinical Analysis")
    print(SEPARATOR)
    print("Loading LLM inference engine (Qwen 2.5)...")
    print("Note: This takes ~30 seconds on CPU. Patient data is already in ChromaDB.\n")

    pipeline = DrMigiRAGPipeline(config_path=CONFIG_PATH)

    # -----------------------------------------------------------------------
    # Question 1: Trend analysis for P001 (pre-diabetic → T2DM patient)
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print("QUESTION 1 — Patient P001: John Doe")
    question1 = "What are the most concerning trends in this patient's history over the last 4 years?"

    print(f"Q: {question1}\n")
    print("Retrieving patient records and generating response...\n")

    result1 = pipeline.ask(
        patient_id="P001",
        question=question1,
        max_new_tokens=300,
        temperature=0.3   # Lower temperature for clinical precision
    )

    print(f"DR. MIGI Response:\n{result1['response']}")
    print(f"\n[Metrics] Tokens: {result1['output_tokens_count']} | "
          f"Speed: {result1['tokens_per_second']:.1f} tok/s | "
          f"Device: {result1['device'].upper()}")

    # -----------------------------------------------------------------------
    # Question 2: Risk assessment for P002 (cardiac risk patient)
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print("QUESTION 2 — Patient P002: Priya Sharma")
    question2 = "Based on the available records, were there early warning signs that preceded the cardiac event?"

    print(f"Q: {question2}\n")
    print("Retrieving patient records and generating response...\n")

    result2 = pipeline.ask(
        patient_id="P002",
        question=question2,
        max_new_tokens=300,
        temperature=0.3
    )

    print(f"DR. MIGI Response:\n{result2['response']}")
    print(f"\n[Metrics] Tokens: {result2['output_tokens_count']} | "
          f"Speed: {result2['tokens_per_second']:.1f} tok/s | "
          f"Device: {result2['device'].upper()}")

    # -----------------------------------------------------------------------
    # Question 3: Full timeline summary for P003 (COPD patient)
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print("QUESTION 3 — Patient P003: Ramesh Iyer (Full Timeline Mode)")
    question3 = "Summarise the key clinical deterioration pattern for this patient across all visits."

    print(f"Q: {question3}\n")
    print("Retrieving full patient timeline...\n")

    result3 = pipeline.ask(
        patient_id="P003",
        question=question3,
        use_full_timeline=True,   # Retrieve all visits, not just top-k
        max_new_tokens=350,
        temperature=0.3
    )

    print(f"DR. MIGI Response:\n{result3['response']}")
    print(f"\n[Metrics] Tokens: {result3['output_tokens_count']} | "
          f"Speed: {result3['tokens_per_second']:.1f} tok/s | "
          f"Device: {result3['device'].upper()}")

    print(SEPARATOR)
    print("[SUCCESS] RAG Demo Complete.")
    print("DR. MIGI is reasoning over real (mock) patient data — not just pretrained knowledge.")
    print(SEPARATOR)


def demo_streaming(embedder, vector_store, retriever):
    """
    Demo Part 3: Streaming response — shows tokens arriving in real time.
    Demonstrates the interactive feel of the RAG pipeline on CPU.
    """
    print(SEPARATOR)
    print("DEMO 3: Streaming Response — Real-Time Token Generation")
    print(SEPARATOR)

    from backend.rag.pipeline import DrMigiRAGPipeline
    pipeline = DrMigiRAGPipeline(config_path=CONFIG_PATH)

    question = "Is this patient at risk of developing complications in the near future?"
    patient_id = "P001"

    print(f"Patient: {patient_id}")
    print(f"Q: {question}\n")
    print("DR. MIGI (streaming):\n")

    for token in pipeline.ask_stream(
        patient_id=patient_id,
        question=question,
        max_new_tokens=200,
        temperature=0.3
    ):
        print(token, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    # Run all three demos in sequence
    embedder, vector_store, retriever = demo_retrieval_only()
    demo_full_rag_pipeline(embedder, vector_store, retriever)
    # demo_streaming is optional — uncomment to run
    # demo_streaming(embedder, vector_store, retriever)
