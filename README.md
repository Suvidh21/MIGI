
# DR. MIGI — AI Healthcare Companion

DR. MIGI is an intelligent AI healthcare companion designed to assist doctors with **longitudinal patient analysis and clinical decision support**. This repository is built in generations — each one adding a new layer of capability.

> **Current Status: Generation 1 (Clinical Brain) ✅ + Generation 2 (Memory/RAG) ✅**

---
"DR.MIGI is an ongoing research and engineering project focused on building an AI healthcare assistant while exploring Retrieval-Augmented Generation (RAG), local LLM inference, memory systems, prompt engineering, and modern AI architectures." Rather than being a finished product, this repository documents the evolution of the project—from foundational concepts and experiments to production-oriented implementations.

This repository includes my personal engineering notes covering transformers, tokenization, embeddings, vector databases, attention mechanisms, prompt engineering, RAG pipelines, and LLM inference. They serve as documentation for the concepts explored while building DR.MIGI.

## 🧠 What is DR. MIGI?

DR. MIGI is not a chatbot. It is not designed for general users. It is a **clinical intelligence system** built for doctors — designed to do what no human can do at scale: read, analyse, and find patterns across years of patient records, and surface what matters before it becomes a crisis.

Most AI systems are built to agree with you — to generate the most probable, human-sounding response. **DR. MIGI is built to do the opposite**: to stay grounded in clinical evidence and flag what the data actually shows, not what sounds reassuring.

See [`notes/who_is_dr_migi.md`](notes/who_is_dr_migi.md) for the full long-term vision.

---

## 🏗️ Generational Architecture

```
Generation 1 — Clinical Brain        ✅ BUILT
    Qwen 2.5 local inference engine
    Clinical system prompt & constraints
    Synchronous + streaming generation
    ↓

Generation 2 — Memory (RAG)          ✅ BUILT
    Patient record ingestion pipeline
    BAAI/bge-small-en-v1.5 embeddings
    ChromaDB vector database
    Semantic retrieval + grounded generation
    ↓

Generation 3 — Vision                🔜 Planned
    YOLO-based ICU camera monitoring
    ↓

Generation 4 — Medical Device Integration   🔜 Planned
    ECG, BP, EEG, Doppler live streams
    ↓

Generation 5 — Multimodal Fusion     🔜 Planned
Generation 6 — Predictive Intelligence    🔜 Planned
Generation 7 — Autonomous Hospital AI    🔜 Planned
```

---

## ⚡ Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Ingest patient records into ChromaDB
```bash
cd d:\Suvidh\Suvidh\DR.MIGI
python scripts/ingest_patients.py
```

### Step 2 — Run the RAG demo
```bash
python notebooks/03_rag_demo.py
```

### Step 3 — Use Phase 1 inference directly (no RAG)
```bash
python notebooks/01_cpu_inference.py
```

---

## 🛠️ Codebase Structure

### Phase 1 — Clinical Brain

| File | Description |
|---|---|
| [`configs/inference_config.json`](configs/inference_config.json) | LLM model config and default generation hyperparameters |
| [`src/model/loader.py`](src/model/loader.py) | HuggingFace environment routing, device detection, model loading |
| [`src/inference/engine.py`](src/inference/engine.py) | `DrMigiEngine` — synchronous and streaming LLM inference |
| [`src/prompts/templates.py`](src/prompts/templates.py) | Clinical system prompt + RAG-grounded prompt templates |
| [`src/utils/logger.py`](src/utils/logger.py) | RAM/VRAM monitoring and execution timing |

### Phase 2 — RAG Pipeline

| File | Description |
|---|---|
| [`configs/rag_config.json`](configs/rag_config.json) | RAG hyperparameters (embedding model, ChromaDB path, top-k) |
| [`src/rag/chunker.py`](src/rag/chunker.py) | Splits patient JSON records into visit-level text chunks |
| [`src/rag/embedder.py`](src/rag/embedder.py) | `BAAI/bge-small-en-v1.5` embedding model wrapper |
| [`src/rag/vector_store.py`](src/rag/vector_store.py) | ChromaDB wrapper — store, upsert, and semantic search |
| [`src/rag/retriever.py`](src/rag/retriever.py) | Query → top-k relevant records → formatted context string |
| [`src/rag/pipeline.py`](src/rag/pipeline.py) | `DrMigiRAGPipeline` — full orchestration (retrieval + LLM) |
| [`scripts/ingest_patients.py`](scripts/ingest_patients.py) | One-time script to load patient data into ChromaDB |
| [`datasets/patients/`](datasets/patients/) | Synthetic patient records (JSON format) |

### Notebooks & Tests

| File | Description |
|---|---|
| [`notebooks/01_cpu_inference.py`](notebooks/01_cpu_inference.py) | Phase 1 — direct LLM inference walkthrough |
| [`notebooks/02_explain_tokens.py`](notebooks/02_explain_tokens.py) | Tokenization deep-dive — subword splits and token IDs |
| [`notebooks/03_rag_demo.py`](notebooks/03_rag_demo.py) | Phase 2 — end-to-end RAG demo with 3 patients |
| [`tests/test_inference.py`](tests/test_inference.py) | Automated test suite for the inference engine |

---

## 🧬 RAG Architecture

```
Patient Records (JSON)
       ↓
PatientRecordChunker
  → Splits each visit into a semantically coherent text chunk
       ↓
MigiEmbedder (BAAI/bge-small-en-v1.5)
  → Converts each chunk to a 384-dimensional vector
       ↓
ChromaDB (local, on-disk)
  → Stores vectors + original text + metadata (patient_id, date)
       ↓
──────────────── QUERY TIME ────────────────
Doctor asks: "Any concerning trends for Patient P001?"
       ↓
MigiEmbedder embeds the query → same 384-dim vector space
       ↓
ChromaDB semantic similarity search
  → Returns top-5 most relevant visit records
       ↓
MigiRetriever formats retrieved records into context string
       ↓
DrMigiRAGPipeline injects context into clinical system prompt
       ↓
Qwen 2.5 (DrMigiEngine) generates grounded clinical response ✅
```

**Key principle:** The LLM is not guessing from general medical knowledge. It is reasoning over actual, retrieved patient data — making responses patient-specific and clinically grounded.

---

## 🎓 LLM Theory Lectures

The theoretical foundation of DR. MIGI is documented in [`notes/`](notes/):

1. **[Lecture 1: LLM Architecture](notes/lecture_01_architecture.md)** — Decoder-only transformers, autoregressive generation
2. **[Lecture 2: Tokenization](notes/lecture_02_tokenization.md)** — BPE, 151,936-token vocabulary, subword splits
3. **[Lecture 3: Embeddings & RoPE](notes/lecture_03_embeddings.md)** — Semantic vectors, rotary position encoding
4. **[Lecture 4: Transformer Blocks](notes/lecture_04_transformer_blocks.md)** — RMSNorm, SwiGLU FFN
5. **[Lecture 5: Attention (GQA)](notes/lecture_05_attention.md)** — Q/K/V, scaled dot-product, grouped-query attention
6. **[Lecture 6: Inference & Decoding](notes/lecture_06_inference.md)** — KV cache, prefill vs decode, temperature sampling

---

## 📦 Stack

| Component | Tool | Notes |
|---|---|---|
| Foundation LLM | `Qwen2.5-0.5B-Instruct` | Runs locally on CPU/GPU — no API required |
| LLM Framework | `HuggingFace Transformers` + `PyTorch` | Standard inference stack |
| Embedding Model | `BAAI/bge-small-en-v1.5` | 130MB, local, 384-dim vectors |
| Vector Database | `ChromaDB` | On-disk, zero config, Apache 2.0 |
| Language | Python 3.10+ | |

**No external APIs. No patient data sent to any server. Fully local.**

---

## ⚕️ Clinical Disclaimer

DR. MIGI is a **reasoning assistant and research tool**, not a licensed medical device. It must not be used as a substitute for qualified clinical judgment. All outputs are for informational and educational purposes only.
