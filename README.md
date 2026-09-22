
# DR. MIGI — AI Healthcare Companion

<p align="center">
  <b>🌐 Live Web App:</b> <a href="https://dr-migi.streamlit.app"><b>https://dr-migi.streamlit.app</b></a>
</p>

<p align="center">
  <img src="images/Screenshot%202026-09-19%20153636.png" width="850">
</p>

DR. MIGI is an intelligent AI healthcare companion designed to assist doctors with **longitudinal patient analysis and clinical decision support**. This repository is built in generations — each one adding a new layer of capability.

> **Current Status: Generation 1 (Clinical Brain) ✅ + Generation 2 (Memory/RAG) ✅ + Fine-Tuned Model ✅ + Live Cloud Deployment ✅**
> 
> Full technical walkthrough of the free cloud deployment architecture: **[`docs/LIVE_DEPLOYMENT_GUIDE.md`](docs/LIVE_DEPLOYMENT_GUIDE.md)**.
<p align="center">
  <img src="images/Screenshot%202026-09-19%20150511.png" width="850">
</p>

<p align="center">
  <img src="images/Screenshot%202026-09-19%20153306.png" width="850">
</p>

---
"DR.MIGI is an ongoing research and engineering project focused on building an AI healthcare assistant while exploring Retrieval-Augmented Generation (RAG), local LLM inference, memory systems, prompt engineering, and modern AI architectures." Rather than being a finished product, this repository documents the evolution of the project—from foundational concepts and experiments to production-oriented implementations.

This repository includes my personal engineering notes covering transformers, tokenization, embeddings, vector databases, attention mechanisms, prompt engineering, RAG pipelines, and LLM inference. They serve as documentation for the concepts explored while building DR.MIGI.

## 🧠 What is DR. MIGI?

DR. MIGI is not a chatbot. It is not designed for general users. It is a **clinical intelligence system** built for doctors — designed to do what no human can do at scale: read, analyse, and find patterns across years of patient records, and surface what matters before it becomes a crisis.

Most AI systems are built to agree with you — to generate the most probable, human-sounding response. **DR. MIGI is built to do the opposite**: to stay grounded in clinical evidence and flag what the data actually shows, not what sounds reassuring.

See [`notes/who_is_dr_migi.md`](notes/who_is_dr_migi.md) for the full long-term vision.

---

## 🚀 Live Cloud Deployment

DR. MIGI is deployed as a **free, 24/7 live web application** — anyone can open it on phone or PC without downloading anything.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│              PRIVATE HUGGING FACE HUB                    │
│   Repository: suvidh21/dr-migi (100% Private)            │
│   Contents: model.safetensors (988 MB merged weights)    │
│             tokenizer, config, chat_template              │
└──────────────────────┬───────────────────────────────────┘
                       │ Authenticated download via HF_TOKEN
                       ▼
┌──────────────────────────────────────────────────────────┐
│              STREAMLIT COMMUNITY CLOUD                   │
│   Container: 1 GB RAM (free tier)                        │
│   Python 3.11 + PyTorch + Transformers                   │
│   Model loaded in float16 (~490 MB)                      │
│   Inference: ~4 tok/s on CPU                             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│   🌐 https://dr-migi.streamlit.app                       │
│   Live 24/7 — $0.00 cost — No downloads required         │
│   Running OUR fine-tuned DR. MIGI model, not base Qwen   │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Private HF repo** (not GitHub) | GitHub has 100 MB file limit; model is 988 MB |
| **float16 precision** on CPU | Halves RAM from ~1.9 GB (float32) to ~490 MB to fit 1 GB container |
| **`low_cpu_mem_usage=True`** | Prevents PyTorch from doubling memory during model init |
| **No HF Serverless API** | HF free serverless only supports public foundation models, not custom fine-tuned models |
| **Our fine-tuned weights** | Live app runs `suvidh21/dr-migi` — our QLoRA fine-tuned + merged Qwen 2.5 0.5B, not the base model |

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
python scripts/data/ingest_patients.py
```

### Step 2 — Run the Web App (Streamlit)
```bash
streamlit run app.py
```

### Step 3 — Run the RAG demo
```bash
python notebooks/03_rag_demo.py
```

### Step 4 — Use Phase 1 inference directly (no RAG)
```bash
python notebooks/01_cpu_inference.py
```

---

## 🛠️ Codebase Structure

### 🖥️ Frontend & Entrypoints

| File | Description |
|---|---|
| [`app.py`](app.py) | Root entrypoint shim for Streamlit Community Cloud and local execution |
| [`frontend/app.py`](frontend/app.py) | Streamlit web interface (high-contrast clinical chat, telemetry, and RAG viewer) |

### 🧠 Backend Core (`backend/`)

| File | Description |
|---|---|
| [`configs/inference_config.json`](configs/inference_config.json) | Model config — points to `suvidh21/dr-migi` (our fine-tuned model) |
| [`configs/rag_config.json`](configs/rag_config.json) | RAG hyperparameters (embedding model, ChromaDB path, top-k) |
| [`backend/model/loader.py`](backend/model/loader.py) | Memory-optimized model loader — float16 on CPU, auto-detects local vs cloud |
| [`backend/inference/engine.py`](backend/inference/engine.py) | `DrMigiEngine` — synchronous and streaming LLM inference (runs our fine-tuned model directly) |
| [`backend/prompts/templates.py`](backend/prompts/templates.py) | Clinical system prompt + RAG-grounded prompt templates |
| [`backend/utils/logger.py`](backend/utils/logger.py) | RAM/VRAM monitoring and execution timing |
| [`backend/rag/chunker.py`](backend/rag/chunker.py) | Splits patient JSON records into visit-level text chunks |
| [`backend/rag/embedder.py`](backend/rag/embedder.py) | `BAAI/bge-small-en-v1.5` embedding model wrapper |
| [`backend/rag/vector_store.py`](backend/rag/vector_store.py) | ChromaDB wrapper — store, upsert, and semantic search |
| [`backend/rag/pipeline.py`](backend/rag/pipeline.py) | `DrMigiRAGPipeline` — full orchestration (retrieval + LLM) |

### 🎓 Training Pipeline (`scripts/training/`)

| File | Description |
|---|---|
| [`scripts/training/train_migi.py`](scripts/training/train_migi.py) | QLoRA fine-tuning on 40K medical QA pairs (4-bit, rank 64) |
| [`scripts/training/compare_migi.py`](scripts/training/compare_migi.py) | Base vs fine-tuned comparative evaluation |
| [`scripts/training/merge_migi_model.py`](scripts/training/merge_migi_model.py) | Merges LoRA adapters into base weights → `models/MIGI-Qwen2.5-0.5B-v1/` |

### 📂 Data Scripts (`scripts/data/`)

| File | Description |
|---|---|
| [`scripts/data/ingest_patients.py`](scripts/data/ingest_patients.py) | Loads patient synthetic records into ChromaDB |
| [`scripts/data/prepare_migi_dataset.py`](scripts/data/prepare_migi_dataset.py) | Formats MedMCQA records for fine-tuning |
| [`scripts/data/clean_migi_dataset.py`](scripts/data/clean_migi_dataset.py) | Dataset validation and cleaning |
| [`scripts/data/download_medmcqa.py`](scripts/data/download_medmcqa.py) | Downloads MedMCQA dataset |
| [`scripts/data/filter_medmcqa.py`](scripts/data/filter_medmcqa.py) | Filters medical QA records |

### 📓 Notebooks, Docs & Tests

| File | Description |
|---|---|
| [`notebooks/01_cpu_inference.py`](notebooks/01_cpu_inference.py) | Phase 1 — direct LLM inference walkthrough |
| [`notebooks/02_explain_tokens.py`](notebooks/02_explain_tokens.py) | Tokenization deep-dive — subword splits and token IDs |
| [`notebooks/03_rag_demo.py`](notebooks/03_rag_demo.py) | Phase 2 — end-to-end RAG demo with 3 patients |
| [`docs/notes/`](docs/notes/) | Architecture notes and lecture references |
| [`docs/LIVE_DEPLOYMENT_GUIDE.md`](docs/LIVE_DEPLOYMENT_GUIDE.md) | Step-by-step guide to deploying DR. MIGI live |
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
DR. MIGI (fine-tuned Qwen 2.5) generates grounded clinical response ✅
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
| Fine-Tuned LLM | `suvidh21/dr-migi` (Qwen 2.5 0.5B + QLoRA) | Our custom fine-tuned model on 40K medical QA pairs |
| LLM Framework | `HuggingFace Transformers` + `PyTorch` | Standard inference stack |
| Embedding Model | `BAAI/bge-small-en-v1.5` | 130MB, local, 384-dim vectors |
| Vector Database | `ChromaDB` | On-disk, zero config, Apache 2.0 |
| Model Hosting | `Hugging Face Hub` (Private) | Secure private vault for 988 MB model weights |
| Web App | `Streamlit Community Cloud` | Free 24/7 hosting, auto-deploys from GitHub |
| Language | Python 3.11 | |

**Model weights hosted privately on Hugging Face. No patient data sent to any server. Inference runs in-process.**

---

## ⚕️ Clinical Disclaimer

DR. MIGI is a **reasoning assistant and research tool**, not a licensed medical device. It must not be used as a substitute for qualified clinical judgment. All outputs are for informational and educational purposes only.
