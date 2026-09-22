# 🌐 End-to-End Live Cloud Deployment Guide: DR. MIGI
> **How DR. MIGI was deployed to a 100% free, 24/7 live public URL using Streamlit Community Cloud and a Private Hugging Face Model Hub.**

Live Web Application: **[https://dr-migi.streamlit.app](https://dr-migi.streamlit.app)**  
Private Model Repository: **[https://huggingface.co/suvidh21/dr-migi](https://huggingface.co/suvidh21/dr-migi)** (🔒 Private)

---

## 📑 Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [The Core Problem: Free Cloud Memory Limits](#2-the-core-problem-free-cloud-memory-limits)
3. [Step-by-Step Deployment Blueprint](#3-step-by-step-deployment-blueprint)
   - [Phase A: Model Merging & Preparation](#phase-a-model-merging--preparation)
   - [Phase B: Private Hugging Face Repository & Token Setup](#phase-b-private-hugging-face-repository--token-setup)
   - [Phase C: Automated Weight Upload](#phase-c-automated-weight-upload)
   - [Phase D: Dual-Mode Backend Engine](#phase-d-dual-mode-backend-engine)
   - [Phase E: Streamlit Community Cloud Deployment](#phase-e-streamlit-community-cloud-deployment)
   - [Phase F: Secure Secrets Configuration](#phase-f-secure-secrets-configuration)
4. [Intellectual Property & Security Protections](#4-intellectual-property--security-protections)
5. [Operational Notes & Troubleshooting](#5-operational-notes--troubleshooting)

---

## 1. Architecture Overview

DR. MIGI is hosted using a **decoupled, serverless cloud architecture** designed to operate at **$0.00 cost** while maintaining high availability, instantaneous responsiveness, and zero memory crashes on mobile devices.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      END USER                          │
                    │         (Mobile Phone / Laptop / Tablet)               │
                    └──────────────────────────┬─────────────────────────────┘
                                               │ HTTPS
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAMLIT COMMUNITY CLOUD (https://dr-migi.streamlit.app)                                │
│                                                                                          │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌───────────────────────────┐  │
│  │     High-Contrast     │   │   ChromaDB Vector DB   │   │  Dual-Mode Engine         │  │
│  │    Clinical Web UI    ├──►│ (Patient Records/RAG)  ├──►│ (backend/inference/engine)│  │
│  │   (frontend/app.py)   │   │ (outputs/chroma_db)    │   │                           │  │
│  └───────────────────────┘   └────────────────────────┘   └─────────────┬─────────────┘  │
│                                  RAM Consumption: ~45 MB                │                │
└─────────────────────────────────────────────────────────────────────────┼────────────────┘
                                                                          │ Authenticated
                                                                          │ HTTPS REST API
                                                                          ▼ (HF_TOKEN)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ HUGGING FACE CLOUD (🔒 Private Model Hub)                                                │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Repository: suvidh21/dr-migi                                                       │  │
│  │ - model.safetensors (988 MB fine-tuned weights)                                    │  │
│  │ - tokenizer.json, tokenizer_config.json, chat_template.jinja                        │  │
│  │ - config.json, generation_config.json                                              │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│  Serverless Compute Infrastructure executes model reasoning & streams tokens back         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Core Problem: Free Cloud Memory Limits

Deploying deep learning language models on free hosting platforms presents a major bottleneck:

| Environment | RAM Available | Memory Footprint of PyTorch + Model | Outcome |
|---|---|---|---|
| **Local PC (RTX GPU)** | 16 GB+ VRAM / RAM | ~1.7 GB (Float32) / 950 MB (FP16) | ✅ Runs smoothly at full speed |
| **Streamlit Cloud (Standard)** | **1.0 GB Cap** | **1.7 GB+ (PyTorch + Weights)** | ❌ **CRASH:** *"App has gone over its resource limits"* |

When users opened the link on mobile devices, Streamlit Cloud attempted to load the full model weights into the container's 1 GB RAM, immediately triggering an Out-Of-Memory (OOM) shutdown.

### The Solution: Decoupled Cloud Architecture
Instead of loading the heavy 1 GB neural network weights inside Streamlit's container:
1. **Frontend & Vector DB** run on Streamlit Cloud (consuming only **~45 MB of RAM**).
2. **Model Weights & Inference** live inside a **Private Hugging Face repository**, queried securely via encrypted tokens.
3. **RAM usage stays below 5% of the cloud limit**, guaranteeing 100% stability across all smartphones and desktop browsers.

---

## 3. Step-by-Step Deployment Blueprint

### Phase A: Model Merging & Preparation
DR. MIGI was trained using **QLoRA (Low-Rank Adaptation)** on top of `Qwen/Qwen2.5-0.5B-Instruct`. Before deployment, the LoRA adapter weights were merged back into the base model weights to produce a standalone model:

```bash
python scripts/training/merge_migi_model.py
```

This produced a complete, production-ready directory at `models/MIGI-Qwen2.5-0.5B-v1/` containing:
- `model.safetensors` (988 MB)
- `config.json` & `generation_config.json`
- `tokenizer.json` & `tokenizer_config.json`
- `chat_template.jinja`

---

### Phase B: Private Hugging Face Repository & Token Setup

1. **Token Generation**:
   - Navigate to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Create an Access Token named `dr-migi-upload` with **Write** permissions (specifically `repo.write` and `repo.content.read`).
2. **Private Repository Creation**:
   A private model repository was created via Python using the `huggingface_hub` SDK:

```python
from huggingface_hub import HfApi

api = HfApi(token="your_hf_write_token")
api.create_repo(
    repo_id="suvidh21/dr-migi",
    repo_type="model",
    private=True,
    exist_ok=True
)
```

---

### Phase C: Automated Weight Upload

The local merged model directory was pushed to Hugging Face with an atomic commit using `api.upload_folder`:

```python
from huggingface_hub import HfApi

api = HfApi(token="your_hf_write_token")
api.upload_folder(
    folder_path="models/MIGI-Qwen2.5-0.5B-v1",
    repo_id="suvidh21/dr-migi",
    repo_type="model",
    commit_message="Initial upload: DR. MIGI fine-tuned clinical model weights"
)
```

All 6 model files plus an auto-generated model card `README.md` were committed securely to `suvidh21/dr-migi`.

---

### Phase D: Dual-Mode Backend Engine

The backend engine (`backend/inference/engine.py` and `backend/model/loader.py`) was engineered with smart detection:

1. **Cloud Mode (Streamlit Cloud)**:
   - Detects `HF_TOKEN` from `st.secrets` or environment variables.
   - Routes inference queries directly over HTTPS to Hugging Face GPU compute.
   - Zero local model weights are loaded into RAM.
2. **Local Mode (Developer Workstation)**:
   - When running offline or on your PC without a token, it detects `models/MIGI-Qwen2.5-0.5B-v1` on your local drive.
   - Loads directly into CUDA / local CPU via PyTorch in ~1.4 seconds with zero network latency.

```python
# Discovers token seamlessly across local .env and Streamlit Cloud
def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return None
```

---

### Phase E: Streamlit Community Cloud Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **"New app"**.
3. Fill in the repository details:
   - **Repository**: `Suvidh21/MIGI`
   - **Branch**: `main`
   - **Main file path**: `app.py` (which shims directly into `frontend/app.py`)
4. Click **Deploy!**

---

### Phase F: Secure Secrets Configuration

Because the Hugging Face repository `suvidh21/dr-migi` is **100% Private**, Streamlit Cloud must authenticate itself to read and query the model:

1. On your Streamlit Cloud app dashboard, click **App settings** → **Secrets**.
2. Paste your Hugging Face token in TOML format:

```toml
HF_TOKEN = "hf_your_token_here"
```

3. Click **Save changes**. Streamlit Cloud will securely inject this token at runtime without exposing it in git or public repositories.

---

## 4. Intellectual Property & Security Protections

A critical requirement of this deployment was preventing anyone from copying or scraping DR. MIGI's custom fine-tuned weights:

- **Private Model Repository**: The Hugging Face repo `suvidh21/dr-migi` has its visibility set to **Private**. Public users browsing Hugging Face receive a 404 error if they attempt to navigate to the URL.
- **Zero Weights in Git**: Git tracks only Python application code, configs, and patient datasets. Heavy model weights are excluded via `.gitignore` (`models/`, `*.safetensors`, `*.pt`).
- **Encrypted Credentials**: `secrets.toml` is added to `.gitignore`. The live credentials exist only inside Streamlit Cloud's encrypted secret store and are never committed to version control.

---

## 5. Operational Notes & Troubleshooting

### Automatic Sleep & Wakeup
Streamlit Community Cloud automatically puts apps to sleep after 48+ hours of inactivity to conserve global server capacity. If a user visits a sleeping app:
- They see a friendly screen: *"This app has gone to sleep due to inactivity"*.
- Clicking **"Yes, get this app back up!"** wakes the container in **10–15 seconds**.

### Root Entrypoint Shim (`app.py`)
Streamlit Community Cloud expects an entrypoint in the repository root. To preserve a clean folder structure (`frontend/`, `backend/`, `scripts/`, `docs/`), the root `app.py` acts as an executive forwarding shim:

```python
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_app_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
with open(_app_path, "r", encoding="utf-8") as _f:
    exec(_f.read(), {"__name__": "__main__", "__file__": _app_path})
```

---

*Authored for the DR. MIGI Project by Mathur (suvidh21).*
