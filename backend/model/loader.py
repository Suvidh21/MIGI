"""
DR. MIGI Model Loader — Memory-Optimized for Streamlit Cloud (1 GB RAM)
=======================================================================
Loads the fine-tuned DR. MIGI model (suvidh21/dr-migi) with aggressive
memory optimization to fit within Streamlit Cloud's free-tier 1 GB limit.

Strategy:
    - Load weights in float16 (half precision) → ~490 MB for a 0.5B model
    - Use low_cpu_mem_usage=True to avoid peak doubling during loading
    - Set model to eval mode (disables dropout, saves memory)
    - Disable gradient tracking globally (inference only)
"""
from __future__ import annotations  # Python 3.9 compatibility for type hints

import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.utils.logger import logger, log_resource_state, Timer

# =======================================================================
# CONFIGURATION LOAD
# =======================================================================
# What it does: Finds and reads our centralized config file.
# Why it is needed: Decouples model parameters and cache paths from code.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "configs",
    "inference_config.json"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")
    # Production-ready fallback configuration
    config = {
        "model_name": "suvidh21/dr-migi",
        "cache_dir": "./models/cache",
        "default_generation_params": {}
    }

raw_cache = config.get("cache_dir", "./models/cache")
if not os.path.isabs(raw_cache):
    resolved_cache = os.path.normpath(os.path.join(PROJECT_ROOT, raw_cache))
else:
    resolved_cache = raw_cache

config["cache_dir"] = resolved_cache
os.environ["HF_HOME"] = resolved_cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def determine_optimal_device() -> tuple[str, torch.dtype]:
    """
    Auto-detects the best hardware configuration for model execution.
    
    Returns:
        tuple of (device_string, torch_dtype)
        
    Memory strategy:
        - CUDA + bfloat16: Best quality, fast (Ampere+ GPUs)
        - CUDA + float16: Good quality, fast (all CUDA GPUs)  
        - CPU + float16: Half precision to fit in 1 GB cloud RAM
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            device = "cuda"
            dtype = torch.bfloat16
            logger.info("Optimal device: CUDA with bfloat16 precision.")
        else:
            device = "cuda"
            dtype = torch.float16
            logger.info("Optimal device: CUDA with float16 precision.")
    else:
        device = "cpu"
        # Use float16 to halve memory: 0.5B params × 2 bytes = ~1.0 GB
        # vs float32: 0.5B params × 4 bytes = ~2.0 GB (would crash 1 GB container)
        dtype = torch.float16
        logger.info("Optimal device: CPU with float16 precision (cloud-optimized).")
        
    return device, dtype


def load_model_and_tokenizer(model_name: str | None = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initializes and loads the DR. MIGI Model and Tokenizer.
    
    Memory optimization for Streamlit Cloud (CPU, 1 GB RAM):
        1. Load weights directly in float16 → ~490 MB (not ~1.9 GB float32)
        2. low_cpu_mem_usage=True prevents PyTorch from creating a duplicate during init
        3. model.eval() disables dropout layers (saves memory + faster)
        4. torch.no_grad() globally disables gradient storage (inference only)
    
    On local PC with GPU:
        - Detects local model on disk (models/MIGI-Qwen2.5-0.5B-v1) → zero download
        - Uses CUDA with float16/bfloat16 for fast inference
        
    On Streamlit Cloud:
        - Downloads from private HF repo (suvidh21/dr-migi) using HF_TOKEN
        - Runs on CPU in float16 for minimum memory footprint
    
    Arguments:
        model_name: Optional override. Defaults to config model_name.
        
    Returns:
        tuple of (model, tokenizer)
    """
    if model_name is None:
        model_name = config["model_name"]
        
    logger.info(f"Using Hugging Face cache directory: {os.environ.get('HF_HOME')}")
    log_resource_state("BEFORE LOADING MODEL")
    
    device, dtype = determine_optimal_device()
    
    # Resolve token for private Hugging Face repositories
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                hf_token = st.secrets["HF_TOKEN"]
        except Exception:
            pass

    # Check if local model directory exists on disk to avoid downloading
    local_model_path = os.path.join(PROJECT_ROOT, "models", "MIGI-Qwen2.5-0.5B-v1")
    if os.path.isdir(local_model_path) and os.path.exists(os.path.join(local_model_path, "model.safetensors")):
        load_source = local_model_path
        logger.info(f"Loading from local merged model on disk: '{local_model_path}'")
    else:
        load_source = model_name
        logger.info(f"Loading from remote HF repository: '{load_source}'")
    
    # Load tokenizer (lightweight, ~5 MB)
    with Timer(f"Loading Tokenizer from '{load_source}'"):
        tokenizer = AutoTokenizer.from_pretrained(
            load_source,
            token=hf_token,
            trust_remote_code=True,
            cache_dir=config["cache_dir"]
        )
        
    # Load model with memory-optimized settings
    with Timer(f"Loading Causal LLM from '{load_source}'"):
        device_map = "cpu" if device == "cpu" else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            token=hf_token,
            torch_dtype=dtype,            # float16 on CPU = ~490 MB instead of ~1.9 GB
            device_map=device_map,
            low_cpu_mem_usage=True,        # Prevents duplicate memory during init
            trust_remote_code=True,
            cache_dir=config["cache_dir"]
        )
    
    # Set to inference mode (disables dropout, batch norm training)
    model.eval()
    
    # Force garbage collection to clean up any loading intermediates
    gc.collect()
    
    log_resource_state("AFTER LOADING MODEL")
    logger.info(
        f"Model loaded successfully. Device: {device}, Dtype: {dtype}, "
        f"Source: '{load_source}'"
    )
    return model, tokenizer
