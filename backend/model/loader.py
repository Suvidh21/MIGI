import os
import json
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
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
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
    Auto-detects the host system's hardware configuration and determines
    the optimal computation device and datatype.
    
    Why it is needed:
        LLM execution speed varies drastically based on hardware. Running in float32
        on GPU can exceed VRAM, and running in float16 on older CPUs is slow and lacks
        emulation support.
        
    How it works:
        Checks `torch.cuda.is_available()`.
        - If CUDA is present: chooses 'cuda' and 'float16' / 'bfloat16' depending on GPU capabilities.
        - If not: defaults to 'cpu' and 'float32' (most efficient representation for modern x86 CPU FPUs).
        
    Performance implications:
        Selecting CUDA enables matrix math acceleration. float32 on CPU prevents numerical precision underflow
        warnings while using CPU tensor execution.
    """
    if torch.cuda.is_available():
        # Check if GPU supports bfloat16 (Ampere architecture and newer e.g. RTX 30/40 series)
        if torch.cuda.is_bf16_supported():
            device = "cuda"
            dtype = torch.bfloat16
            logger.info("Optimal device found: CUDA with bfloat16 precision.")
        else:
            device = "cuda"
            dtype = torch.float16
            logger.info("Optimal device found: CUDA with float16 precision.")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("Optimal device found: CPU with float32 precision.")
        
    return device, dtype

def load_model_and_tokenizer(model_name: str | None = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initializes and loads the Model and Tokenizer instances from local cache or Hugging Face.
    
    Arguments:
        model_name: Optional override path or repository ID. Defaults to configs/inference_config.json model_name.
        
    Returns:
        tuple of (AutoModelForCausalLM, AutoTokenizer)
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
        logger.info(f"Loading from repository: '{load_source}'")
    
    # Load tokenizer
    with Timer(f"Loading Tokenizer from '{load_source}'"):
        tokenizer = AutoTokenizer.from_pretrained(
            load_source,
            token=hf_token,
            trust_remote_code=True,
            cache_dir=config["cache_dir"]
        )
        
    # Load model
    with Timer(f"Loading Causal LLM from '{load_source}'"):
        device_map = "cpu" if device == "cpu" else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            token=hf_token,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=config["cache_dir"]
        )
        
    log_resource_state("AFTER LOADING MODEL")
    return model, tokenizer
