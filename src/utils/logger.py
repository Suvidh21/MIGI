import logging
import time
import os
import psutil
import torch

# Configure the logger formatting for professional production standards.
# What it does: Sets up a root configuration for console output showing time, log level, and the message.
# Why it is needed: Provides standard observability across the entire app execution lifecycle.
# Best practices: Use structured formatters so logs can be ingested by monitoring systems.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("DR_MIGI")

def get_memory_usage():
    """
    Returns current memory consumption metrics of the system.
    
    Why it is needed:
        LLMs are memory intensive. We must monitor system RAM (especially on CPU execution)
        to prevent Out-Of-Memory (OOM) crashes and system page swaps.
        
    How it works internally:
        Queries the operating system through `psutil` for RAM, and `torch.cuda` for VRAM if available.
    """
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    
    device_info = {}
    if torch.cuda.is_available():
        # CUDA VRAM stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        device_info["cuda_vram_allocated_mb"] = allocated
        device_info["cuda_vram_reserved_mb"] = reserved
        
    return {
        "system_ram_mb": ram_usage_mb,
        "cuda_info": device_info
    }

def log_resource_state(stage: str):
    """
    Utility wrapper to print memory resources at specific execution stages.
    """
    mem = get_memory_usage()
    cuda_str = ""
    if mem["cuda_info"]:
        c = mem["cuda_info"]
        cuda_str = f" | VRAM: {c['cuda_vram_allocated_mb']:.1f}MB/{c['cuda_vram_reserved_mb']:.1f}MB"
    logger.info(f"[{stage}] RAM Usage: {mem['system_ram_mb']:.1f}MB{cuda_str}")

class Timer:
    """
    A context manager to measure execution duration of code blocks.
    
    How it works:
        Uses high-resolution monotonic clock (`time.perf_counter()`) to avoid NTP synchronization adjustments.
    """
    def __init__(self, description: str):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        logger.info(f"{self.description} took {duration:.4f} seconds.")
