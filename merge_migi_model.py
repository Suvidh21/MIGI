from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# DR. MIGI V1 — MERGE LoRA INTO QWEN 0.5B
# ============================================================

PROJECT_DIR = Path("I:/suvidh/MIGI-main")

MODEL_CACHE = (
    PROJECT_DIR
    / "models"
    / "cache"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
)

ADAPTER_PATH = (
    PROJECT_DIR
    / "migi_qlora_output"
    / "full_training"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "models"
    / "MIGI-Qwen2.5-0.5B-v1"
)


print("=" * 70)
print("DR. MIGI V1 — MERGING LoRA ADAPTER")
print("=" * 70)


# ============================================================
# FIND BASE MODEL
# ============================================================

snapshots_dir = MODEL_CACHE / "snapshots"

snapshots = [
    p for p in snapshots_dir.iterdir()
    if p.is_dir()
]

if not snapshots:
    raise FileNotFoundError(
        "Qwen 2.5 0.5B snapshot not found."
    )

MODEL_PATH = snapshots[0]

print("\nBase model:")
print(MODEL_PATH)

print("\nMIGI adapter:")
print(ADAPTER_PATH)

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"MIGI adapter not found:\n{ADAPTER_PATH}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOutput:")
print(OUTPUT_PATH)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

print("✅ Tokenizer loaded")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base Qwen model in FP16...")

base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

print("✅ Base model loaded")


# ============================================================
# LOAD LoRA
# ============================================================

print("\nLoading MIGI LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
    local_files_only=True,
)

print("✅ MIGI adapter loaded")


# ============================================================
# MERGE
# ============================================================

print("\nMerging LoRA weights into Qwen...")

merged_model = model.merge_and_unload()

print("✅ LoRA successfully merged")


# ============================================================
# SAVE
# ============================================================

print("\nSaving standalone MIGI model...")

merged_model.save_pretrained(
    str(OUTPUT_PATH),
    safe_serialization=True,
)

tokenizer.save_pretrained(
    str(OUTPUT_PATH)
)

print("\n✅ MIGI model saved")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("🎉 MIGI V1 MERGE COMPLETE")
print("=" * 70)

print("\nStandalone model:")
print(OUTPUT_PATH)

print("\nThis model now contains:")
print("Qwen 2.5 0.5B + MIGI LoRA training")

print("\nThe original Qwen model was NOT modified.")

print("\nDone.")