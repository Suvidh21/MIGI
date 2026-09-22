from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

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


# ============================================================
# FIND LOCAL QWEN SNAPSHOT
# ============================================================

snapshots_dir = MODEL_CACHE / "snapshots"

snapshots = [
    p for p in snapshots_dir.iterdir()
    if p.is_dir()
]

if not snapshots:
    raise FileNotFoundError(
        "Qwen 0.5B snapshot was not found."
    )

MODEL_PATH = snapshots[0]


# ============================================================
# CHECK ADAPTER
# ============================================================

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"MIGI adapter not found:\n{ADAPTER_PATH}"
    )

print("=" * 70)
print("DR. MIGI — BASE vs FINE-TUNED MODEL")
print("=" * 70)

print(f"\nBase model:")
print(MODEL_PATH)

print(f"\nMIGI adapter:")
print(ADAPTER_PATH)


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"\nGPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Tokenizer loaded")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading Qwen 2.5 0.5B base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    local_files_only=True,
)

base_model.eval()

print("✅ Base model loaded")


# ============================================================
# LOAD MIGI ADAPTER
# ============================================================

print("\nLoading MIGI LoRA adapter...")

migi_model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
    local_files_only=True,
)

migi_model.eval()

print("✅ MIGI adapter loaded")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_answer(model, question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical research assistant. "
                "Provide accurate, technically detailed medical "
                "information. Distinguish established facts from "
                "clinical interpretation and uncertainty. "
                "Do not invent patient information."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=400,

            temperature=0.7,

            top_p=0.9,

            do_sample=True,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# QUESTIONS
# ============================================================

questions = [

    "Explain the pathophysiology of multiple sclerosis.",

    "What is the mechanism of action of metformin?",

    "Explain why chronic hypertension can lead to left ventricular hypertrophy.",

]


# ============================================================
# RUN COMPARISON
# ============================================================

for number, question in enumerate(
    questions,
    start=1,
):

    print("\n")
    print("=" * 70)
    print(f"QUESTION {number}")
    print("=" * 70)

    print(f"\n{question}")

    print("\n")
    print("-" * 70)
    print("QWEN 2.5 0.5B — BASE MODEL")
    print("-" * 70)

    base_answer = generate_answer(
        base_model,
        question,
    )

    print(base_answer)


    print("\n")
    print("-" * 70)
    print("DR. MIGI — FINE-TUNED MODEL")
    print("-" * 70)

    migi_answer = generate_answer(
        migi_model,
        question,
    )

    print(migi_answer)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)