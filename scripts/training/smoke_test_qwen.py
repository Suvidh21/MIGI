import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("=" * 60)
print("DR. MIGI — Qwen 2.5 7B QLoRA Smoke Test")
print("=" * 60)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen 2.5 7B in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("\nModel loaded successfully!")

print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")

# Test prompt
messages = [
    {
        "role": "system",
        "content": (
            "You are DR. MIGI, a medical research and clinical reasoning "
            "assistant. Provide medically rigorous explanations and clearly "
            "distinguish established facts from uncertainty."
        ),
    },
    {
        "role": "user",
        "content": (
            "Explain the pathophysiological relationship between chronic "
            "hypertension and left ventricular hypertrophy."
        ),
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

print("\nGenerating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("\n" + "=" * 60)
print("DR. MIGI TEST RESPONSE")
print("=" * 60)
print(response)
print("=" * 60)

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"\nGPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved:  {reserved:.2f} GB")

print("\n✅ Qwen 2.5 7B 4-bit smoke test completed.")