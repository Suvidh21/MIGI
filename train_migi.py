from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# DR. MIGI — QWEN 2.5 0.5B QLoRA TRAINING
# ============================================================
#
# REAL TRAINING RUN
#
# Base model:
#   Qwen 2.5 0.5B Instruct
#
# Dataset:
#   38,000 training examples
#   2,000 validation examples
#
# Training:
#   QLoRA / 4-bit
#   LoRA
#   1 epoch
#   768 max sequence length
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path("I:/suvidh/MIGI-main")

# Existing LOCAL Qwen 2.5 0.5B
MODEL_CACHE = (
    PROJECT_DIR
    / "models"
    / "cache"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
)

# Cleaned dataset created earlier
DATASET_PATH = (
    PROJECT_DIR
    / "migi_medical_40k_clean"
)

# Final training output
OUTPUT_DIR = (
    PROJECT_DIR
    / "migi_qlora_output"
    / "full_training"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_LENGTH = 768

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

EPOCHS = 1


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("DR. MIGI — QWEN 2.5 0.5B QLoRA TRAINING")
print("=" * 70)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available. "
        "Training cannot proceed safely."
    )


gpu_name = torch.cuda.get_device_name(0)

gpu_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)


print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_vram:.2f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")


# ============================================================
# FIND LOCAL QWEN MODEL
# ============================================================

print("\nSearching for local Qwen 2.5 0.5B model...")


if not MODEL_CACHE.exists():

    raise FileNotFoundError(
        f"\nQwen model cache not found:\n{MODEL_CACHE}"
    )


snapshots_dir = MODEL_CACHE / "snapshots"


if snapshots_dir.exists():

    snapshots = [
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir()
    ]

else:

    snapshots = []


if snapshots:

    # Select the available Hugging Face snapshot
    model_path = snapshots[0]

else:

    model_path = MODEL_CACHE


print("\nUsing model:")

print(model_path)


# ============================================================
# VERIFY MODEL FILE
# ============================================================

model_file = model_path / "model.safetensors"


if not model_file.exists():

    raise FileNotFoundError(
        f"""
model.safetensors was not found:

{model_file}

The local Qwen 0.5B model may be incomplete.
"""
    )


print("✅ Qwen 2.5 0.5B model found")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading cleaned MIGI dataset...")


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"""
Dataset not found:

{DATASET_PATH}
"""
    )


dataset = load_from_disk(
    str(DATASET_PATH)
)


print("\nDataset:")

print(dataset)


train_dataset = dataset["train"]

validation_dataset = dataset["test"]


print(
    f"\nTraining examples: "
    f"{len(train_dataset)}"
)


print(
    f"Validation examples: "
    f"{len(validation_dataset)}"
)


# ============================================================
# VERIFY DATASET SIZE
# ============================================================

if len(train_dataset) != 38000:

    print(
        "\n⚠️ WARNING:"
        f" Expected 38,000 training examples, "
        f"but found {len(train_dataset)}."
    )


if len(validation_dataset) != 2000:

    print(
        "\n⚠️ WARNING:"
        f" Expected 2,000 validation examples, "
        f"but found {len(validation_dataset)}."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    local_files_only=True,
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print("✅ Tokenizer loaded")


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("\nConfiguring 4-bit QLoRA...")


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD QWEN MODEL
# ============================================================

print("\nLoading Qwen 2.5 0.5B in 4-bit...")


model = AutoModelForCausalLM.from_pretrained(

    str(model_path),

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.float16,

    local_files_only=True,
)


# Important for gradient checkpointing
model.config.use_cache = False


print("✅ Qwen model loaded")


# ============================================================
# PREPARE MODEL FOR LoRA
# ============================================================

print("\nPreparing model for LoRA...")


model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nConfiguring LoRA...")


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config
)


print("\n✅ LoRA adapter attached")


model.print_trainable_parameters()


# ============================================================
# TOKENIZE DATASET
# ============================================================

print("\nTokenizing training dataset...")


def tokenize_function(example):

    instruction = example["instruction"]

    input_text = example["input"]

    output_text = example["output"]


    text = instruction


    if input_text:

        text += "\n\n"

        text += input_text


    text += "\n\n"

    text += output_text


    return tokenizer(

        text,

        truncation=True,

        max_length=MAX_LENGTH,
    )


# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

train_tokenized = train_dataset.map(

    tokenize_function,

    remove_columns=train_dataset.column_names,

    desc="Tokenizing training data",
)


# ------------------------------------------------------------
# VALIDATION DATA
# ------------------------------------------------------------

print("\nTokenizing validation dataset...")


validation_tokenized = validation_dataset.map(

    tokenize_function,

    remove_columns=validation_dataset.column_names,

    desc="Tokenizing validation data",
)


print("\n✅ Tokenization complete")


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=False,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("\nConfiguring training...")


training_args = TrainingArguments(

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir=str(OUTPUT_DIR),


    # --------------------------------------------------------
    # BATCHING
    # --------------------------------------------------------

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),


    # --------------------------------------------------------
    # LEARNING
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    num_train_epochs=EPOCHS,


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_steps=10,


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=500,


    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=500,

    save_total_limit=2,


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    fp16=True,

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # MEMORY-EFFICIENT OPTIMIZER
    # --------------------------------------------------------

    optim="paged_adamw_8bit",


    # --------------------------------------------------------
    # NO WANDB / EXTERNAL LOGGING
    # --------------------------------------------------------

    report_to="none",


    # --------------------------------------------------------
    # DATASET HANDLING
    # --------------------------------------------------------

    remove_unused_columns=False,
)


# ============================================================
# CREATE TRAINER
# ============================================================

print("\nCreating Trainer...")


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=validation_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,
)


print("✅ Trainer created")


# ============================================================
# FINAL CONFIGURATION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    "\nBase model:"
    "\n  Qwen 2.5 0.5B-Instruct"
)

print(
    "\nTraining dataset:"
    "\n  38,000 examples"
)

print(
    "\nValidation dataset:"
    "\n  2,000 examples"
)

print(
    "\nTraining method:"
    "\n  4-bit QLoRA + LoRA"
)

print(
    f"\nMaximum sequence length:"
    f"\n  {MAX_LENGTH} tokens"
)

print(
    f"\nBatch size:"
    f"\n  {TRAIN_BATCH_SIZE}"
)

print(
    f"\nGradient accumulation:"
    f"\n  {GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"\nEffective batch size:"
    f"\n  "
    f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"\nLearning rate:"
    f"\n  {LEARNING_RATE}"
)

print(
    f"\nEpochs:"
    f"\n  {EPOCHS}"
)

print(
    "\nOutput:"
    f"\n  {OUTPUT_DIR}"
)

print("\n")
print("=" * 70)
print("🚀 STARTING REAL MIGI TRAINING")
print("=" * 70)
print(
    "\nDo not close this terminal or put the PC to sleep."
)
print(
    "Training progress will be displayed below."
)
print("=" * 70)


# ============================================================
# START TRAINING
# ============================================================

trainer.train()


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print("\nSaving final MIGI LoRA adapter...")


trainer.save_model(
    str(OUTPUT_DIR)
)


tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    print(
        f"\nGPU memory allocated:"
        f" {allocated:.2f} GB"
    )

    print(
        f"GPU memory reserved:"
        f" {reserved:.2f} GB"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("🎉 MIGI TRAINING COMPLETED")
print("=" * 70)

print("\nBase model:")
print("Qwen 2.5 0.5B-Instruct")

print("\nTraining:")
print("38,000 examples × 1 epoch")

print("\nValidation:")
print("2,000 examples")

print("\nMethod:")
print("4-bit QLoRA + LoRA")

print("\nFinal adapter:")
print(OUTPUT_DIR)

print("\nDone.")