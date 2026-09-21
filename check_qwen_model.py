import os
from pathlib import Path

print("=" * 70)
print("        DR. MIGI — LOCAL QWEN MODEL CHECK")
print("=" * 70)

# Common locations where Hugging Face models may exist
locations = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path("I:/"),
    Path("I:/suvidh"),
    Path("I:/suvidh/MIGI-main"),
]

found = []

print("\nSearching for Qwen model folders...\n")

for location in locations:
    if not location.exists():
        continue

    try:
        for path in location.rglob("*"):
            if path.is_dir() and "qwen" in path.name.lower():
                if path not in found:
                    found.append(path)
    except PermissionError:
        pass

if not found:
    print("❌ No Qwen folder found in the searched locations.")
    print("\nIf you know where the model is stored, we can check that path directly.")
    exit()

print(f"Found {len(found)} possible Qwen folder(s):\n")

for i, path in enumerate(found, 1):
    print(f"[{i}] {path}")

print("\n" + "=" * 70)
print("Checking model files")
print("=" * 70)

for path in found:

    # Only inspect directories that actually look like model directories
    model_files = []

    for file in path.rglob("*"):
        if file.is_file():
            if file.name.endswith((
                ".safetensors",
                ".bin",
                ".gguf",
                ".json",
                ".model"
            )):
                model_files.append(file)

    if not model_files:
        continue

    print("\n" + "-" * 70)
    print(f"MODEL DIRECTORY:")
    print(path)

    # Total size
    total_size = sum(f.stat().st_size for f in model_files)
    total_gb = total_size / (1024 ** 3)

    print(f"\nTotal model-related file size: {total_gb:.2f} GB")

    print("\nFiles:")

    for file in model_files:
        size_gb = file.stat().st_size / (1024 ** 3)
        size_mb = file.stat().st_size / (1024 ** 2)

        if size_gb >= 1:
            size_text = f"{size_gb:.2f} GB"
        else:
            size_text = f"{size_mb:.2f} MB"

        print(f"  {file.name:<55} {size_text}")

    # Look for model metadata
    config = path / "config.json"

    if config.exists():
        print("\n✅ config.json found")

        try:
            import json

            with open(config, "r", encoding="utf-8") as f:
                data = json.load(f)

            print("\nModel information:")

            for key in [
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "vocab_size"
            ]:
                if key in data:
                    print(f"  {key}: {data[key]}")

        except Exception as e:
            print(f"Could not read config.json: {e}")

    # Tokenizer
    tokenizer = path / "tokenizer.json"

    if tokenizer.exists():
        print("\n✅ tokenizer.json found")

    # Safetensors
    safetensors = list(path.rglob("*.safetensors"))

    if safetensors:
        print(f"\n✅ Safetensors files found: {len(safetensors)}")

    # GGUF
    gguf = list(path.rglob("*.gguf"))

    if gguf:
        print(f"\n✅ GGUF files found: {len(gguf)}")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)