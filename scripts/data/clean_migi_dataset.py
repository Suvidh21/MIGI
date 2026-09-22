from datasets import load_from_disk
import re

INPUT_PATH = "migi_medical_40k"
OUTPUT_PATH = "migi_medical_40k_clean"


def clean_explanation(text):
    if text is None:
        return ""

    text = str(text).strip()

    # Remove obvious dangling reference fragments at the END
    text = re.sub(
        r"\s*(Ref\.?|Ref:|Reference:?)\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Clean excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


print("Loading MIGI dataset...")

dataset = load_from_disk(INPUT_PATH)

print(f"Train examples: {len(dataset['train']):,}")
print(f"Validation examples: {len(dataset['test']):,}")


def clean_example(example):
    example["output"] = clean_explanation(example["output"])
    return example


print("\nCleaning train set...")
train = dataset["train"].map(clean_example)

print("Cleaning validation set...")
validation = dataset["test"].map(clean_example)


clean_dataset = {
    "train": train,
    "test": validation
}

from datasets import DatasetDict

clean_dataset = DatasetDict(clean_dataset)

print("\nSaving cleaned dataset...")

clean_dataset.save_to_disk(OUTPUT_PATH)

print("\n" + "=" * 70)
print("CLEAN DATASET CREATED")
print("=" * 70)

print(f"Location: {OUTPUT_PATH}")
print(f"Train: {len(train):,}")
print(f"Validation: {len(validation):,}")

print("\nSample output:")
print(train[0]["output"])