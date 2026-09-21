from datasets import load_dataset, Dataset
from collections import defaultdict
import random
import json

# ============================================================
# CONFIG
# ============================================================

SEED = 42
TARGET_EXAMPLES = 40000
MIN_EXPLANATION_LENGTH = 200

random.seed(SEED)

# ============================================================
# LOAD MEDMCQA
# ============================================================

print("Loading MedMCQA...")

dataset = load_dataset("araag2/MedMCQA", "source")
train = dataset["train"]

print(f"Original training examples: {len(train):,}")

# ============================================================
# CREATE CANDIDATE POOL
# ============================================================

candidates = []

for item in train:

    explanation = item["exp"]

    if explanation is None:
        continue

    explanation = str(explanation).strip()

    if len(explanation) < MIN_EXPLANATION_LENGTH:
        continue

    question = str(item["question"]).strip()

    if not question:
        continue

    # Validate options
    options = [
        item["opa"],
        item["opb"],
        item["opc"],
        item["opd"]
    ]

    if any(x is None or str(x).strip() == "" for x in options):
        continue

    candidates.append(item)

print(f"Candidate pool: {len(candidates):,}")

# ============================================================
# BALANCE BY SUBJECT
# ============================================================

by_subject = defaultdict(list)

for item in candidates:
    subject = item["subject_name"]

    if subject is None:
        subject = "Unknown"

    by_subject[subject].append(item)

print("\nAvailable examples by subject:")

for subject, items in sorted(
    by_subject.items(),
    key=lambda x: len(x[1]),
    reverse=True
):
    print(f"{subject:35} {len(items):,}")

# ============================================================
# SAMPLE BALANCED DATASET
# ============================================================

subjects = list(by_subject.keys())

# Shuffle every subject independently
for subject in subjects:
    random.shuffle(by_subject[subject])

# Calculate approximately equal target
per_subject = TARGET_EXAMPLES // len(subjects)

selected = []

# First pass: balanced sampling
for subject in subjects:

    available = by_subject[subject]

    amount = min(per_subject, len(available))

    selected.extend(available[:amount])

# Second pass: fill remaining slots
remaining = TARGET_EXAMPLES - len(selected)

if remaining > 0:

    used_ids = set(id(x) for x in selected)

    remaining_pool = [
        x for x in candidates
        if id(x) not in used_ids
    ]

    random.shuffle(remaining_pool)

    selected.extend(
        remaining_pool[:remaining]
    )

# Shuffle final dataset
random.shuffle(selected)

# Limit exactly
selected = selected[:TARGET_EXAMPLES]

print(f"\nSelected examples: {len(selected):,}")

# ============================================================
# CONVERT TO MIGI INSTRUCTION FORMAT
# ============================================================

migi_examples = []

for item in selected:

    question = str(item["question"]).strip()

    options = (
        f"A. {str(item['opa']).strip()}\n"
        f"B. {str(item['opb']).strip()}\n"
        f"C. {str(item['opc']).strip()}\n"
        f"D. {str(item['opd']).strip()}"
    )

    # MedMCQA uses 1=A, 2=B, 3=C, 4=D
    correct_index = int(item["cop"])

    letters = ["A", "B", "C", "D"]

    correct_letter = letters[correct_index - 1]

    correct_text = str(
        item[f"op{letters[correct_index - 1].lower()}"]
    ).strip()

    explanation = str(item["exp"]).strip()

    subject = str(
        item["subject_name"]
        if item["subject_name"] is not None
        else "Medicine"
    ).strip()

    topic = item["topic_name"]

    if topic is None:
        topic = ""

    topic = str(topic).strip()

    # --------------------------------------------------------
    # MIGI instruction
    # --------------------------------------------------------

    instruction = (
        "Answer the following medical question using "
        "accurate medical reasoning. Explain why the correct "
        "answer is appropriate and relate the explanation "
        "to the underlying medical concept."
    )

    user_content = (
        f"Medical Question:\n"
        f"{question}\n\n"
        f"Options:\n"
        f"{options}"
    )

    assistant_content = (
        f"Correct Answer: {correct_letter}. {correct_text}\n\n"
        f"Medical Reasoning:\n"
        f"{explanation}"
    )

    migi_examples.append({
        "instruction": instruction,
        "input": user_content,
        "output": assistant_content,
        "subject": subject,
        "topic": topic
    })

# ============================================================
# CREATE HF DATASET
# ============================================================

migi_dataset = Dataset.from_list(migi_examples)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split = migi_dataset.train_test_split(
    test_size=0.05,
    seed=SEED
)

# ============================================================
# SAVE
# ============================================================

output_path = "migi_medical_40k"

split.save_to_disk(output_path)

print("\n" + "=" * 80)
print("MIGI DATASET CREATED")
print("=" * 80)

print(f"Train examples : {len(split['train']):,}")
print(f"Validation     : {len(split['test']):,}")

print(f"\nSaved to: {output_path}")

# ============================================================
# SHOW ONE EXAMPLE
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE MIGI TRAINING EXAMPLE")
print("=" * 80)

example = split["train"][0]

print("\nINSTRUCTION:")
print(example["instruction"])

print("\nINPUT:")
print(example["input"])

print("\nOUTPUT:")
print(example["output"])

print("\nSUBJECT:")
print(example["subject"])

print("\nTOPIC:")
print(example["topic"])