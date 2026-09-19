from datasets import load_dataset
import re

dataset = load_dataset("araag2/MedMCQA", "source")
train = dataset["train"]

candidates = []

for item in train:
    explanation = item["exp"]

    if explanation is None:
        continue

    explanation = str(explanation).strip()

    # First-pass candidate pool
    if len(explanation) < 200:
        continue

    question = str(item["question"]).strip()

    # Basic quality flags
    flags = []

    if len(question) < 20:
        flags.append("short_question")

    if len(explanation) > 5000:
        flags.append("very_long_explanation")

    # Common encoding / formatting artifacts
    if "aEUR" in explanation:
        flags.append("encoding_artifact")

    if "�" in explanation:
        flags.append("replacement_character")

    # Obvious reference-heavy material
    if "Ref Robbins" in explanation:
        flags.append("reference_text")

    # Excessive repeated punctuation / formatting
    if re.search(r"[#]{3,}", explanation):
        flags.append("formatting_noise")

    candidates.append({
        "question": question,
        "explanation": explanation,
        "subject": item["subject_name"],
        "topic": item["topic_name"],
        "correct": item["cop"],
        "flags": flags
    })


print("=" * 80)
print("MEDMCQA FIRST-PASS QUALITY FILTER")
print("=" * 80)

print(f"\nOriginal training examples : {len(train):,}")
print(f"Candidate examples         : {len(candidates):,}")

flagged = [x for x in candidates if x["flags"]]

print(f"Flagged examples            : {len(flagged):,}")
print(
    f"Currently unflagged         : "
    f"{len(candidates) - len(flagged):,}"
)

print("\n" + "=" * 80)
print("FLAG COUNTS")
print("=" * 80)

from collections import Counter

flag_counts = Counter()

for item in candidates:
    for flag in item["flags"]:
        flag_counts[flag] += 1

for flag, count in flag_counts.most_common():
    print(f"{flag:30} : {count:,}")


# Show 20 flagged examples for manual inspection
print("\n" + "=" * 80)
print("SAMPLE FLAGGED EXAMPLES")
print("=" * 80)

for i, item in enumerate(flagged[:20], 1):

    print("\n" + "-" * 80)
    print(f"FLAGGED EXAMPLE {i}")
    print("FLAGS   :", item["flags"])
    print("SUBJECT :", item["subject"])
    print("QUESTION:", item["question"])
    print("EXPLANATION:")
    print(item["explanation"][:1000])


print("\n" + "=" * 80)
print("FILTER ANALYSIS COMPLETE")
print("=" * 80)