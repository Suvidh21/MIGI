from datasets import load_dataset
from collections import Counter

dataset = load_dataset("araag2/MedMCQA", "source")
train = dataset["train"]

print("\n" + "=" * 90)
print("MEDMCQA — USABLE DATA BY SUBJECT")
print("=" * 90)

# Count examples by subject at different explanation-length thresholds
thresholds = [0, 50, 200, 500]

results = {}

for threshold in thresholds:

    counts = Counter()

    for item in train:
        explanation = item["exp"]

        if explanation is None:
            continue

        explanation = str(explanation).strip()

        if len(explanation) >= threshold:
            counts[item["subject_name"]] += 1

    results[threshold] = counts


# --------------------------------------------------
# Overall counts
# --------------------------------------------------

print("\nOVERALL USABLE EXAMPLES")
print("-" * 90)

for threshold in thresholds:

    total = sum(results[threshold].values())

    print(
        f"Explanation >= {threshold:>3} chars : "
        f"{total:7,} examples "
        f"({total / len(train) * 100:.2f}%)"
    )


# --------------------------------------------------
# Subject distribution
# --------------------------------------------------

subjects = sorted(
    set(item["subject_name"] for item in train)
)

print("\n" + "=" * 90)
print("SUBJECT DISTRIBUTION")
print("=" * 90)

header = (
    f"{'Subject':35} "
    f"{'>=0':>10} "
    f"{'>=50':>10} "
    f"{'>=200':>10} "
    f"{'>=500':>10}"
)

print(header)
print("-" * 90)

for subject in subjects:

    values = [
        results[threshold][subject]
        for threshold in thresholds
    ]

    print(
        f"{str(subject):35} "
        f"{values[0]:10,} "
        f"{values[1]:10,} "
        f"{values[2]:10,} "
        f"{values[3]:10,}"
    )


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)