# Lecture 2: Tokenization (Task 2)

## 1. What it does
Tokenization partitions a continuous string of text into discrete, indexable semantic sub-units called **tokens**, which are then mapped to integer IDs from a fixed vocabulary.

## 2. Why it is needed
Neural networks perform matrix algebra on continuous vectors, not text strings. Tokenization bridges human characters and numerical representations.

## 3. How it works internally
Qwen 2.5 employs **Byte-Pair Encoding (BPE)** based on `tiktoken`. BPE is a subword tokenization algorithm:
1. It begins with a base vocabulary containing all individual bytes/characters.
2. It iteratively identifies the most frequent adjacent pairs of tokens in a training corpus and merges them into a new vocabulary entry (e.g., `h` + `e` $\rightarrow$ `he`).
3. This is repeated until the target vocabulary size is reached. Qwen's vocabulary contains **151,936** tokens.

## 4. What happens if it is removed
Without tokenization, one could represent text at the character level (too long, high computational cost, loses semantic grouping) or the word level (cannot handle out-of-vocabulary words like newly developed medical terms).

## 5. Best Practices & Performance
Using pre-compiled tokenizers (written in Rust, like Hugging Face's fast tokenizers or `tiktoken`) prevents tokenization from becoming a CPU bottleneck during data preparation.
