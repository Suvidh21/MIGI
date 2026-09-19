# Lecture 3: Embeddings (Task 3)

## 1. What it does
An embedding converts discrete token IDs into dense, low-dimensional continuous vector representations ($x \in \mathbb{R}^{d_{model}}$) that capture semantic meaning.

## 2. Why it is needed
A simple integer (like ID 1045) conveys no relational information (e.g., the model cannot mathematically know if ID 1045 is semantically closer to ID 1046 or ID 98200). Continuous vectors represent words in a high-dimensional space where distance (e.g., Cosine Similarity) maps to semantic similarity.

## 3. How it works internally
1. **Token Embeddings ($W_{te}$)**: A large matrix of size $V \times d_{model}$ (where $V$ is vocabulary size and $d_{model}$ is the hidden dimension of the model). When a token ID $i$ is inputted, we perform a lookup operation (analogous to multiplying a one-hot vector $e_i$ by $W_{te}$) to retrieve the $i$-th row.
2. **Rotary Position Embeddings (RoPE)**: Traditional Transformers add a static positional vector to the token embedding. Qwen uses **RoPE (Rotary Position Embedding)**. Instead of adding a vector, RoPE applies a rotation matrix to the Query and Key vectors in the self-attention mechanism:
$$R_{\Theta, m}^d x$$
This rotates the vectors in 2D planes based on their sequence index $m$, mathematically capturing the relative distance between tokens.

## 4. What happens if it is removed
Without embeddings, neural networks cannot process tokens. Without positional representation (like RoPE), the model treats the input as a "bag of words", unable to distinguish "the dog bit the man" from "the man bit the dog".

## 5. Best Practices & Performance
Embedding lookups are memory-bandwidth bound operations (O(1) lookup in memory). Positional embeddings like RoPE are computed on-the-fly, which trades minor GPU computation to save parameters and support long contexts (up to 32k or 128k tokens).
