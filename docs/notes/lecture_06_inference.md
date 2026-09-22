# Lecture 6: Inference & Decoders (Task 6)

## 1. What it does
Inference is the execution of a pre-trained model to produce new text. It operates in two phases:
- **Prefill (Context processing)**: The input prompt is evaluated. Keys and Values are cached.
- **Decoding (Autoregressive loop)**: The model predicts one token, appends it to the context, and repeats.

## 2. Why it is needed
To interact with humans, the system must generate natural language text dynamically.

## 3. How it works internally
1. The last layer outputs a vector $h_{last} \in \mathbb{R}^{d_{model}}$ for the final token.
2. The language modeling head (LM Head) projects this vector back to the vocabulary size ($V = 151,936$):
$$\text{Logits} = h_{last} W_{lm\_head}$$
3. **Sampling**: Logits are raw unnormalized scores. We apply:
   - **Temperature ($T$)**: Modulates randomness. Logits are divided by $T$. Lower $T$ approaches argmax (greedy search). Higher $T$ flattens the distribution.
   - **Top-P (Nucleus Sampling)**: Keeps only the top tokens whose cumulative probability exceeds $P$.
   - **Top-K**: Keeps only the top $K$ highest-probability tokens.
4. Softmax is computed over the filtered logits:
$$P(t_i) = \frac{e^{\text{Logits}_i / T}}{\sum_j e^{\text{Logits}_j / T}}$$
5. We sample a token ID from this distribution, feed it back to the input, and repeat.

## 4. What happens if it is removed
Without sampling, we would perform greedy decoding, leading to repetitive, deterministic, and unnatural text generation.

## 5. Best Practices & Performance
Use KV caching to avoid quadratic complexity ($O(N^2)$) on every new token. Caching brings complexity down to $O(N)$ memory lookups during decoding.
