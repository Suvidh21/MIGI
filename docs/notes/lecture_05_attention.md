# Lecture 5: Attention Mechanism (Task 5)

## 1. What it does
Self-Attention calculates a dynamic weight representing how much each token in a sequence should focus on (attend to) every other token in the sequence.

## 2. Why it is needed
In medicine, terms depend heavily on context. In "The patient presents with acute myocardial infarction," the word "acute" modifies "myocardial infarction". Attention links these tokens together across distances.

## 3. How it works internally
1. Project the input $X$ into Queries ($Q$), Keys ($K$), and Values ($V$) using weight matrices $W_q, W_k, W_v$:
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
2. Apply RoPE rotation to $Q$ and $K$.
3. Compute attention scores using Scaled Dot-Product:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $M$ is a causal mask (triangular matrix of $-\infty$ for future elements and $0$ for past/present elements) ensuring tokens cannot attend to future tokens.
4. **Grouped-Query Attention (GQA)**: In standard Multi-Head Attention (MHA), every query head has a unique key and value head. In Multi-Query Attention (MQA), all query heads share one key and value head. Qwen uses **GQA**, which groups query heads (e.g., 8 queries per group) and assigns one Key/Value head group to each query group. This reduces the memory footprint of the Key-Value (KV) cache during generation.

## 4. What happens if it is removed
Without attention, the model is a standard feedforward neural network that cannot model long-range context or dynamic word-to-word relationships.

## 5. Best Practices & Performance
- **FlashAttention**: Re-orders attention matrix multiplication in SRAM to bypass memory bottlenecks.
- **KV Caching**: Caches keys ($K$) and values ($V$) from previous steps so we do not recalculate them for past tokens during generation loops.
