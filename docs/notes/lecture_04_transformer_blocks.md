# Lecture 4: Transformer Blocks (Task 4)

## 1. What it does
A Transformer Block is the fundamental building block of the model. It contains:
- Layer Normalization (specifically **RMSNorm**)
- Multi-Head/Grouped-Query Attention
- A Feed-Forward Network (FFN) with **SwiGLU** activation
- Residual (skip) connections

## 2. Why it is needed
Stacked Transformer blocks enable the model to build hierarchical abstractions of the text. Lower layers capture syntax and grammar, while deeper layers process clinical reasoning, complex dependencies, and contextual logic.

## 3. How it works internally
A single Qwen 2.5 block operates as follows:
Given input $x_{l-1}$:
1. **RMSNorm & Attention**:
$$a_l = x_{l-1} + \text{Attention}(\text{RMSNorm}(x_{l-1}))$$
RMSNorm normalizes activation vectors by their Root-Mean-Square instead of variance, which is computationally faster:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
2. **RMSNorm & Feed-Forward**:
$$x_l = a_l + \text{FFN}(\text{RMSNorm}(a_l))$$
Qwen uses **SwiGLU** in its FFN:
$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{gate}) \cdot x W_{up} \right) W_{down}$$
where $\text{Swish}(y) = y \cdot \sigma(\beta y)$.

```
   Input Vector (x)
         │
         ├───► RMSNorm ──► Self-Attention ──► Residual Add (x + Attn)
         │                                            │
         └────────────────────────────────────────────┼────────► (a)
                                                      │
         ┌────────────────────────────────────────────┘
         │
         ├───► RMSNorm ──► SwiGLU FFN ──────► Residual Add (a + FFN)
         │                                            │
         └────────────────────────────────────────────┴────────► Output Vector
```

## 4. What happens if it is removed
Without the residual connections, deeper models suffer from vanishing/exploding gradients and fail to train. Without the FFN, the model cannot store factual knowledge or map non-linear relationships.

## 5. Best Practices & Performance
Using RMSNorm and SwiGLU offers higher performance and training stability compared to traditional LayerNorm and ReLU/GELU activations.
