# Lecture 1: LLM Architecture (Task 1)

## 1. What it does
An autoregressive causal language model is a neural network trained to model the probability distribution of sequences of text. Specifically, given a sequence of tokens $(t_1, t_2, \dots, t_k)$, the model predicts the probability distribution of the next token $t_{k+1}$:
$$P(t_{k+1} \mid t_1, t_2, \dots, t_k)$$

## 2. Why it is needed
Generative text tasks require predicting sequential relationships. In medicine, clinical reasoning is sequential: symptom presentation leads to history, which leads to physical exam finding, leading to lab tests, and finally to a differential diagnosis. Modelling this causally ensures the model generates cohesive explanations step by step.

## 3. How it works internally
The model uses a **Decoder-Only Transformer** architecture. Unlike the original encoder-decoder Transformer, a decoder-only model processes input tokens through stacked self-attention blocks with a causal mask, and feeds the outputs directly to a language modeling linear head.

```
+-------------------------------------------------------------+
| Input Text: "The patient has a history of..."               |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Tokenization: Text is converted into token IDs [1037, 8560]  |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Embedding Layer: IDs map to continuous vectors (d_model)   |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| RoPE (Rotary Positional Embeddings): Encodes token order    |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Stacked Transformer Blocks (x N Blocks)                     |
|  ├─ RMSNorm -> Layer normalization                          |
|  ├─ Self-Attention (GQA) -> Contextual relationship mapping |
|  └─ SwiGLU Feed-Forward Network (FFN) -> Non-linear memory  |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Language Modeling Head: Linear projection to vocab size     |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
| Logits -> Softmax -> Sampling -> Next Token ID              |
+-------------------------------------------------------------+
```

## 4. What happens if it is removed
Without the causal constraint (e.g., if we used a bidirectional encoder like BERT), the model would attend to "future" tokens during training. It would fail at generation tasks because it wouldn't learn to generate the next token from left to right in a self-contained manner.

## 5. Best Practices & Performance
For local resource-constrained execution (like a CPU with limited RAM), minimizing model parameters or utilizing quantization (e.g., INT4, INT8, FP16) reduces memory consumption. Qwen2.5-0.5B-Instruct uses FP32 on CPU (occupying ~2GB RAM) and FP16/BF16 on GPU (occupying ~1GB VRAM).
