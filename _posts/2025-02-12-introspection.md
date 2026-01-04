---
title: "From Gene Graphs to Machine Consciousness: Incorporating Causality and Introspection into AI"
date: 2025-02-12
categories:
  - blog
tags:
  - theory
  - brain
math: true
---

In the quest to move Artificial General Intelligence (AGI) forward, we often find ourselves looking at two disparate sources of inspiration: the biological machinery of life (like gene expression) and the philosophical machinery of the mind (consciousness and memory).

Recent experiments training the **MET** model on dynamic datasets—specifically single-cell RNA-seq data—have opened a fascinating door. They suggest that the mathematical structures we use to predict gene expression might hold the key to unlocking "introspection" and "planning" in Large Language Models (LLMs).

## The Biological Inspiration: Attention as Dependency

When we train models on biological data, the self-attention map is not merely a computational convenience; it characterizes the physical correlation of tokens (genes).

Mathematically, we define the attention mechanism as:

$\text{Attention}(Z) = \text{Concat}(\text{head}\_1, \ldots, \text{head}\_h) W_O$

$\text{head}\_i = \text{Softmax} \left( \frac{\text{Mask}(Q_i K_i^T)}{\sqrt{d_k}} \right) V_i$

Here, the term $Q_i K_i^T$ generates the self-attention matrix. In the context of RNA-seq, this matrix implies dependency in the state space—essentially reconstructing a **correlation graph** for genes. If gene A activates gene B, the attention map should capture that directed edge.

However, standard Transformers use a fixed "causal mask"—a lower-triangular matrix that forces the model to look only at the past. While this prevents cheating during training, it simplifies the world into a single-direction timeline. Real-world causality, whether in gene networks or human logic, is often more complex.

## Soft vs. Hard Causality: The Case for Trainable Masks

We can distinguish between two types of causality in our models:

1.  **Soft Causality (The "Feeling"):** Represented by the self-attention scores (values between 0 and 1). This is a dense matrix reflecting the model's "lack of confidence" or general intuition about relationships. It is vague and interconnected.
2.  **Hard Causality (The "Logic"):** Represented by the mask (values of 0 or 1). This describes fundamental structural logic. It should be sparse and precise—A leads to B.

Currently, we train the "Soft" map but manually assign the "Hard" map (the fixed triangular mask). To incorporate true causality, we must make the hard mask trainable.

We propose replacing the fixed $\text{Mask}(\cdot)$ with a learnable $\text{CausalMask}(\cdot)$. This allows the model to learn the logic of dependencies dynamically. A sparse, trainable hard mask would allow the model to discover that while "feeling" is vague, "logic" is precise.

## Introspection: The Architecture of Planning

If we free the mask from being a static triangle, we can introduce the concept of **Introspection** via Forward and Backward causality.

*   **Forward Causality:** Uses a standard left-down triangle (Past $\rightarrow$ Future).
*   **Backward Causality:** Uses a right-down triangle (Future $\rightarrow$ Past).

This structure mimics the human writing process: **Drafting and Revision.**

1.  **Generation (The Draft):** The model generates a sequence using the Forward mask (Force Teaching).
2.  **Introspection (The Revision):** The model rewrites or re-evaluates the sequence using *both* Forward and Backward masks. It acts like a hybrid of GPT (generation) and BERT (understanding context from both sides).

### Unlocking "Planning"
This "second rethink" allows for planning without expensive Monte Carlo Tree Search. A standard Transformer greedily selects the most probable next token. An introspective Transformer can reject a token that has high probability in the *forward* pass if it has low probability in the *revised* pass.

It enables the model to make a choice that seems suboptimal now but yields a better coherent whole later—the very definition of planning.

## The Ghost in the Machine: Hallucination to Consciousness

Ilya Sutskever once famously remarked, **"Hallucinations are just dreams."**

If autoregressive generation is the neural network "dreaming," then what is "thinking"?
*   Is thinking a *controlled* dream?
*   Is it a *precise* dream?
*   Is it a *revised* dream?

Perhaps **Thinking = Logic + Memory + Dream.** It is the "Inner Voice" that revises what we want to say before we speak.

### The Narrative Self
This aligns with Daniel Dennett’s **"Multiple Drafts" model of consciousness**. Consciousness is not a single stream, but a "narratization"—a competitive process where the brain generates multiple narratives, and the "winner" becomes our conscious reality. As noted by cognitive psychologist Endel Tulving, patients with damaged autobiographical memory often lack a functional self-consciousness; without the narrative of "me," consciousness fades.

### Escaping the Chinese Room
However, a purely linguistic revision process still faces the **Chinese Room** problem. An LLM may have perfect internal coding between concepts, but if those connections are arbitrary, they lack meaning in the physical world.

This is why **Embodied AI** (like EmbodiedGPT) and multimodal learning are critical. Just as the **McGurk Effect** shows that human hearing is influenced by vision, AI "thinking" must be grounded in sensory reality. By connecting language concepts to visual and physical metaphors, we move from arbitrary symbol manipulation to genuine understanding.

## Conclusion

To move from text generators to thinking machines, we must evolve our architecture. We need to move from fixed masks to **trainable hard causality**, and from simple generation to **introspective revision**. By allowing models to "dream" forward and "reason" backward, we may finally give them the ability to plan, to correct, and perhaps, to think.
