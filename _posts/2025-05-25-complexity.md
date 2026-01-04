---
title: "The essence of complexity: from algorithm length to manifold volume"
date: 2025-05-25
categories:
  - blog
tags:
  - theory
  - brain
math: true
---


What exactly does "complex" mean?

In the contexts of science and philosophy, we frequently use the term "Complexity," yet defining it precisely is notoriously difficult. Intuition tells us that a crystal is ordered, but not complex; a gas is random, but also not complex. True complexity seems to reside somewhere in between—at the edge of order and chaos.

This article explores several mainstream definitions of complexity, attempting to present a panoramic view from the perspectives of information theory, statistical mechanics, and manifold geometry.

---

## 1. Kolmogorov-Chaitin Complexity

The most classic measure of complexity comes from information theory. **Kolmogorov Complexity $K(x)$** defines the complexity of a sequence as: **the length of the shortest program for a Universal Turing Machine (UTM) capable of producing that sequence.**

In short, it is about "compression." If an object can be described by a very short formula or code (such as the algorithm for generating $\pi$, or Newton's laws), it is, in a sense, "simple," even if it contains infinite information.

### The Limitation: Randomness $\neq$ Complexity
While Kolmogorov complexity is theoretically fair, it presents two major problems:

1.  **Uncomputability**: It is formally undecidable; there is no general algorithm to determine $K(x)$ for any given sequence.
2.  **Intuitive Discrepancy**: Consider a completely random sequence of bits. As the sequence length increases, we cannot find a program shorter than the sequence itself to generate it. Consequently, Kolmogorov complexity increases monotonically with randomness.
    *   This leads to a paradox: **A completely random sequence is measured as "most complex."**
    *   This contradicts our physical intuition. A random uniform distribution (like coin flips) is structurally trivial—banal, even—containing no interesting patterns or organization.

### Algorithm Length vs. Computational Effort
It is also worth noting that the brevity of physical laws (the algorithm) does not imply the simplicity of the computation. 

For example, consider the **Partition Function** in statistical mechanics. The algorithm describing it is rather short (low Kolmogorov complexity), but computing the actual solution is often intractable. The computational time complexity is exponentially dependent on the number of degrees of freedom in the system.

---

## 2. Statistical Complexity

To correct the intuitive bias of Kolmogorov complexity, we need a metric that distinguishes "randomness" from "structure." This brings us to **Statistical Complexity ($C_\mu(x)$)**, proposed by James P. Crutchfield and colleagues.

### Core Idea
Statistical complexity measures complexity by introducing **historical information**. It is interpreted as: **the minimum amount of historical information required to make optimal forecasts of the future at a given error rate $h_\mu$.**

*   **Perfectly Ordered Systems** (e.g., crystals): Very little information is needed to predict the future. Complexity is low.
*   **Perfectly Random Systems** (e.g., ideal gases): Past information does not help predict the future (because it is unpredictable). No history needs to be stored. Complexity is low.
*   **Complex Systems**: These contain both rules and surprises. The system has internal structure, where past states constrain future possibilities. It is here, between order and noise, that $C_\mu$ reaches its maximum.

---

## 3. The Edge of Chaos

Why do complex systems tend to emerge in specific regions? C.G. Langton proposed the famous **Edge of Chaos** hypothesis.

Langton argued that systems capable of maximizing computational capability and complex behavior tend to exist at the **phase transition edge between ordered and chaotic phases**.

*   **Emergence**: The hallmark of chaos is the emergence of macroscopic uncertainty from underlying, completely deterministic mechanisms.
*   **Critical Points**:
    *   In computer models of **Boids** (bird flocks), complex group behavior emerges only when noise is at a moderate level—neither too low nor too high.
    *   In the **Ising Model**, the system undergoes a phase transition at a critical temperature between total disorder (high temp) and total order (low temp).

As Joseph Lizier noted, local information dynamics in distributed computation are most active at these critical points. Even without a distinct physical phase transition, complexity can rise significantly due to an increase in correlations between degrees of freedom (Feldman et al., 2008).

---

## 4. My proposal: A Manifold Perspective

Finally, we can attempt to redefine complexity from the perspective of geometry and topology: **Maximum Compressed Embedding** or the **Volume of the Minimum Manifold**.

This view suggests that the solution space of a problem can be embedded into a manifold. Complexity is defined by the **volume of the minimum manifold** that embeds this solution space. This volume provides an estimation of the minimum search effort needed to find a solution.

### Manifolds in Dynamical Systems
We can understand this by looking at the steady-states of dynamical systems:

1.  **Deterministic Systems**: The embedding manifold is effectively a single line (the trajectory is determined). The volume approaches zero, and the search space is minimal.
2.  **Completely Stochastic Systems**: While seemingly chaotic, in probability space, all states share the same probability. The entire state space degenerates into a single uniform state. There is no complex geometric structure to explore.
3.  **Mixed Systems (Deterministic + Stochastic)**:
    *   **Uncorrelated**: If degrees of freedom are not correlated, the manifold degrades into a low-dimensional structure (e.g., 1 dimension for freedom + 1 for state values). Complexity is limited.
    *   **Strongly Correlated**: When the system is strongly correlated, the manifold expands in multidimensional space and cannot be easily reduced. The manifold volume expands alongside the state space. In this scenario, the system exhibits the **highest complexity**.

## Conclusion

From the tape length of a Turing machine to predictive models in statistical mechanics, and finally to the volume of geometric manifolds, our understanding of complexity is constantly deepening.

True complexity is not mere chaos, nor is it rigid order. It is the **manner in which information flows through structure**, the **entanglement of correlations in multidimensional space**, and the interplay of deterministic laws within computational irreducibility.

---

### References

*   C.G. Langton, *Computation at the edge of chaos: phase transitions and emergent computation*. Phys. D 42(1–3), 12–37 (1990)
*   Crutchfield, James P. *"The calculi of emergence: computation, dynamics and induction."* Physica D: Nonlinear Phenomena 75.1-3 (1994)
*   Shalizi, Cosma Rohilla, and James P. Crutchfield. *"Computational mechanics: Pattern and prediction, structure and simplicity."* Journal of statistical physics 104 (2001)
*   D.P. Feldman, C.S. McTague, J.P. Crutchfield, *The organization of intrinsic computation: complexity-entropy diagrams and the diversity of natural information processing*. Chaos 18(4), 043106 (2008)

