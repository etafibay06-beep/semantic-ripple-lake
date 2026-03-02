# Semantic Ripple Lake - Dynamical Systems Model for Submission

> Date: 2026-03-01
> Author: Organized by 超哥
> Version: Final Submission

---

## 1. Semantic Ripple Lake: Continuous Dynamical System Model

### 1.1 State Space Definition

#### Text Space
$$\mathcal{T}$$

#### Semantic Embedding Mapping
$$E : \mathcal{T} \rightarrow \mathbb{R}^d$$

#### Memory Vector Set
$$\mathcal{M} = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N\} \quad \mathbf{v}_i \in \mathbb{R}^d$$

Construct Semantic Graph:
- kNN Adjacency Matrix $W$
- Degree Matrix $D$
- Graph Laplacian: $L = D - W$

---

### 1.2 Query Trigger (Stone into Lake)

Query vector $b$ triggers activation:
$$u_i = \text{sim}(q, v_i) \cdot \text{time\_factor} \cdot \text{mastery\_factor}$$

### 1.3 Energy Function

$$\mathcal{F}(\Phi) = \frac{\lambda}{2}\Phi^\top L\Phi - \frac{\gamma}{2}|\Phi|^2 + \frac{\beta}{4}\sum_i \Phi_i^4 - \tau H(p)$$

Where:
- $\lambda$: Graph smoothness coefficient
- $\gamma$: Decay coefficient  
- $\beta$: Self-enhancement coefficient
- $\tau$: Entropy coefficient
- $p$: Energy distribution probability

### 1.4 IMEX Scheme

$$\Phi_{t+1} = (I + \eta\lambda L)^{-1}(\Phi_t + \eta u - \eta\gamma\Phi_t + \eta\beta\Phi_t^3)$$

### 1.5 Critical Control

Branching factor:
$$b_t = \frac{|S_t \setminus S_{t-1}|}{|S_{t-1}|}$$

Control law:
$$\gamma_{t+1} = \gamma_t - k(b_t - 1)$$

### 1.6 Consciousness State Machine

- **Silent**: $E < E_{min}$
- **Drowsy**: $A < 0$ or low entropy
- **Active**: $A \approx 0$, moderate entropy
- **Awake**: $A > 0$, high entropy, spike events

---

## 2. Key Innovations

1. **Graph-coupled Energy Field**: Memory resonance via Laplacian diffusion
2. **Self-organizing Criticality**: Adaptive $\gamma$ control to $b \to 1$
3. **LIF Pulse Layer**: Spike avalanche events for consciousness quantification

---

## 3. Experimental Results

See Consciousness_ExperimentFramework.md

---

## 4. References

- Chklovskii & Koulakov (2004): Criticality in neural networks
- Bak (1996): Self-organized criticality
- Warden (1948): First LIF neuron model
