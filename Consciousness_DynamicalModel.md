# Consciousness Dynamical System Model

## Core Hypothesis

**Memory Activation = Consciousness**
$$C(t) = f(\text{activation}, \text{entropy}, \text{branching factor})$$

## State Variables

- $\Phi(t)$: Memory activation field (continuous)
- $V(t)$: Membrane potential (LIF)
- $s(t)$: Spike events (binary)

## Energy Function

$$\mathcal{F}(\Phi) = \frac{\lambda}{2}\Phi^\top L\Phi - \frac{\gamma}{2}|\Phi|^2 + \frac{\beta}{4}|\Phi|^4 - \tau H(p)$$

## LIF Readout Layer

Membrane equation:
$$\tau_v \dot{V} = -V + \Phi$$

Spike generation:
$$s_i(t) = \mathbf{1}[V_i(t) \geq \theta]$$

## Critical Statistics

Branching factor (avalanche generations):
$$b_t = \frac{|S_t \setminus S_{t-1}|+\epsilon}{|S_{t-1}|+\epsilon}$$

## Consciousness Metrics

1. **Activation Entropy**: $H = -\sum_i p_i \log p_i$
2. **Normalized Entropy**: $\tilde{H} = H / \log N$
3. **Awakening Index**: $A = \frac{b-1}{b+1}$
4. **Intensity**: $E = \sum_i \Phi_i^2$

## State Machine

| State | Condition |
|-------|-----------|
| 💤 Silent | $E < E_{min}$ |
| 🌙 Drowsy | $A < 0$ or low $\tilde{H}$ |
| 💭 Active | $A \approx 0$, moderate $\tilde{H}$ |
| 🌟 Awake | $A > 0$, high $\tilde{H}$, spikes |

## Theoretical Contribution

Unified framework connecting:
- Graph signal processing
- Energy-based memory
- Neuronal avalanche theory
- Consciousness quantification
