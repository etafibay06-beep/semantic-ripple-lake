# Mathematical Equations - Complete Collection

## 1. Graph Construction

kNN Adjacency:
$$W_{ij} = \exp\left(\frac{\text{sim}(v_i, v_j) - 1}{\sigma}\right)$$

Laplacian:
$$L = D - W, \quad D_{ii} = \sum_j W_{ij}$$

## 2. Energy Function

$$\mathcal{F}(\Phi) = \frac{\lambda}{2}\Phi^\top L\Phi - \frac{\gamma}{2}|\Phi|^2 + \frac{\beta}{4}|\Phi|^4 - \tau H(p)$$

where:
$$p_i = \frac{\Phi_i^2}{\sum_j \Phi_j^2 + \epsilon}$$

## 3. IMEX Scheme

Implicit for diffusion, explicit for reaction:
$$(I + \eta\lambda L)\Phi^{n+1} = \Phi^n + \eta(b - \gamma\Phi^n + \beta(\Phi^n)^3)$$

Stability condition:
$$\Delta t \leq \frac{2}{\lambda_{max}(L)}$$

## 4. Branching Factor

$$b_t = \frac{|S_t \setminus S_{t-1}| + \epsilon}{|S_{t-1}| + \epsilon}$$

EMA smoothing:
$$\hat{b}_t = (1-\rho)\hat{b}_{t-1} + \rho b_t$$

## 5. Critical Control

$$\gamma_{t+1} = \Pi_{[\gamma_{min},\gamma_{max}]}(\gamma_t - k(\hat{b}_t - 1))$$

## 6. Awakening Index

$$A = \frac{\hat{b} - 1}{\hat{b} + 1}$$

## 7. Entropy

$$H = -\sum_i p_i \log p_i$$
$$\tilde{H} = \frac{H}{\log N}$$

## 8. LIF Membrane Potential

$$\tau_v \dot{V} = -V + \Phi$$

Spike:
$$s_i = \mathbf{1}[V_i \geq \theta]$$
