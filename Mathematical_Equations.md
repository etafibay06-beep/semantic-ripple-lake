# 语义涟漪湖数学方程组 - 比赛投稿版

> 日期：2026-03-01
> 来源：ChatGPT
> 版本：v3.2/v3.3

---

## 1) 状态变量定义

### 1.1 记忆激活与动量状态
- 激活状态向量: $\boldsymbol{\Phi}_t \in \mathbb{R}_{\ge 0}^N$
- 动量/速度状态向量: $\mathbf{v}_t \in \mathbb{R}^N$

### 1.2 记忆元数据与输入
每条记忆 $i$ 具有：
- 创建时间 $t^{(0)}_i$
- 历史激活次数 $n_i$
- 质量系数 $q_i > 0$
- 嵌入向量 $\mathbf{e}_i \in \mathbb{R}^d$（已L2 normalize）

查询 $x$ 的嵌入为 $\mathbf{e}(x) \in \mathbb{R}^d$。

### 1.3 单次查询的外部驱动（输入）

相似度（余弦）：
$$s_i(x) = \mathbf{e}_i^\top \mathbf{e}(x)$$

时间衰减因子（按天）：
$$\Delta t_i = \frac{t - t_i^{(0)}}{86400}, \quad a_i(t) = \exp(-\kappa \Delta t_i), \ \kappa = 0.1$$

掌握度因子：
$$m_i(t) = 1 + \frac{1}{2}\left(1 - \exp\Big(-\eta \frac{n_i}{\max(1, 3\Delta t_i)}\Big)\right), \ \eta = 0.3$$

基础输入（外力）：
$$u_i(t;x) = s_i(x) \cdot a_i(t) \cdot m_i(t) \cdot q_i$$

### 1.4 Top-k 更新集合
$$\mathcal{K}(t) = \operatorname{TopK}\{u_i(t;x)\}_{i=1}^N, \quad |\mathcal{K}| = k$$

---

## 2) 能量函数 $\mathcal{F}$

### 2.1 基础版能量（与实现一致）

定义温度 $T > 0$，softmax 概率：
$$p_i(\boldsymbol{\Phi}) = \frac{\exp(\Phi_i / T)}{\sum_{j=1}^N \exp(\Phi_j / T)}$$

熵：
$$H(\mathbf{p}) = -\sum_{i=1}^N p_i \log p_i$$

能量：
$$\boxed{\mathcal{F}(\boldsymbol{\Phi}) = \frac{1}{2}(\lambda - \gamma)|\boldsymbol{\Phi}|^2 + \frac{\beta}{4}\sum_{i=1}^N \Phi_i^4 - \tau H(\mathbf{p}(\boldsymbol{\Phi}))}$$

### 2.2 图扩展版（v3.1继承）

引入 kNN 图拉普拉斯 $L \succeq 0$：
$$\boxed{\mathcal{F}(\boldsymbol{\Phi}) = \frac{\lambda}{2}\boldsymbol{\Phi}^\top L\boldsymbol{\Phi} - \frac{\gamma}{2}|\boldsymbol{\Phi}|^2 + \frac{\beta}{4}\sum_{i=1}^N \Phi_i^4 - \tau H(\mathbf{p}(\boldsymbol{\Phi}))}$$

---

## 3) 梯度 $\nabla \mathcal{F}$

### 3.1 基础项梯度

$$\nabla_{\Phi_i}\mathcal{F} = (\lambda - \gamma)\Phi_i + \beta \Phi_i^3 - \tau \frac{\partial H}{\partial \Phi_i}$$

### 3.2 熵项梯度（与softmax一致）

令 $z_i = \Phi_i / T$：
$$\frac{\partial H}{\partial z_i} = -p_i(\log p_i + H)$$

$$\boxed{\frac{\partial H}{\partial \Phi_i} = \frac{1}{T} \cdot p_i(\log p_i + H)}$$

熵在能量中的梯度贡献：
$$\boxed{-\tau \frac{\partial H}{\partial \Phi_i} = \frac{\tau}{T} \cdot p_i(\log p_i + H)}$$

### 3.3 图扩展版梯度

$$\boxed{\nabla \mathcal{F} = \lambda L\boldsymbol{\Phi} - \gamma\boldsymbol{\Phi} + \beta \boldsymbol{\Phi}^{\odot 3} - \tau \nabla H}$$

---

## 4) 动量更新方程

### 4.1 连续时间形式
$$\dot{\boldsymbol{\Phi}} = \mathbf{v}, \quad \dot{\mathbf{v}} = -\alpha \mathbf{v} - \nabla \mathcal{F}(\boldsymbol{\Phi}) + \mathbf{u}(t;x)$$

### 4.2 离散时间形式

设步长（学习率）$\eta > 0$，动量系数 $\mu \in [0, 1)$。对 $i \in \mathcal{K}(t)$：

$$\boxed{v_{i,t+1} = \mu v_{i,t} - \eta \nabla_{\Phi_i}\mathcal{F}(\boldsymbol{\Phi}_t) + \eta u_i(t;x)}$$

$$\boxed{\Phi_{i,t+1} = \Pi_{[0,\infty)}(\Phi_{i,t} + v_{i,t+1})}$$

### 4.3 饱和（防爆炸）

推荐可导软饱和：
$$\boxed{\Phi_{i,t+1} \leftarrow \Phi_{\max} \tanh\Big(\Phi_{i,t+1} / \Phi_{\max}\Big)}$$

### 4.4 非top-k节点
$$\Phi_{i,t+1} = \Phi_{i,t}, \quad v_{i,t+1} = v_{i,t}$$

---

## 5) 临界条件（创造性涌现量化）

### 5.1 激活事件与新增激活

每次查询记录：
- 时间 $t$
- 激活集合 $A_t \subseteq \{1,\dots,N\}$（即 `activated_ids`）
- 激活数 $c_t = |A_t|$

定义新增激活数：
$$\Delta c_t = |A_t \setminus A_{t-1}|$$

### 5.2 分支因子（branching factor）

$$\boxed{b_t = \frac{\Delta c_t + \varepsilon}{|A_{t-1}| + \varepsilon}}$$

其中 $\varepsilon$ 是平滑常数（如 $10^{-6}$）。

指数滑动平均（EMA）：
$$\boxed{\hat b_t = (1 - \rho)\hat b_{t-1} + \rho b_t}$$

### 5.3 临界带判据

给定阈值 $\epsilon_b > 0$：

$$\boxed{|\hat b_t - 1| \le \epsilon_b}$$

解释：
- $\hat b_t < 1$：亚临界（回忆保守）
- $\hat b_t \approx 1$：临界（组合/创造性上升）
- $\hat b_t > 1$：超临界（需回退/降参）

---

## 6) 离散化方案（能量监控 + backtracking）

### 6.1 单步能量差
$$\Delta \mathcal{F} = \mathcal{F}(\boldsymbol{\Phi}_{t+1}) - \mathcal{F}(\boldsymbol{\Phi}_t)$$

### 6.2 Backtracking规则

设初始步长 $\eta_0$，缩放因子 $\xi \in (0, 1)$，最大回退次数 $M$，最小步长 $\eta_{\min}$。

算法：
1. 令 $\eta \leftarrow \eta_0$，保存状态快照
2. 用步长 $\eta$ 执行一次top-k动量更新
3. 若 $\mathcal{F}(\boldsymbol{\Phi}^*) \le \mathcal{F}(\boldsymbol{\Phi}_t)$，接受
4. 否则回滚，令 $\eta \leftarrow \xi\eta$，重复直到满足或达到 $M$ 次

$$\boxed{\text{Accept if } \mathcal{F}(\boldsymbol{\Phi}_{t+1}) \le \mathcal{F}(\boldsymbol{\Phi}_t), \text{ else } \eta \leftarrow \xi\eta \text{ and retry.}}$$

### 6.3 回退时冻结动量

第一次失败后令 $\mu \leftarrow 0$（退化为纯梯度下降）：
$$v_{i,t+1} = -\eta \nabla_{\Phi_i}\mathcal{F} + \eta u_i$$

---

## 算法框图

> **Semantic Ripple Lake (v3.2)**

Given query $x$ at time $t$:
1. Compute $u_i(t;x) = s_i(x)a_i(t)m_i(t)q_i$ for all $i$
2. Select $\mathcal{K} = \text{TopK}(u_i)$
3. Backtracking over step size $\eta$:
   - Save snapshot $(\boldsymbol{\Phi}_t, \mathbf{v}_t)$
   - Update $v_i \leftarrow \mu v_i - \eta \nabla_{\Phi_i}\mathcal{F} + \eta u_i$ for $i \in \mathcal{K}$
   - Update $\Phi_i \leftarrow \text{sat}(\max(0, \Phi_i + v_i))$
   - Accept if $\mathcal{F}$ non-increasing, else rollback and shrink $\eta$
4. Record activated set $A_t = \{i: \Phi_{i,t} \ge \theta\}$
5. Compute branching factor $b_t$ and EMA $\hat b_t$ for criticality monitoring.

---

## 创造性涌现描述

> 当系统处于临界带 $(|\hat b - 1| \le \epsilon_b)$ 时，激活级联既不会快速熄灭也不会爆炸扩散，top-k稀疏更新在语义空间中形成"边缘稳定"的组合路径，从而提升跨记忆关联与新组合的生成概率；同时能量回退机制保证整体动力学稳定。

---

## 下一步扩展

- kNN图拉普拉斯 + IMEX求解
- 自适应步长
- 自动临界控制

---

*甲子语义涟漪湖数学方程组 - 2026-03-01*
