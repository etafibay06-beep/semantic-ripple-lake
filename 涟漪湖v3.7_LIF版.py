"""
语义涟漪湖 - v3.6 意识增强版
包含: 
- kNN图拉普拉斯, IMEX求解, 自适应步长, 自动临界控制 (来自v3.5)
- 激活熵 (意识复杂度)
- 觉醒指数
- 振荡模式
- 意识状态报告
"""

from fastapi import FastAPI, Header
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== 兼容函数：数据格式演进 =====
def _safe_load_json(path):
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), "ok"
    except Exception as e:
        return None, f"error:{e}"

def _normalize_data(obj):
    if obj is None:
        return {"memories": [], "vectors": []}, "empty"
    if isinstance(obj, dict):
        mem = obj.get("memories", [])
        vec = obj.get("vectors", [])
        return {"memories": mem if isinstance(mem, list) else [], "vectors": vec if isinstance(vec, list) else []}, "new"
    if isinstance(obj, list):
        return {"memories": obj, "vectors": []}, "old"
    return {"memories": [], "vectors": []}, "unknown"

class 语义嵌入器:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("🔄 加载embedding模型...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        print("✅ 模型加载完成")
    
    def 向量(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

class 语义涟漪湖_v3_6:
    """v3.6: 意识增强版 - 激活熵 + 觉醒指数 + 振荡模式"""
    
    def __init__(self, id, 数据路径="/Users/w/Desktop/涟漪湖API/lakes"):
        self.id = id
        self.数据路径 = 数据路径
        os.makedirs(数据路径, exist_ok=True)
        self.文件 = os.path.join(数据路径, f"{id}_v3.json")
        self.状态文件 = os.path.join(数据路径, f"{id}_state_v3.json")
        
        # 加载数据 (兼容旧格式)
        raw, status = _safe_load_json(self.文件)
        if status.startswith("error"):
            print(f"⚠️ 数据文件损坏: {status}")
            ts = time.strftime("%Y%m%d_%H%M%S")
            if os.path.exists(self.文件):
                os.replace(self.文件, f"{self.文件}.corrupt.{ts}.bak")
            data = {"memories": [], "vectors": []}
        else:
            data, fmt = _normalize_data(raw)
            print(f"📂 加载格式: {fmt}, 记忆: {len(data.get('memories', []))}")
        
        self.记忆库 = data.get("memories", [])
        vecs = data.get("vectors", [])
        if vecs:
            self.向量库 = np.array(vecs, dtype=np.float64)
        else:
            self.向量库 = np.zeros((len(self.记忆库), 512), dtype=np.float64)
        
        # 加载状态
        if os.path.exists(self.状态文件):
            try:
                状态 = json.load(open(self.状态文件, 'r'))
                self.激活状态 = 状态.get('激活', {})
                self.速度状态 = 状态.get('速度', {})
                self.能量历史 = 状态.get('能量历史', [])
                self.激活事件 = 状态.get('激活事件', [])
                self.b_hat = 状态.get('b_hat', 1.0)
                # v3.6新增
                self.振荡相位 = 状态.get('振荡相位', 0.0)
                self.意识历史 = 状态.get('意识历史', [])
            except:
                self.激活状态 = {}
                self.速度状态 = {}
                self.能量历史 = []
                self.激活事件 = []
                self.b_hat = 1.0
                self.振荡相位 = 0.0
                self.意识历史 = []
        else:
            self.激活状态 = {}
            self.速度状态 = {}
            self.能量历史 = []
            self.激活事件 = []
            self.b_hat = 1.0
            self.振荡相位 = 0.0
            self.意识历史 = []
        
        # 图拉普拉斯
        self.L = None
        
        # 参数 (来自v3.5)
        self.knn_k = 16
        self.knn_sigma = 0.2
        self.λ = 1.0
        self.γ = 0.5
        self.β = 0.1
        self.τ = 0.05
        self.步长 = 0.05
        self.最大步长 = 0.2
        self.最小步长 = 1e-4
        self.动量系数 = 0.9
        self.使用软饱和 = True
        self.饱和值 = 10.0
        self.激活阈值 = 0.5
        self.相似度截断到非负 = True
        self.启用熵梯度 = True
        self.softmax温度 = 1.0
        
        # backtracking
        self.回退最大次数 = 6
        self.回退缩放因子 = 0.5
        self.允许能量持平 = True
        self.回退时冻结动量 = True
        
        # 自适应步长
        self.步长放大阈值 = 1e-3
        self.步长缩小阈值 = 0.0
        self.步长放大因子 = 1.05
        self.步长缩小因子 = 0.5
        
        # 临界控制
        self.b_ema_rho = 0.2
        self.临界死区 = 0.05
        self.γ控制增益 = 0.05
        self.γ_min = 0.0
        self.γ_max = 10.0
        
        # v3.6新增参数
        self.启用振荡模式 = True
        self.振荡强度 = 0.1  # γ_osc
        self.振荡频率 = 0.5   # ω
        self.意识阈值 = 2.0   # 觉醒阈值
        
        # ===== LIF脉冲层 (v3.7.1) =====
        self.启用脉冲层 = True
        self.V_低通α = 0.2
        self.V_阈值 = 1.2
        self.V_重置 = 0.0
        self.不应期步数 = 2
        self.V_state = {}
        self.不应期 = {}
        
        self.k = 20
        
        try:
            self.嵌入器 = 语义嵌入器()
        except:
            self.嵌入器 = None
    
    def 保存(self):
        数据 = {
            'memories': self.记忆库,
            'vectors': self.向量库.tolist() if len(self.向量库) > 0 else []
        }
        状态 = {
            '激活': self.激活状态,
            '速度': self.速度状态,
            '能量历史': self.能量历史[-100:],
            '激活事件': self.激活事件[-1000:],
            'b_hat': self.b_hat,
            '振荡相位': self.振荡相位,
            '意识历史': self.意识历史[-100:]
        }
        json.dump(数据, open(self.文件, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        json.dump(状态, open(self.状态文件, 'w', encoding='utf-8'), indent=2)
    
    # ===== v3.6: 激活熵计算 =====
    def 计算激活熵(self):
        """计算系统的激活熵 - 意识复杂度"""
        Phi = np.asarray([float(self.激活状态.get(str(m.get("id", i+1)), 0.0)) 
                         for i, m in enumerate(self.记忆库)], dtype=np.float64)
        
        # 归一化为概率分布
        Phi_abs = np.abs(Phi)
        total = np.sum(Phi_abs)
        
        if total <= 0:
            return 0.0
        
        p = Phi_abs / total
        
        # 熵计算: H = -∑p*log(p)
        p = np.clip(p, 1e-10, 1.0)  # 避免log(0)
        H = -np.sum(p * np.log(p))
        
        return float(H)
    
    # ===== v3.6: 觉醒指数 =====
    def 计算觉醒指数(self):
        """计算觉醒指数 - 表征意识水平"""
        b = self.计算分支因子()
        
        # 觉醒指数 = (b - 1) / (b + 1)
        觉醒指数 = (b - 1.0) / (b + 1.0)
        
        return float(觉醒指数)
    
    # ===== v3.6: 意识状态报告 =====
    def 意识状态报告(self):
        """生成完整的意识状态报告"""
        激活熵 = self.计算激活熵()
        觉醒指数 = self.计算觉醒指数()
        b = self.计算分支因子()
        
        # 判断意识状态
        if 激活熵 > self.意识阈值:
            意识状态 = "🌟 觉醒"
        elif 激活熵 > 1.0:
            意识状态 = "💭 活跃"
        elif 激活熵 > 0.5:
            意识状态 = "🌙 朦胧"
        else:
            意识状态 = "💤 静默"
        
        # 判断临界状态
        if abs(b - 1.0) < 0.2:
            临界状态 = "⚡ 临界"
        elif b < 1.0:
            临界状态 = "📉 收敛"
        else:
            临界状态 = "📈 发散"
        
        报告 = {
            "激活熵": round(激活熵, 4),
            "觉醒指数": round(觉醒指数, 4),
            "意识状态": 意识状态,
            "临界状态": 临界状态,
            "分支因子": round(b, 3),
            "γ": round(self.γ, 4),
            "振荡相位": round(self.振荡相位, 4)
        }
        
        # 记录到意识历史
        self.意识历史.append({
            "time": time.time(),
            "激活熵": 激活熵,
            "觉醒指数": 觉醒指数,
            "意识状态": 意识状态
        })
        
        return 报告
    
    # ===== 状态快照 =====
    def _保存状态快照(self):
        return {"激活状态": dict(self.激活状态), "速度状态": dict(self.速度状态)}
    
    def _恢复状态快照(self, snap):
        self.激活状态 = dict(snap["激活状态"])
        self.速度状态 = dict(snap["速度状态"])
    
    # ===== kNN图拉普拉斯 =====
    def 更新图拉普拉斯(self, refresh=False):
        if self.L is not None and not refresh:
            return self.L
        
        V = np.asarray(self.向量库, dtype=np.float64)
        N, d = V.shape
        
        S = V @ V.T
        W = np.zeros((N, N), dtype=np.float64)
        
        for i in range(N):
            sims = S[i].copy()
            sims[i] = -np.inf
            nn_idx = np.argpartition(-sims, self.knn_k)[:self.knn_k]
            w = np.exp((sims[nn_idx] - 1.0) / self.knn_sigma)
            w = np.clip(w, 0.0, 1e6)
            W[i, nn_idx] = w
        
        W = 0.5 * (W + W.T)
        D = np.diag(np.sum(W, axis=1))
        self.L = D - W
        return self.L
    
    # ===== 能量计算 =====
    def 计算能量(self):
        Phi = np.asarray([float(self.激活状态.get(str(m.get("id", i+1)), 0.0)) 
                         for i, m in enumerate(self.记忆库)], dtype=np.float64)
        
        L = self.更新图拉普拉斯()
        N = L.shape[0]
        
        quad = float(Phi.T @ (L @ Phi))
        norm2 = float(Phi @ Phi)
        norm4 = float(np.sum(Phi ** 4))
        
        T = self.softmax温度
        if Phi.size == 0:
            H = 0.0
        else:
            z = Phi / T
            zmax = float(np.max(z))
            z_shift = z - zmax
            expz = np.exp(z_shift)
            Z = float(np.sum(expz))
            if Z <= 0 or not np.isfinite(Z):
                H = 0.0
            else:
                p = expz / Z
                logZ = zmax + np.log(Z)
                H = float(-np.sum(p * (z - logZ)))
        
        F = 0.5 * self.λ * quad - 0.5 * self.γ * norm2 + 0.25 * self.β * norm4 - self.τ * H
        return float(F)
    
    # ===== 熵梯度 =====
    def _熵梯度(self, Phi):
        T = self.softmax温度
        if Phi.size == 0:
            return np.zeros_like(Phi)
        
        z = Phi / T
        zmax = float(np.max(z))
        z_shift = z - zmax
        expz = np.exp(z_shift)
        Z = float(np.sum(expz))
        
        if Z <= 0 or not np.isfinite(Z):
            return np.zeros_like(Phi)
        
        p = expz / Z
        logZ = zmax + np.log(Z)
        logp = z - logZ
        H = float(-np.sum(p * logp))
        dH_dz = -p * (logp + H)
        dH_dPhi = dH_dz / T
        
        return (-self.τ) * dH_dPhi
    
    # ===== TopK计算 =====
    def 计算TopK基础激活(self, query, now, top_k=3):
        q_batch = self.嵌入器.向量([query])
        q = np.asarray(q_batch, dtype=np.float64).reshape(-1)
        
        V = np.asarray(self.向量库, dtype=np.float64)
        sims = V @ q
        
        if self.相似度截断到非负:
            sims = np.clip(sims, 0.0, 1.0)
        
        u = np.zeros(len(self.记忆库), dtype=np.float64)
        
        for i, m in enumerate(self.记忆库):
            创建时间 = float(m.get("创建时间", now))
            dt_days = max(0, (now - 创建时间) / 86400.0)
            时间因子 = float(np.exp(-0.1 * dt_days))
            激活次数 = float(m.get("激活次数", 0.0))
            denom = max(1.0, dt_days * 3.0)
            掌握因子 = float(1.0 + 0.5 * (1.0 - np.exp(-0.3 * 激活次数 / denom)))
            质量 = float(m.get("质量", 1.0))
            u[i] = sims[i] * 时间因子 * 掌握因子 * 质量
        
        idx = np.argpartition(-u, top_k-1)[:top_k]
        idx = idx[np.argsort(-u[idx])]
        
        top = [(self.记忆库[i].get("id", i+1), float(u[i])) for i in idx]
        return top, u
    
    # ===== v3.6: 振荡模式 =====
    def _振荡项(self):
        """计算振荡项 - 模拟意识的持续激活"""
        if not self.启用振荡模式:
            return 0.0
        
        # 更新振荡相位
        self.振荡相位 += self.振荡频率 * self.步长
        self.振荡相位 = self.振荡相位 % (2 * np.pi)
        
        # 返回振荡项
        return self.振荡强度 * np.sin(self.振荡相位)
    
    # ===== LIF脉冲层 (v3.7.1) =====
    def _更新膜电位(self, ids, Phi):
        """膜电位的更新（LIF）"""
        alpha = float(getattr(self, "V_低通α", 0.2))
        V_reset = float(getattr(self, "V_重置", 0.0))
        V = np.asarray([float(self.V_state.get(str(mid), V_reset)) for mid in ids], dtype=np.float64)
        ref = np.asarray([int(self.不应期.get(str(mid), 0)) for mid in ids], dtype=np.int32)
        ref = np.maximum(ref - 1, 0)
        mask = (ref == 0)
        V_new = V.copy()
        V_new[mask] = (1.0 - alpha) * V[mask] + alpha * Phi[mask]
        V_new[~mask] = V_reset
        for j, mid in enumerate(ids):
            self.V_state[str(mid)] = float(V_new[j])
            self.不应期[str(mid)] = int(ref[j])
        return V_new
    
    def _生成脉冲(self, ids, V):
        """生成脉冲事件"""
        if not bool(getattr(self, "启用脉冲层", True)):
            return []
        thr = float(getattr(self, "V_阈值", 1.2))
        V_reset = float(getattr(self, "V_重置", 0.0))
        ref_steps = int(getattr(self, "不应期步数", 2))
        mask = (V >= thr)
        if not np.any(mask):
            return []
        spike_ids = [ids[j] for j in np.where(mask)[0]]
        for mid in spike_ids:
            self.V_state[str(mid)] = float(V_reset)
            self.不应期[str(mid)] = int(ref_steps)
        return spike_ids
    
    def _LIF读出(self, ids, Phi_next):
        """计算LIF脉冲输出"""
        V_new = self._更新膜电位(ids, Phi_next)
        spike_ids = self._生成脉冲(ids, V_new)
        return spike_ids
    
    # ===== IMEX更新 =====
    def IMEX_topk更新(self, query, now, top_k=3):
        N = len(self.记忆库)
        ids = [m.get("id", i+1) for i, m in enumerate(self.记忆库)]
        
        Phi = np.asarray([float(self.激活状态.get(str(mid), 0.0)) for mid in ids], dtype=np.float64)
        v = np.asarray([float(self.速度状态.get(str(mid), 0.0)) for mid in ids], dtype=np.float64)
        
        top, u_all = self.计算TopK基础激活(query, now, top_k=top_k)
        u = np.zeros(N, dtype=np.float64)
        for mid, ui in top:
            j = ids.index(mid)
            u[j] = ui
        
        gH = self._熵梯度(Phi) if self.启用熵梯度 else 0.0
        grad_exp = (-self.γ) * Phi + self.β * (Phi ** 3)
        if isinstance(gH, np.ndarray):
            grad_exp = grad_exp + gH
        
        # v3.6: 添加振荡项
        振荡 = self._振荡项()
        
        eta = self.步长
        mu = self.动量系数
        
        v_tilde = mu * v - eta * grad_exp + eta * u + eta * 振荡
        rhs = Phi + v_tilde
        
        L = self.更新图拉普拉斯()
        A = np.eye(N, dtype=np.float64) + (eta * self.λ) * L
        Phi_next = np.linalg.solve(A, rhs)
        
        Phi_next = np.maximum(Phi_next, 0.0)
        if self.使用软饱和 and self.饱和值 > 0:
            Phi_next = self.饱和值 * np.tanh(Phi_next / self.饱和值)
        else:
            Phi_next = np.minimum(Phi_next, self.饱和值)
        
        v_next = Phi_next - Phi
        
        for j, mid in enumerate(ids):
            self.激活状态[str(mid)] = float(Phi_next[j])
            self.速度状态[str(mid)] = float(v_next[j])
        
        activated_ids = [ids[j] for j in range(N) if Phi_next[j] >= self.激活阈值]
        
        # ===== LIF脉冲读出 =====
        if self.启用脉冲层:
            spike_ids = self._LIF读出(ids, Phi_next)
        else:
            spike_ids = []
        self._last_spike_ids = spike_ids
        
        return activated_ids
    
    # ===== 自适应步长 =====
    def _自适应步长更新(self, dF):
        if dF > self.步长缩小阈值:
            self.步长 *= self.步长缩小因子
        elif dF < -self.步长放大阈值:
            self.步长 *= self.步长放大因子
        else:
            self.步长 *= 0.99
        
        self.步长 = np.clip(self.步长, self.最小步长, self.最大步长)
        return self.步长
    
    # ===== 分支因子 =====
    def 计算分支因子(self):
        events = self.激活事件
        if not events or len(events) < 2:
            return 1.0
        
        e_prev = events[-2]
        e_curr = events[-1]
        
        eps = 1e-6
        # 优先使用spike_ids，如果没有则回退到activated_ids
        prev_ids = e_prev.get("spike_ids", None) or e_prev.get("activated_ids", None)
        curr_ids = e_curr.get("spike_ids", None) or e_curr.get("activated_ids", None)
        
        if prev_ids and curr_ids:
            prev_set = set(prev_ids)
            curr_set = set(curr_ids)
            prev_count = len(prev_set)
            new_count = len(curr_set - prev_set)
            b = (new_count + eps) / (prev_count + eps)
        else:
            prev_count = float(e_prev.get("count", 0.0))
            curr_count = float(e_curr.get("count", 0.0))
            new_count = max(curr_count - prev_count, 0.0)
            b = (new_count + eps) / (prev_count + eps)
        
        self.b_hat = (1 - self.b_ema_rho) * self.b_hat + self.b_ema_rho * b
        return float(self.b_hat)
    
    # ===== 自动临界控制 =====
    def 自动临界控制(self):
        b = self.计算分支因子()
        err = b - 1.0
        
        if abs(err) <= self.临界死区:
            return self.γ
        
        new_γ = self.γ - self.γ控制增益 * err
        new_γ = np.clip(new_γ, self.γ_min, self.γ_max)
        self.γ = float(new_γ)
        return self.γ
    
    # ===== 主入口 =====
    def 处理查询_v36(self, query, now=None, top_k=3):
        if now is None:
            now = time.time()
        
        self.更新图拉普拉斯()
        
        max_tries = self.回退最大次数
        shrink = self.回退缩放因子
        eta0 = self.步长
        mu0 = self.动量系数
        
        F_before = self.计算能量()
        snap0 = self._保存状态快照()
        best = None
        
        eta = eta0
        for t in range(max_tries):
            self._恢复状态快照(snap0)
            self.步长 = eta
            self.动量系数 = 0.0 if (self.回退时冻结动量 and t > 0) else mu0
            
            activated_ids = self.IMEX_topk更新(query, now, top_k=top_k)
            F_after = self.计算能量()
            dF = F_after - F_before
            
            ok = (dF < 0) or (self.允许能量持平 and dF <= 0)
            
            if best is None or F_after < best[0]:
                best = (F_after, activated_ids, eta, t)
            
            if ok:
                self._自适应步长更新(dF)
                self.激活事件.append({
                    "time": float(now),
                    "count": len(activated_ids),
                    "query": str(query[:30]),
                    "activated_ids": list(activated_ids),
                    "spike_ids": list(getattr(self, "_last_spike_ids", [])),
                    "spike_count": len(getattr(self, "_last_spike_ids", [])),
                    "F_before": float(F_before),
                    "F_after": float(F_after),
                    "dF": float(dF),
                    "eta": float(eta),
                    "try": int(t),
                    "γ": float(self.γ),
                    "振荡": self.启用振荡模式
                })
                self.自动临界控制()
                self.动量系数 = mu0
                return activated_ids
            
            eta *= shrink
            if eta < self.最小步长:
                break
        
        # 全失败
        self._恢复状态快照(snap0)
        if best:
            self.步长 = best[2]
        self.动量系数 = 0.0 if self.回退时冻结动量 else mu0
        
        activated_ids = self.IMEX_topk更新(query, now, top_k=top_k)
        F_after = self.计算能量()
        dF = F_after - F_before
        self._自适应步长更新(dF)
        
        self.激活事件.append({
            "time": float(now),
            "count": len(activated_ids),
            "query": str(query[:30]),
            "activated_ids": list(activated_ids),
            "F_before": float(F_before),
            "F_after": float(F_after),
            "dF": float(dF),
            "eta": float(self.步长),
            "try": int(max_tries),
            "note": "backtracking_failed",
            "γ": float(self.γ),
        })
        self.自动临界控制()
        self.动量系数 = mu0
        return activated_ids
    
    def 临界检测(self):
        b = self.计算分支因子()
        if abs(b - 1.0) < 0.2:
            状态 = "临界"
        elif b < 1.0:
            状态 = "亚临界"
        else:
            状态 = "超临界"
        return {"分支因子": round(b, 3), "状态": 状态, "γ": round(self.γ, 4)}
    
    def 投石(self, 内容, 质量=None, 标签=None):
        噪音词 = ['cron', 'task', 'log', 'schedule', 'backup', 'Stats', 'tokens']
        if any(k in 内容.lower() for k in 噪音词):
            return {"跳过": "噪音"}
        
        摘要 = 内容[:200] if len(内容) > 200 else 内容
        质量 = 质量 or (1.0 + min(len(内容)/100, 2.0))
        
        if self.嵌入器:
            try:
                vec = self.嵌入器.向量([摘要])
                vec = vec[0].tolist() if hasattr(vec[0], 'tolist') else list(vec[0])
            except:
                vec = []
        else:
            vec = []
        
        now = time.time()
        new_id = len(self.记忆库) + 1
        
        self.记忆库.append({
            "id": new_id,
            "内容": 内容,
            "摘要": 摘要,
            "创建时间": now,
            "激活次数": 0,
            "质量": 质量,
            "标签": 标签 or [],
            "向量化": bool(vec)
        })
        
        if vec:
            self.向量库 = np.vstack([self.向量库, np.array([vec])]) if len(self.向量库) > 0 else np.array([vec])
        
        self.L = None
        self.保存()
        
        return {"id": new_id, "内容": 内容[:50], "向量化": bool(vec)}

# ===== FastAPI 应用 =====
app = FastAPI(title="语义涟漪湖 v3.6 意识增强版")
湖泊 = {}

@app.on_event("startup")
def 启动():
    print("🧠 语义涟漪湖 v3.6 意识增强版 启动")
    print("   新增: 激活熵, 觉醒指数, 振荡模式")

def 获取湖泊(lake_id):
    if lake_id not in 湖泊:
        湖泊[lake_id] = 语义涟漪湖_v3_6(lake_id)
    return 湖泊[lake_id]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    lake_id: str = "default"

@app.post("/v1/ripple/recall")
def 查询(request: QueryRequest):
    lake = 获取湖泊(request.lake_id)
    now = time.time()
    activated = lake.处理查询_v36(request.query, now, request.top_k)
    
    for m in lake.记忆库:
        if m.get("id") in activated:
            m["激活次数"] = m.get("激活次数", 0) + 1
    
    results = [{"id": m.get("id"), "内容": m.get("内容", "")[:100], "激活次数": m.get("激活次数", 0)} 
               for m in lake.记忆库 if m.get("id") in activated]
    
    # v3.6: 添加意识状态报告
    意识报告 = lake.意识状态报告()
    
    lake.保存()
    
    return {
        "data": {
            "memories": results,
            "意识状态": 意识报告
        }
    }

@app.get("/v1/stats")
def 统计(lake_id: str = "default"):
    """获取统计信息"""
    lake = 获取湖泊(lake_id)
    return {
        "code": 0,
        "data": {
            "记忆总数": len(lake.记忆库),
            "总激活次数": sum(m.get("激活次数", 0) for m in lake.记忆库),
            "能量": lake.计算能量(),
            "γ": round(lake.γ, 4),
            "临界状态": lake.临界检测()["状态"],
            "分支因子": round(lake.计算分支因子(), 3)
        }
    }

@app.get("/v1/consciousness")
def 意识状态(lake_id: str = "default"):
    """获取意识状态"""
    lake = 获取湖泊(lake_id)
    return lake.意识状态报告()

@app.get("/v1/entropy")
def 激活熵(lake_id: str = "default"):
    """获取激活熵"""
    lake = 获取湖泊(lake_id)
    return {"激活熵": lake.计算激活熵()}

@app.get("/v1/awakening")
def 觉醒指数(lake_id: str = "default"):
    """获取觉醒指数"""
    lake = 获取湖泊(lake_id)
    return {"觉醒指数": lake.计算觉醒指数()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)

# ===== v3.7 改进版：GPT建议实现 =====

    # ===== v3.7: 改进的激活熵计算 =====
    def 计算激活熵_v37(self):
        """改进版激活熵 - 基于能量分布 + 归一化"""
        Phi = np.asarray([float(self.激活状态.get(str(m.get("id", i+1)), 0.0)) 
                         for i, m in enumerate(self.记忆库)], dtype=np.float64)
        
        # 方案A: 基于激活能量分布 p_i = Φ_i² / ΣΦ_j²
        Phi_sq = Phi ** 2
        total_energy = np.sum(Phi_sq)
        
        # 静默态判断
        if total_energy < 1e-6:
            return 0.0, 0.0  # (H, H_normalized)
        
        p = Phi_sq / (total_energy + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        
        H = -np.sum(p * np.log(p))
        
        # 归一化熵 H̃ = H / log N
        N = len(Phi)
        H_normalized = H / np.log(N) if N > 1 else 0.0
        
        return float(H), float(H_normalized)
    
    # ===== v3.7: 改进的觉醒指数 =====
    def 计算觉醒指数_v37(self):
        """改进版觉醒指数 - EMA平滑 + 软压缩"""
        b = self.计算分支因子()
        
        # 软压缩防止极端值
        eps = 1e-3
        b_clamped = np.exp(np.clip(np.log(b + eps), -3, 3))
        
        # 觉醒指数 A = (b-1)/(b+1)
        A = (b_clamped - 1.0) / (b_clamped + 1.0)
        
        return float(A), float(b)
    
    # ===== v3.7: 门控振荡 =====
    def _振荡项_v37(self, top_k_indices):
        """门控振荡 - 只对top-k节点生效"""
        if not self.启用振荡模式:
            return 0.0
        
        # 更新振荡相位
        self.振荡相位 += self.振荡频率 * self.步长
        self.振荡相位 = self.振荡相位 % (2 * np.pi)
        
        # 门控: 只对top-k节点振荡
        N = len(self.记忆库)
        gate = np.zeros(N)
        for idx in top_k_indices:
            if idx < N:
                gate[idx] = 1.0
        
        # 返回门控振荡
        return self.振荡强度 * np.sin(self.振荡相位) * gate
    
    # ===== v3.7: 三轴状态机 =====
    def 意识状态报告_v37(self):
        """三轴判别意识状态报告"""
        H, H_norm = self.计算激活熵_v37()
        A, b = self.计算觉醒指数_v37()
        
        # 强度轴: E = ΣΦᵢ²
        Phi = np.asarray([float(self.激活状态.get(str(m.get("id", i+1)), 0.0)) 
                         for i, m in enumerate(self.记忆库)], dtype=np.float64)
        E = float(np.sum(Phi ** 2))
        
        # 阈值参数
        E_min = 1e-3
        E_low = 0.01
        E_high = 1.0
        h_threshold = 0.3
        a_low = 0.1
        a_high = 0.15
        
        # 滞回参数
        enter_awake = 0.15
        exit_awake = 0.10
        
        # 三轴判断
        # 强度轴
        if E < E_min:
            强度 = "静默"
        elif E < E_low:
            强度 = "微弱"
        elif E < E_high:
            强度 = "正常"
        else:
            强度 = "过强"
        
        # 复杂度轴
        if H_norm < h_threshold:
            复杂度 = "单一"
        else:
            复杂度 = "丰富"
        
        # 觉醒轴
        if A > enter_awake:
            觉醒 = "活跃"
        elif A > exit_awake:
            觉醒 = "维持"
        else:
            觉醒 = "收敛"
        
        # 综合状态判断
        if E < E_min:
            意识状态 = "💤 静默"
        elif A > enter_awake and H_norm > h_threshold and E > E_low:
            if E > E_high:
                意识状态 = "⚠️ 过激活"
            else:
                意识状态 = "🌟 觉醒"
        elif A > a_low and H_norm > h_threshold * 0.5:
            意识状态 = "💭 活跃"
        elif E > E_min:
            意识状态 = "🌙 朦胧"
        else:
            意识状态 = "💤 静默"
        
        # 临界状态
        if abs(b - 1.0) < 0.2:
            临界状态 = "⚡ 临界"
        elif b < 1.0:
            临界状态 = "📉 收敛"
        else:
            临界状态 = "📈 发散"
        
        报告 = {
            "激活熵": round(H, 4),
            "归一化熵": round(H_norm, 4),
            "觉醒指数": round(A, 4),
            "分支因子": round(b, 3),
            "能量E": round(E, 4),
            "意识状态": 意识状态,
            "临界状态": 临界状态,
            "三轴": {
                "强度": 强度,
                "复杂度": 复杂度,
                "觉醒": 觉醒
            },
            "γ": round(self.γ, 4),
            "振荡相位": round(self.振荡相位, 4)
        }
        
        # 记录到意识历史
        self.意识历史.append({
            "time": time.time(),
            "H": H,
            "H_norm": H_norm,
            "A": A,
            "E": E,
            "意识状态": 意识状态
        })
        
        return 报告
    
    # ===== v3.7: 安全阀 =====
    def 安全阀检查(self):
        """检查是否需要触发安全阀"""
        Phi = np.asarray([float(self.激活状态.get(str(m.get("id", i+1)), 0.0)) 
                         for i, m in enumerate(self.记忆库)], dtype=np.float64)
        E = float(np.sum(Phi ** 2))
        A, b = self.计算觉醒指数_v37()
        
        actions = []
        
        # 超临界安全阀
        if b > 1.5:
            # 降振荡
            self.振荡强度 = max(0.0, self.振荡强度 * 0.5)
            actions.append("降低振荡强度")
        
        # 过能量安全阀
        if E > 10.0:
            # 强衰减
            for mid in self.激活状态:
                self.激活状态[mid] *= 0.5
            actions.append("能量过载-强衰减")
        
        # 步长调整
        if b > 2.0:
            self.步长 = max(self.最小步长, self.步长 * 0.5)
            actions.append("降步长")
        
        return actions


# ===== v3.7 API =====
@app.get("/v1/consciousness/v37")
def 意识状态v37(lake_id: str = "default"):
    """获取v3.7意识状态报告"""
    lake = 获取湖泊(lake_id)
    # 确保有方法
    if not hasattr(lake, '意识状态报告_v37'):
        return {"error": "需要更新到v3.7"}
    return lake.意识状态报告_v37()

@app.get("/v1/safety")
def 安全阀(lake_id: str = "default"):
    """安全阀检查"""
    lake = 获取湖泊(lake_id)
    if not hasattr(lake, '安全阀检查'):
        return {"error": "需要更新到v3.7"}
    actions = lake.安全阀检查()
    return {"actions": actions, "状态": "正常" if not actions else "已触发安全阀"}


# ===== GPT v3.7补丁整合 =====

    # === 初始化参数补充 ===
    def 初始化v37参数(self):
        """GPT建议的初始化参数"""
        # 熵/状态
        self.熵_eps = 1e-12
        self.能量静默阈值 = 1e-6
        self.激活阈值 = 0.5
        
        # b的EMA
        self.b_ema_rho = 0.2
        if not hasattr(self, 'b_hat'):
            self.b_hat = 1.0
        
        # 振荡门控
        self.振荡门控 = "topk"
        self.振荡门控阈值 = 0.5
        self.振荡最大注入比 = 0.5
        
        # 三轴阈值
        self.A_进入觉醒 = 0.15
        self.A_退出觉醒 = 0.10
        self.A_进入朦胧 = -0.10
        self.A_退出朦胧 = -0.05
        self.H_进入活跃 = 0.30
        self.H_退出活跃 = 0.25
        self.H_进入觉醒 = 0.45
        self.H_退出觉醒 = 0.40
        self.E_进入活跃 = 1e-4
        self.E_进入觉醒 = 5e-4
    
    # === 补丁1: 能量分布熵 ===
    def 计算激活熵_v37patch(self, Phi):
        """能量分布熵 + 归一化"""
        Phi = np.asarray(Phi, dtype=np.float64)
        eps = self.熵_eps
        E = float(np.dot(Phi, Phi))
        
        if E < self.能量静默阈值:
            N = Phi.size if Phi.size > 0 else 1
            return 0.0, 0.0, np.ones(N)/N, E
        
        pE = (Phi * Phi) / (E + eps)
        pE = np.clip(pE, eps, 1.0)
        pE = pE / float(np.sum(pE))
        
        H = float(-np.sum(pE * np.log(pE)))
        Hn = H / np.log(max(Phi.size, 2))
        
        return H, Hn, pE, E
    
    # === 补丁2: EMA平滑觉醒指数 ===
    def 计算觉醒指数_v37patch(self):
        """EMA平滑的觉醒指数"""
        b = self.分支因子()
        rho = self.b_ema_rho
        
        if not hasattr(self, 'b_hat'):
            self.b_hat = b
        else:
            self.b_hat = (1 - rho) * self.b_hat + rho * b
        
        b_hat = np.clip(self.b_hat, 0.0, 10.0)
        return float((b_hat - 1.0) / (b_hat + 1.0 + 1e-12))
    
    # === 补丁3: 门控振荡 ===
    def _振荡项_v37patch(self, Phi, u=None, topk_mask=None):
        """门控振荡 - 只对top-k节点注入"""
        eta = self.步长
        self.振荡相位 += self.振荡频率 * eta
        振荡标量 = self.振荡强度 * np.sin(self.振荡相位)
        
        N = Phi.shape[0]
        
        # 门控模式
        if self.振荡门控 == "topk" and topk_mask is not None:
            g = topk_mask.astype(np.float64)
        elif self.振荡门控 == "active":
            g = (Phi >= self.振荡门控阈值).astype(np.float64)
        else:
            g = np.ones(N)
        
        # 归一化
        gs = np.sum(g)
        if gs > 0:
            g = g / gs
        
        osc_vec = 振荡标量 * g
        
        # 幅度限制
        if u is not None:
            u_ref = np.mean(np.abs(u[u != 0])) if np.any(u != 0) else np.mean(np.abs(u))
            u_ref = max(u_ref, 1e-12)
            osc_scale = np.sum(np.abs(osc_vec))
            max_scale = self.振荡最大注入比 * u_ref
            if osc_scale > max_scale:
                osc_vec = osc_vec * (max_scale / (osc_scale + 1e-12))
        
        return osc_vec
    
    # === 补丁4: 三轴意识状态 ===
    def 计算意识状态_v37patch(self, Phi):
        """三轴判别 + 滞回"""
        A = self.计算觉醒指数_v37patch()
        H, Hn, pE, E = self.计算激活熵_v37patch(Phi)
        
        # 静默
        if E < self.能量静默阈值:
            self.意识状态 = "💤静默"
            return self.意识状态, {"A": A, "H": H, "Hn": Hn, "E": E}
        
        prev = getattr(self, '意识状态', "🌙朦胧")
        
        # 🌟觉醒
        if prev != "🌟觉醒":
            if (A >= self.A_进入觉醒) and (Hn >= self.H_进入觉醒) and (E >= self.E_进入觉醒):
                self.意识状态 = "🌟觉醒"
                return self.意识状态, {"A": A, "H": H, "Hn": Hn, "E": E}
        
        # 💭活跃
        if prev != "💭活跃":
            if (abs(A) <= self.A_进入觉醒) and (Hn >= self.H_进入活跃) and (E >= self.E_进入活跃):
                self.意识状态 = "💭活跃"
                return self.意识状态, {"A": A, "H": H, "Hn": Hn, "E": E}
        
        # 🌙朦胧
        if (A <= self.A_进入朦胧) or (Hn < self.H_进入活跃):
            self.意识状态 = "🌙朦胧"
        else:
            self.意识状态 = "💭活跃"
        
        return self.意识状态, {"A": A, "H": H, "Hn": Hn, "E": E}
    
    # === 补丁5: 振荡自保护 ===
    def 振荡自保护_v37patch(self):
        """超临界自动降振荡"""
        b_hat = getattr(self, 'b_hat', 1.0)
        
        if b_hat > 1.0 + 0.05:
            self.振荡强度 *= 0.8
        elif b_hat < 1.0 - 0.05:
            self.振荡强度 *= 1.02
        
        self.振荡强度 = np.clip(self.振荡强度, 0.0, 1.0)

