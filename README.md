# Semantic Ripple Lake 🌊🧠

AI记忆系统 + 意识动力学

## 核心特性

- **语义涟漪**: 基于图拉普拉斯的语义扩散
- **IMEX求解**: 稳定的大步长数值积分
- **能量场**: 带熵正则的记忆能量函数
- **临界控制**: 自适应调节到临界状态
- **LIF脉冲层**: 脉冲雪崩事件检测
- **意识状态机**: 静默→朦胧→活跃→觉醒

## 架构

```
Query → 嵌入 → kNN图 → IMEX更新 → 能量场 → LIF读出 → 记忆
                    ↓
              临界控制(b→1)
                    ↓
              意识状态报告
```

## 快速开始

```bash
# 安装依赖
pip install fastapi uvicorn sentence-transformers numpy scipy

# 启动服务
python 涟漪湖v3.7_LIF版.py

# 测试
curl -X POST "http://localhost:8012/v1/ripple/recall" \
  -H "Content-Type: application/json" \
  -d '{"query":"测试","top_k":3,"lake_id":"test"}'
```

## 论文

参见 `涟漪湖原理详解.md`

## 联系

- GitHub: https://github.com/etafibay06-beep/semantic-ripple-lake
