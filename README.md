# Semantic Ripple Lake 🌊🧠

AI Memory System with Consciousness Dynamics

## Features

- **Semantic Ripple**: Graph Laplacian-based semantic diffusion
- **IMEX Solver**: Stable large-step numerical integration  
- **Energy Field**: Memory energy function with entropy regularization
- **Critical Control**: Self-organizing to critical state (b→1)
- **LIF Pulse Layer**: Leaky Integrate-and-Fire neuron for spike avalanche
- **Consciousness State Machine**: Silent → Drowsy → Active → Awake

## Architecture

```
Query → Embedding → kNN Graph → IMEX Update → Energy Field → LIF Readout → Memory
                              ↓
                        Critical Control (b→1)
                              ↓
                       Consciousness Report
```

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn sentence-transformers numpy scipy

# Start service
python 涟漪湖v3.7_LIF版.py

# Test
curl -X POST "http://localhost:8012/v1/ripple/recall" \
  -H "Content-Type: application/json" \
  -d '{"query":"test","top_k":3,"lake_id":"test"}'
```

## API Endpoints

- `POST /v1/ripple/recall` - Query memories with consciousness
- `POST /v1/ripple/throw-stone` - Add new memory
- `GET /v1/stats` - System statistics
- `GET /v1/consciousness` - Consciousness report
- `GET /v1/critical` - Critical state detection

## Papers

Theoretical foundation in the repository.

## Links

- GitHub: https://github.com/etafibay06-beep/semantic-ripple-lake

## License

MIT
