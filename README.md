---
title: Warehouse RL Optimizer
emoji: 📦
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Warehouse RL Optimizer

An [OpenEnv](https://github.com/speckai/openenv)-compatible reinforcement learning environment for optimizing warehouse load distributions.

## Features

- **3 difficulty modes**: Easy (2D), Medium (2D + adjacency), Hard (3D + safety rules)
- **Greedy agent visualization**: Watch the agent place products in real-time
- **OpenEnv compatible**: Exposes `/reset` and `/health` endpoints for validation

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Gradio UI |
| `/reset` | POST | Reset environment (OpenEnv validator) |
| `/health` | GET | Health check |

## Running Inference

Set environment variables and run:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export WAREHOUSE_TASK=easy

python inference.py
```

## Environment Modes

- **Easy**: 5×5 2D grid, 5 boxes, maximize compactness
- **Medium**: 6×6 2D grid, 10 products with adjacency constraints
- **Hard**: 4×4×3 3D grid, 12 products with safety rules (fragile/flammable/size)
