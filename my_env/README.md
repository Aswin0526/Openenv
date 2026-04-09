# Warehouse Load Distribution Environment

An [OpenEnv](https://github.com/speckai/openenv)-compatible, **Gymnasium-ready** reinforcement learning environment built for optimizing warehouse load distributions. 

## Features
- **RL-Optimized Output**: The environment natively exposes `observation, reward, done, info` tuples for immediate integration into Stable Baselines3, RLLib, or custom RL training loops.
- **Rich State Vectors**: Observations export `agent_matrix` layout schemas and mapped `product_features` as dense float representations `[id, is_fragile, is_flammable, size]`. No textual processing required.
- **Exploit Prevention**: Strict tracking bounds map out-of-bounds/collisions to negative penalties (`-0.1`) and force timeouts to prevent infinite loops, guaranteeing reliable learning flows. 

## Task Modes

### Easy — 2D Grid, Simple Placement
- **Grid:** 5x5 with 3 blocked cells
- **Products:** 5 uniform boxes (Box A through E)
- **Goal:** Maximize space utilization and prevent fragmentation. Clusters identical objects together.
- **Action shape:** `[row, col]`

### Medium — 2D Grid, Adjacency Constraints
- **Grid:** 6x6, starts empty
- **Products:** 10 items (categorized under tools, finishing, adhesives)
- **Constraints:** 6 adjacency rules (e.g., Hammer MUST go near Nails, Paint near Brush).
- **Goal:** Grouping objects intelligently relative to their categorical relations. 
- **Action shape:** `[row, col]`

### Hard — 3D Grid, Safety + Space Configurations
- **Grid:** 4x4x3 (4 rows, 4 cols, 3 rack levels)
- **Products:** 12 items with multiple dimensions (size: big/medium/small, fragile, flammable)
- **Constraints:**
  - Fragile items -> top rack.
  - Big items -> bottom rack. 
  - Flammable items -> strictly isolated from predefined heat and electrical zones. Separate from regular inventory. 
- **Danger zones:** Explicitly defined heat and electrical matrices. 
- **Action shape:** `[row, col, level]`

## Architecture Structure

```
my_env/
├── client.py               # WarehouseEnv async HTTP Python wrapper
├── models.py               # Pydantic Schemas mapping Actions, Observations, and Features
├── rllib_example.py        # Coming soon
└── server/
    ├── your_environment.py  # Core environment logic and task matrix definitions
    ├── app.py               # FastAPI HTTP wrapper hosting endpoints
    └── Dockerfile
```

## Setup & Training 

The `your_environment.WarehouseEnvironment` supports direct imports for standard Python RL pipelines:
```python
from my_env.server.your_environment import WarehouseEnvironment

env = WarehouseEnvironment()
env.start(mode="hard")

obs = env.reset()
done = False
while not done:
    # 1. Provide an action matching the dimensions 
    action = {"position": [2, 3, 0]}  # Network output
    
    # 2. Native Gym tuple output 
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
```

### Alternatively: API Server Pipeline

If you want to train over HTTP distributed calls or a Docker orchestration:

**Start the Server:**
```bash
cd my_env/server
uvicorn app:app --reload --port 8000
```

**Train over Web API**
```python
import requests
BASE = "http://localhost:8000"

# 1. Configure mode
requests.post(f"{BASE}/start", json={"mode": "hard"})
# 2. Extract initial matrix
obs = requests.post(f"{BASE}/reset").json()["observation"]

# 3. HTTP Step Loop
done = False
while not done:
    position = agent.forward(obs["agent_matrix"], obs["product_features"])
    
    # Packs back to [row, col, level]
    r = requests.post(f"{BASE}/step", json={"position": position})
    
    data = r.json()
    obs = data["observation"]
    reward = data["reward"]
    done = data["done"]
```

## Reward Dynamics
The rewards scale intelligently utilizing normalized outputs `[0.0, 1.0]`.  
- **Dense Progressions:** Compact spaces evaluate proximity heuristics (Manhattan distances towards identically grouped components).
- **Safety Score Scaling:** Flammable and Fragile violations severely reduce episode reward averages. 
- **Constraints Penalty**: Any agent logic invoking impossible moves (hitting already occupied squares or firing arrays outside the 4x4 matrix borders) inherits an immediate `-0.1` reward and wastes one of the `max_steps`, encouraging the model to instantly map borders limits logic.
  
## License

MIT
