"""
app.py — FastAPI application for the Warehouse Load Distribution Environment.

Endpoints:
    POST /start  — Configure task mode (easy / medium / hard)
    POST /reset  — Reset environment, get initial observation
    POST /step   — Place a product at a position
    GET  /state  — Get current episode state
    GET  /health — Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from your_environment import WarehouseEnvironment

# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Warehouse Load Distribution API",
    description="OpenEnv-compatible warehouse environment with easy/medium/hard modes",
    version="0.1.0",
)

env = WarehouseEnvironment()


# ── Request Models ────────────────────────────────────────────────────

class StartRequest(BaseModel):
    mode: str = "easy"

class ActionRequest(BaseModel):
    position: list[int]     # [row, col] for 2D  |  [row, col, level] for 3D


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/start")
async def start(request: StartRequest):
    """Configure the warehouse for a task mode."""
    try:
        result = env.start(request.mode)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    """Reset the environment and return initial observation."""
    try:
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: ActionRequest):
    """Place the next product at the given position."""
    try:
        obs, reward, done, info = env.step(action.model_dump())
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get current episode state."""
    try:
        result = env.state()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
