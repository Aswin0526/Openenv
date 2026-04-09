"""
server/app.py — FastAPI server for Warehouse Load Distribution Environment.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.your_environment import WarehouseEnvironment

app = FastAPI(
    title="Warehouse Load Distribution API",
    version="0.1.0",
)

env = WarehouseEnvironment()


class StartRequest(BaseModel):
    mode: str = "easy"


class ActionRequest(BaseModel):
    position: list[int]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset():
    try:
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start")
async def start(request: StartRequest):
    try:
        result = env.start(request.mode)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(action: ActionRequest):
    try:
        obs, reward, done, info = env.step(action.model_dump())
        return {"observation": obs, "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
