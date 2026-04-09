"""
client.py — Async HTTP client for the Warehouse environment.
"""

import httpx
from typing import Optional

from .models import WarehouseAction, WarehouseObservation, WarehouseState


class WarehouseEnv:
    """
    Client for the Warehouse Load Distribution Environment.

    Usage:
        async with WarehouseEnv(base_url="http://localhost:8000") as env:
            await env.start(mode="easy")
            obs = await env.reset()
            result = await env.step(WarehouseAction(position=[2, 3]))
            state = await env.state()
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def start(self, mode: str = "easy") -> dict:
        """Configure the warehouse for a task mode."""
        response = await self._client.post("/start", json={"mode": mode})
        response.raise_for_status()
        return response.json()

    async def reset(self) -> WarehouseObservation:
        """Reset the environment and return the initial observation."""
        response = await self._client.post("/reset")
        response.raise_for_status()
        data = response.json()
        return WarehouseObservation(**data.get("observation", {}))

    async def step(self, action: WarehouseAction) -> dict:
        """Place a product at the given position."""
        payload = {"position": action.position}
        response = await self._client.post("/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return {
            "observation": WarehouseObservation(**data.get("observation", {})),
            "reward": data.get("reward", 0.0),
            "done": data.get("done", False),
            "info": data.get("info", {}),
        }

    async def state(self) -> WarehouseState:
        """Get the current episode state."""
        response = await self._client.get("/state")
        response.raise_for_status()
        data = response.json()
        return WarehouseState(**data.get("state", {}))
