"""
models.py — Warehouse environment data contracts.

Action, Observation, and State dataclasses for the Warehouse Load
Distribution Environment. Used by the client for type-safe serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WarehouseAction:
    """Agent action: place the current product at this position."""
    position: list[int] = field(default_factory=list)
    # [row, col]        for easy / medium
    # [row, col, level] for hard


@dataclass
class WarehouseObservation:
    """Observation returned after step or reset."""
    message: str = ""
    mode: str = "easy"
    grid: list = field(default_factory=list)
    grid_shape: list[int] = field(default_factory=list)
    current_product: Optional[dict] = None
    products_remaining: int = 0
    done: bool = False
    reward: float = 0.0

    # Easy mode
    blocked_cells: list = field(default_factory=list)

    # Medium mode
    adjacency_constraints: list = field(default_factory=list)
    related_products: list = field(default_factory=list)
    placed_products: list = field(default_factory=list)

    # Hard mode
    heat_zones: list = field(default_factory=list)
    electrical_zones: list = field(default_factory=list)
    safety_rules: list = field(default_factory=list)

    # RL specific mappings
    product_features: list = field(default_factory=list)
    agent_matrix: list = field(default_factory=list)


@dataclass
class WarehouseState:
    """Full episode state snapshot."""
    episode_id: str = ""
    mode: str = "easy"
    step_count: int = 0
    max_steps: int = 20
    grid_shape: list[int] = field(default_factory=list)
    grid: list = field(default_factory=list)
    products_placed: int = 0
    products_remaining: int = 0
    placed_products: list = field(default_factory=list)
    done: bool = False
