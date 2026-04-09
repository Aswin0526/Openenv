"""
Warehouse Load Distribution Environment

Easy:   2D grid (5x5), simple box placement, avoid fragmentation
Medium: 2D grid (6x6), products with category + adjacency constraints
Hard:   3D grid (4x4x3), products with size/fragile/flammable + zone constraints

Each task mode runs as a completely separate episode.
Rewards are normalized to [0.0, 1.0] and computed per step.
"""

import copy
import uuid
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
#  Grid values:  0 = empty | -1 = blocked/obstacle | >0 = product ID
# ═══════════════════════════════════════════════════════════════════════

EASY_TASK = {
    "grid_shape": [5, 5],
    "initial_grid": [
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
    ],
    "products": [
        {"id": 1, "name": "Box A"},
        {"id": 2, "name": "Box B"},
        {"id": 3, "name": "Box C"},
        {"id": 4, "name": "Box D"},
        {"id": 5, "name": "Box E"},
    ],
    "max_steps": 20,
    "description": "Place boxes on a 2D grid. Avoid fragmentation.",
}


MEDIUM_TASK = {
    "grid_shape": [6, 6],
    "initial_grid": [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    "products": [
        {"id": 1,  "name": "Hammer",         "category": "tools"},
        {"id": 2,  "name": "Nails",           "category": "tools"},
        {"id": 3,  "name": "Screwdriver",     "category": "tools"},
        {"id": 4,  "name": "Screws",          "category": "tools"},
        {"id": 5,  "name": "Paint",           "category": "finishing"},
        {"id": 6,  "name": "Brush",           "category": "finishing"},
        {"id": 7,  "name": "Sandpaper",       "category": "finishing"},
        {"id": 8,  "name": "Wood Glue",       "category": "adhesives"},
        {"id": 9,  "name": "Tape",            "category": "adhesives"},
        {"id": 10, "name": "Measuring Tape",  "category": "tools"},
    ],
    "adjacency_constraints": [
        [1, 2],    # Hammer ↔ Nails
        [3, 4],    # Screwdriver ↔ Screws
        [5, 6],    # Paint ↔ Brush
        [5, 7],    # Paint ↔ Sandpaper
        [8, 9],    # Wood Glue ↔ Tape
        [1, 10],   # Hammer ↔ Measuring Tape
    ],
    "max_steps": 40,
    "description": "Place products while respecting adjacency constraints.",
}


HARD_TASK = {
    # 3 levels (0 = bottom rack, 2 = top rack), 4 rows x 4 cols each
    "grid_shape": [4, 4, 3],
    "initial_grid": [
        # Level 0 — bottom rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # Level 1 — middle rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # Level 2 — top rack
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ],
    "products": [
        {"id": 1,  "name": "Glass Vase",       "size": "small",  "fragile": True,  "flammable": False},
        {"id": 2,  "name": "Ceramic Plate Set", "size": "medium", "fragile": True,  "flammable": False},
        {"id": 3,  "name": "Fuel Can",          "size": "medium", "fragile": False, "flammable": True},
        {"id": 4,  "name": "Propane Tank",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 5,  "name": "Steel Shelf Unit",  "size": "big",    "fragile": False, "flammable": False},
        {"id": 6,  "name": "Wooden Crate",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 7,  "name": "LED Bulbs",         "size": "small",  "fragile": True,  "flammable": False},
        {"id": 8,  "name": "Paint Thinner",     "size": "small",  "fragile": False, "flammable": True},
        {"id": 9,  "name": "Rubber Tires",      "size": "big",    "fragile": False, "flammable": True},
        {"id": 10, "name": "Cotton Fabric",     "size": "medium", "fragile": False, "flammable": True},
        {"id": 11, "name": "Mirror",            "size": "medium", "fragile": True,  "flammable": False},
        {"id": 12, "name": "Concrete Bags",     "size": "big",    "fragile": False, "flammable": False},
    ],
    # Cells near heat sources — flammable items must NOT go here
    "heat_zones": [
        [0, 0, 0], [0, 1, 0],   # row 0, cols 0–1, level 0
        [0, 0, 1], [0, 1, 1],   # row 0, cols 0–1, level 1
    ],
    # Cells near electrical panels — flammable items must NOT go here
    "electrical_zones": [
        [3, 3, 0], [3, 2, 0],   # row 3, cols 2–3, level 0
        [3, 3, 1], [3, 2, 1],   # row 3, cols 2–3, level 1
    ],
    "safety_rules": [
        "Fragile items -> top rack (level 2)",
        "Flammable items -> away from heat zones and electrical zones",
        "Big items -> bottom rack (level 0)",
        "Flammable and non-flammable items must be separated",
    ],
    "max_steps": 60,
    "description": "3D warehouse with safety, size, and zone constraints.",
}


TASKS = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}


# ═══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class WarehouseEnvironment:
    """
    Warehouse Load Distribution Environment.

    Lifecycle per episode:
      start(mode)  → configure task (easy / medium / hard)
      reset()      → fresh episode, build grid + product queue
      step(action) → agent places one product per step
      state()      → inspect current episode metadata
    """

    VALID_MODES = list(TASKS.keys())

    def __init__(self):
        self._mode: str = "easy"
        self._episode_id: str = ""
        self._step_count: int = 0
        self._max_steps: int = 20
        self._done: bool = False

        self._grid: list = []
        self._grid_shape: list[int] = []
        self._products_queue: list[dict] = []
        self._placed_products: list[dict] = []
        self._task_config: dict = {}

    # ── LIFECYCLE ─────────────────────────────────────────────────────

    def start(self, mode: str = "easy") -> dict:
        """Configure for a task mode. Must call reset() after."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.VALID_MODES}")

        self._mode = mode
        self._task_config = TASKS[mode]
        self._grid_shape = self._task_config["grid_shape"]
        self._max_steps = self._task_config["max_steps"]

        # Clear episode state
        self._episode_id = ""
        self._step_count = 0
        self._done = False
        self._grid = []
        self._products_queue = []
        self._placed_products = []

        return {
            "mode": mode,
            "grid_shape": self._grid_shape,
            "total_products": len(self._task_config["products"]),
            "max_steps": self._max_steps,
            "description": self._task_config["description"],
            "status": "configured — call /reset to begin",
        }

    def reset(self) -> dict:
        """Start a fresh episode for the current mode."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._grid = copy.deepcopy(self._task_config.get("initial_grid", []))
        self._products_queue = list(self._task_config.get("products", []))
        self._placed_products = []

        return self._get_rl_observation("Episode started.")

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Place the next product at the given position."""
        if self._done:
            return (
                self._get_rl_observation("Episode finished. Call reset()."),
                0.0,
                True,
                {"error": "episode_done"}
            )

        if not self._episode_id:
            return (
                self._get_rl_observation("No active episode. Call reset()."),
                0.0,
                False,
                {"error": "no_episode"}
            )

        if not self._products_queue:
            self._done = True
            return (
                self._get_rl_observation("All products placed."),
                0.0,
                True,
                {"result": "all_placed"}
            )

        self._step_count += 1
        position = action.get("position", [])
        
        # Check timeout early for invalid moves
        is_timeout = self._step_count >= self._max_steps
        if is_timeout:
            self._done = True

        # ── Validate position ────────────────────────────────────────
        expected_dims = 3 if self._mode == "hard" else 2
        penalty = -0.1  # small penalty for invalid moves to guide RL agents

        if len(position) != expected_dims:
            return (
                self._get_rl_observation(f"Invalid position dimensions. Expected {expected_dims}D, got {len(position)}D."),
                penalty,
                self._done,
                {"error": "bad_dimensions"}
            )

        if not self._is_in_bounds(position):
            return (
                self._get_rl_observation(f"Position {position} is out of grid bounds {self._grid_shape}."),
                penalty,
                self._done,
                {"error": "out_of_bounds"}
            )

        cell_value = self._get_cell(position)
        if cell_value != 0:
            label = "blocked" if cell_value == -1 else f"occupied by product {cell_value}"
            return (
                self._get_rl_observation(f"Position {position} is {label}. Choose an empty cell."),
                penalty,
                self._done,
                {"error": "cell_unavailable"}
            )

        # ── Place the product ────────────────────────────────────────
        product = self._products_queue.pop(0)
        self._set_cell(position, product["id"])
        self._placed_products.append({
            "product": product,
            "position": position,
        })

        # Check if episode is done
        done = len(self._products_queue) == 0 or self._step_count >= self._max_steps
        self._done = done

        # ── Compute reward for this placement ────────────────────────
        if self._mode == "easy":
            reward = self._reward_easy(position)
        elif self._mode == "medium":
            reward = self._reward_medium(product, position)
        elif self._mode == "hard":
            reward = self._reward_hard(product, position)
        else:
            reward = 0.0

        msg = f"Placed '{product['name']}' (id={product['id']}) at {position}."
        if done and len(self._products_queue) == 0:
            msg += " All products placed."
        elif done:
            msg += " Max steps reached."

        return (
            self._get_rl_observation(msg),
            reward,
            done,
            {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "step_count": self._step_count,
                "product_placed": product,
                "position": position,
            }
        )

    def state(self) -> dict:
        """Return current episode metadata."""
        return {
            "state": {
                "episode_id": self._episode_id,
                "mode": self._mode,
                "step_count": self._step_count,
                "max_steps": self._max_steps,
                "grid_shape": self._grid_shape,
                "grid": self._grid,
                "products_placed": len(self._placed_products),
                "products_remaining": len(self._products_queue),
                "placed_products": self._placed_products,
                "done": self._done,
            }
        }

    # ── GRID HELPERS ─────────────────────────────────────────────────

    def _get_cell(self, position: list[int]) -> int:
        if self._mode == "hard":
            row, col, level = position
            return self._grid[level][row][col]
        else:
            row, col = position
            return self._grid[row][col]

    def _set_cell(self, position: list[int], value: int):
        if self._mode == "hard":
            row, col, level = position
            self._grid[level][row][col] = value
        else:
            row, col = position
            self._grid[row][col] = value

    def _is_in_bounds(self, position: list[int]) -> bool:
        if self._mode == "hard":
            row, col, level = position
            rows, cols, levels = self._grid_shape
            return 0 <= row < rows and 0 <= col < cols and 0 <= level < levels
        else:
            row, col = position
            rows, cols = self._grid_shape
            return 0 <= row < rows and 0 <= col < cols

    # ── REWARD FUNCTIONS ──────────────────────────────────────────────

    @staticmethod
    def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    # ── Easy: compactness via filled neighbors ───────────────────────

    def _count_neighbors_2d(self, position: list[int]) -> int:
        """Count filled neighbors (up/down/left/right) in a 2D grid."""
        row, col = position
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self._grid_shape[0] and 0 <= nc < self._grid_shape[1]:
                if self._grid[nr][nc] > 0:
                    count += 1
        return count

    def _reward_easy(self, position: list[int]) -> float:
        """
        Compact placement reward for easy mode.
        Base reward for valid placement + bonus per filled cardinal neighbor.
        Gradient: 0.1 (isolated) → 0.325 (1) → 0.55 (2) → 0.775 (3) → 1.0 (4)
        """
        num_neighbors = self._count_neighbors_2d(position)
        reward = 0.1 + 0.225 * num_neighbors
        return self._clip(reward)

    # ── Medium: relationship proximity + compactness ─────────────────

    def _manhattan_2d(self, a: list[int], b: list[int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_placed_position(self, product_id: int) -> list[int] | None:
        """Return the position of a previously placed product, or None."""
        for entry in self._placed_products:
            if entry["product"]["id"] == product_id:
                return entry["position"]
        return None

    def _reward_medium(self, product: dict, position: list[int]) -> float:
        """
        Relationship proximity + compactness reward for medium mode.
        Primary (70%): inverse Manhattan distance to nearest related placed product.
        Secondary (30%): proximity to nearest ANY placed product (cluster tightness).
        """
        constraints = self._task_config.get("adjacency_constraints", [])
        pid = product["id"]

        # Find IDs of products related to this one
        related_ids = set()
        for a, b in constraints:
            if a == pid:
                related_ids.add(b)
            elif b == pid:
                related_ids.add(a)

        # Relationship proximity
        if related_ids:
            min_dist = None
            for rid in related_ids:
                pos = self._find_placed_position(rid)
                if pos is not None:
                    d = self._manhattan_2d(position, pos)
                    if min_dist is None or d < min_dist:
                        min_dist = d
            if min_dist is not None:
                rel_reward = 1.0 / (1.0 + min_dist)
            else:
                # Related products exist in constraints but none placed yet
                rel_reward = 0.3
        else:
            # This product has no adjacency constraints
            rel_reward = 0.5

        # Compactness — proximity to nearest ANY placed product
        min_cluster_dist = None
        for entry in self._placed_products:
            if entry["product"]["id"] == pid:
                continue
            d = self._manhattan_2d(position, entry["position"])
            if min_cluster_dist is None or d < min_cluster_dist:
                min_cluster_dist = d
        if min_cluster_dist is not None:
            comp_reward = 1.0 / (1.0 + min_cluster_dist)
        else:
            # First product placed — neutral
            comp_reward = 0.5

        reward = 0.7 * rel_reward + 0.3 * comp_reward
        return self._clip(reward)

    # ── Hard: safety + relationship + compactness ────────────────────

    def _count_neighbors_3d(self, position: list[int]) -> int:
        """Count filled neighbors in a 3D grid (6 directions: ±row, ±col, ±level)."""
        row, col, level = position
        rows, cols, levels = self._grid_shape
        count = 0
        for dr, dc, dl in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nr, nc, nl = row + dr, col + dc, level + dl
            if 0 <= nr < rows and 0 <= nc < cols and 0 <= nl < levels:
                if self._grid[nl][nr][nc] > 0:
                    count += 1
        return count

    def _manhattan_3d(self, a: list[int], b: list[int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _is_in_zone(self, position: list[int], zone_list: list[list[int]]) -> bool:
        """Check if a 3D position falls within a zone list."""
        for zone in zone_list:
            if position[0] == zone[0] and position[1] == zone[1] and position[2] == zone[2]:
                return True
        return False

    def _reward_hard(self, product: dict, position: list[int]) -> float:
        """
        Multi-objective reward for hard mode.
        Safety compliance (50%): fragile→top, flammable→away from zones, big→bottom.
        Relationship proximity (30%): inverse distance to nearest same-property placed item.
        Compactness (20%): 3D neighbor count / max possible (6).
        """
        row, col, level = position
        is_fragile = product.get("fragile", False)
        is_flammable = product.get("flammable", False)
        size = product.get("size", "medium")

        # ── Safety compliance (3 sub-criteria, each 0 or 1) ──────────
        safety_points = 0
        safety_criteria = 3

        # 1. Fragile items should be on top rack (level 2)
        if is_fragile:
            if level == 2:
                safety_points += 1
        else:
            safety_points += 1  # non-fragile items satisfy this rule trivially

        # 2. Flammable items must be away from heat AND electrical zones
        heat_zones = self._task_config.get("heat_zones", [])
        elec_zones = self._task_config.get("electrical_zones", [])
        if is_flammable:
            in_heat = self._is_in_zone(position, heat_zones)
            in_elec = self._is_in_zone(position, elec_zones)
            if not in_heat and not in_elec:
                safety_points += 1
        else:
            safety_points += 1  # non-flammable items satisfy this rule trivially

        # 3. Big items should be on bottom rack (level 0)
        if size == "big":
            if level == 0:
                safety_points += 1
        else:
            safety_points += 1  # non-big items satisfy this rule trivially

        safety_score = safety_points / safety_criteria

        # ── Relationship proximity ───────────────────────────────────
        # Group by shared properties: same flammability status or same size
        min_dist = None
        for entry in self._placed_products:
            other = entry["product"]
            other_pos = entry["position"]
            # Skip self (just placed, already in _placed_products)
            if other["id"] == product["id"]:
                continue
            # Consider items related if they share a safety-relevant property
            same_flammable = other.get("flammable", False) == is_flammable
            same_fragile = other.get("fragile", False) == is_fragile
            if same_flammable or same_fragile:
                d = self._manhattan_3d(position, other_pos)
                if min_dist is None or d < min_dist:
                    min_dist = d

        if min_dist is not None:
            rel_score = 1.0 / (1.0 + min_dist)
        else:
            # First product of its kind — neutral score
            rel_score = 0.3

        # ── Compactness ──────────────────────────────────────────────
        num_neighbors = self._count_neighbors_3d(position)
        max_neighbors = 6  # 3D: up/down/left/right/above/below
        comp_score = num_neighbors / max_neighbors

        # ── Final weighted reward ────────────────────────────────────
        reward = (
            0.5 * safety_score
            + 0.3 * rel_score
            + 0.2 * comp_score
        )
        return self._clip(reward)

    # ── OBSERVATION BUILDER ──────────────────────────────────────────

    def _build_observation(self, message: str) -> dict:
        """Build a mode-appropriate observation dict."""
        current_product = self._products_queue[0] if self._products_queue else None

        obs = {
            "message": message,
            "mode": self._mode,
            "grid": self._grid,
            "grid_shape": self._grid_shape,
            "current_product": current_product,
            "products_remaining": len(self._products_queue),
            "done": self._done,
        }

        # ── Easy: no extra constraints ───────────────────────────────
        if self._mode == "easy":
            obs["blocked_cells"] = self._find_blocked_cells()

        # ── Medium: add adjacency constraints + related products ─────
        elif self._mode == "medium":
            constraints = self._task_config.get("adjacency_constraints", [])
            obs["adjacency_constraints"] = constraints
            if current_product:
                obs["related_products"] = self._get_related_products(current_product["id"])
            obs["placed_products"] = self._placed_products

        # ── Hard: add zones, safety rules, product properties ────────
        elif self._mode == "hard":
            obs["heat_zones"] = self._task_config.get("heat_zones", [])
            obs["electrical_zones"] = self._task_config.get("electrical_zones", [])
            obs["safety_rules"] = self._task_config.get("safety_rules", [])
            obs["placed_products"] = self._placed_products

        return obs

    def _find_blocked_cells(self) -> list[list[int]]:
        """Find all blocked (-1) cells in a 2D grid."""
        blocked = []
        for r, row in enumerate(self._grid):
            for c, val in enumerate(row):
                if val == -1:
                    blocked.append([r, c])
        return blocked

    def _get_related_products(self, product_id: int) -> list[dict]:
        """For medium mode: find products that should be adjacent."""
        constraints = self._task_config.get("adjacency_constraints", [])
        all_products = self._task_config.get("products", [])
        product_map = {p["id"]: p for p in all_products}

        related_ids = set()
        for a, b in constraints:
            if a == product_id:
                related_ids.add(b)
            elif b == product_id:
                related_ids.add(a)

        return [product_map[rid] for rid in related_ids if rid in product_map]

    def _get_rl_observation(self, message: str) -> dict:
        """
        Return the observation dict extended with numerical matrices 
        and feature vectors optimized for RL.
        """
        obs = self._build_observation(message)
        
        # Add useful numerical mappings for the agent
        # Create a numerical vector for the current product
        product_vector = []
        if obs.get("current_product"):
            p = obs["current_product"]
            # Convert properties to numerical flags/values
            size_map = {"small": 0, "medium": 1, "big": 2}
            
            product_vector = [
                float(p.get("id", 0)),
                float(p.get("fragile", False)),
                float(p.get("flammable", False)),
                float(size_map.get(p.get("size", "medium"), 1))
            ]
        obs["product_features"] = product_vector
        obs["agent_matrix"] = self._grid  # explicitly provide matrix reference
        
        return obs
