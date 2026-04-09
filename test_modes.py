"""
Test script — demonstrates execution across easy → medium → hard.
Each mode is a completely separate episode.
"""

import requests
import json

BASE = "http://127.0.0.1:8000"

def pp(label, data):
    print(f"\n{label}")
    print(json.dumps(data, indent=2))

# ═══════════════════════════════════════════════════════════════════
#  EASY MODE
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EASY MODE — 5×5 2D Grid, 5 Simple Boxes, Blocked Cells")
print("=" * 70)

r = requests.post(f"{BASE}/start", json={"mode": "easy"})
pp("[POST /start]", r.json())

r = requests.post(f"{BASE}/reset")
obs = r.json()["observation"]
print("\n[POST /reset] Initial grid:")
for row in obs["grid"]:
    print(f"  {row}")
print(f"  Blocked cells: {obs['blocked_cells']}")
print(f"  Current product: {obs['current_product']}")
print(f"  Products remaining: {obs['products_remaining']}")

# Place Box A at (0, 0)
r = requests.post(f"{BASE}/step", json={"position": [0, 0]})
d = r.json()
print(f"\n[POST /step] Place at [0,0]: {d['observation']['message']}  reward={d['reward']:.4f}")
print("  Grid:")
for row in d["observation"]["grid"]:
    print(f"    {row}")

# Try blocked cell (1,1)
r = requests.post(f"{BASE}/step", json={"position": [1, 1]})
print(f"\n[POST /step] Try blocked [1,1]: {r.json()['observation']['message']}  reward={r.json()['reward']:.4f}")

# Place Box B at (0, 1)
r = requests.post(f"{BASE}/step", json={"position": [0, 1]})
d2 = r.json()
print(f"[POST /step] Place at [0,1]: {d2['observation']['message']}  reward={d2['reward']:.4f}")

r = requests.get(f"{BASE}/state")
s = r.json()["state"]
print(f"\n[GET /state] Steps={s['step_count']}  Placed={s['products_placed']}  Remaining={s['products_remaining']}")


# ═══════════════════════════════════════════════════════════════════
#  MEDIUM MODE — completely fresh episode
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  MEDIUM MODE — 6×6 2D Grid, 10 Products, Adjacency Constraints")
print("=" * 70)

r = requests.post(f"{BASE}/start", json={"mode": "medium"})
pp("[POST /start]", r.json())

r = requests.post(f"{BASE}/reset")
obs = r.json()["observation"]
print("\n[POST /reset] Initial grid:")
for row in obs["grid"]:
    print(f"  {row}")
print(f"  Current product: {obs['current_product']}")
print(f"  Products remaining: {obs['products_remaining']}")
print(f"  Adjacency constraints: {obs['adjacency_constraints']}")
print(f"  Related products for current: {obs['related_products']}")

# Place Hammer at (0, 0)
r = requests.post(f"{BASE}/step", json={"position": [0, 0]})
d = r.json()
print(f"\n[POST /step] Place Hammer at [0,0]: {d['observation']['message']}  reward={d['reward']:.4f}")
print(f"  Next product: {d['observation']['current_product']}")
print(f"  Related products for Nails: {d['observation']['related_products']}")

# Place Nails at (0, 1) — adjacent to Hammer
r = requests.post(f"{BASE}/step", json={"position": [0, 1]})
d = r.json()
print(f"\n[POST /step] Place Nails at [0,1] (adjacent to Hammer): {d['observation']['message']}  reward={d['reward']:.4f}")
print(f"  Grid after 2 placements:")
for row in d["observation"]["grid"]:
    print(f"    {row}")

r = requests.get(f"{BASE}/state")
s = r.json()["state"]
print(f"\n[GET /state] Steps={s['step_count']}  Placed={s['products_placed']}  Remaining={s['products_remaining']}")
print(f"  Placed products: {[(p['product']['name'], p['position']) for p in s['placed_products']]}")


# ═══════════════════════════════════════════════════════════════════
#  HARD MODE — completely fresh episode
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  HARD MODE — 4×4×3 (3D), 12 Products, Safety + Zone Constraints")
print("=" * 70)

r = requests.post(f"{BASE}/start", json={"mode": "hard"})
pp("[POST /start]", r.json())

r = requests.post(f"{BASE}/reset")
obs = r.json()["observation"]
print("\n[POST /reset] Initial 3D grid (3 levels):")
for lvl, level_grid in enumerate(obs["grid"]):
    print(f"  Level {lvl}:")
    for row in level_grid:
        print(f"    {row}")
print(f"\n  Current product: {obs['current_product']}")
print(f"  Products remaining: {obs['products_remaining']}")
print(f"  Heat zones: {obs['heat_zones']}")
print(f"  Electrical zones: {obs['electrical_zones']}")
print(f"  Safety rules:")
for rule in obs["safety_rules"]:
    print(f"    - {rule}")

# Place Glass Vase (fragile) at top rack level 2 — correct placement
r = requests.post(f"{BASE}/step", json={"position": [1, 1, 2]})
d = r.json()
print(f"\n[POST /step] Place Glass Vase at [1,1,2] (top rack): {d['observation']['message']}  reward={d['reward']:.4f}")

# Place Fuel Can (flammable) — should go away from heat/electrical zones
r = requests.post(f"{BASE}/step", json={"position": [2, 2, 0]})
d = r.json()
print(f"[POST /step] Place Ceramic Plate Set at [2,2,0]: {d['observation']['message']}  reward={d['reward']:.4f}")

# Place another
r = requests.post(f"{BASE}/step", json={"position": [2, 1, 0]})
d = r.json()
print(f"[POST /step] Place Fuel Can at [2,1,0] (away from zones): {d['observation']['message']}  reward={d['reward']:.4f}")

print(f"\n  3D Grid after placements:")
for lvl, level_grid in enumerate(d["observation"]["grid"]):
    print(f"  Level {lvl}:")
    for row in level_grid:
        print(f"    {row}")

r = requests.get(f"{BASE}/state")
s = r.json()["state"]
print(f"\n[GET /state] Steps={s['step_count']}  Placed={s['products_placed']}  Remaining={s['products_remaining']}")
print(f"  Placed: {[(p['product']['name'], p['position']) for p in s['placed_products']]}")


# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  SUMMARY — Each mode ran as a completely independent episode")
print("=" * 70)
print("  Easy:   5×5 grid  | 5 boxes   | blocked cells only")
print("  Medium: 6×6 grid  | 10 items  | + adjacency constraints")
print("  Hard:   4×4×3 3D  | 12 items  | + size/fragile/flammable + zones")
print("  Rewards: Computed per step, normalized [0.0, 1.0]")