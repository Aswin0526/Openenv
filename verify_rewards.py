"""Reward verification with grid display after each placement."""
import requests

BASE = "http://127.0.0.1:8000"

def print_grid_2d(grid):
    for row in grid:
        print(f"    {row}")

def print_grid_3d(grid):
    for lvl, level_grid in enumerate(grid):
        print(f"    Level {lvl}: {level_grid}")

# === EASY MODE ===
print("=== EASY MODE (5x5 2D) ===\n")
requests.post(f"{BASE}/start", json={"mode": "easy"})
requests.post(f"{BASE}/reset")

for pos, desc in [([0,0],"isolated"), ([0,1],"adj"), ([0,2],"adj"), ([3,0],"isolated"), ([3,1],"1 neighbor")]:
    r = requests.post(f"{BASE}/step", json={"position": pos}).json()
    print(f"  {desc:12s} at {str(pos):10s} -> reward={r['reward']:.4f}")
    print_grid_2d(r["observation"]["grid"])
    print()

# === MEDIUM MODE — Scenario A: Screwdriver placed FAR ===
print("\n=== MEDIUM MODE — Screwdriver FAR from cluster ===\n")
requests.post(f"{BASE}/start", json={"mode": "medium"})
requests.post(f"{BASE}/reset")

for pos, desc in [([0,0],"Hammer"), ([0,1],"Nails(rel)"), ([5,5],"Screw.FAR"), ([5,4],"Screws(rel)")]:
    r = requests.post(f"{BASE}/step", json={"position": pos}).json()
    print(f"  {desc:12s} at {str(pos):10s} -> reward={r['reward']:.4f}")
    print_grid_2d(r["observation"]["grid"])
    print()

# === MEDIUM MODE — Scenario B: Screwdriver placed NEAR ===
print("\n=== MEDIUM MODE — Screwdriver NEAR cluster ===\n")
requests.post(f"{BASE}/start", json={"mode": "medium"})
requests.post(f"{BASE}/reset")

for pos, desc in [([0,0],"Hammer"), ([0,1],"Nails(rel)"), ([0,3],"Screw.NEAR"), ([0,4],"Screws(rel)")]:
    r = requests.post(f"{BASE}/step", json={"position": pos}).json()
    print(f"  {desc:12s} at {str(pos):10s} -> reward={r['reward']:.4f}")
    print_grid_2d(r["observation"]["grid"])
    print()

# === HARD MODE ===
print("\n=== HARD MODE (4x4x3 3D) ===\n")
requests.post(f"{BASE}/start", json={"mode": "hard"})
requests.post(f"{BASE}/reset")

for pos, desc in [
    ([1,1,2], "Vase(frag)OK"),
    ([2,2,0], "Plate(frag)BAD"),
    ([2,1,0], "FuelCan OK"),
    ([0,0,0], "Propane BAD"),
    ([1,0,0], "Steel OK"),
]:
    r = requests.post(f"{BASE}/step", json={"position": pos}).json()
    print(f"  {desc:15s} at {str(pos):12s} -> reward={r['reward']:.4f}")
    print_grid_3d(r["observation"]["grid"])
    print()
