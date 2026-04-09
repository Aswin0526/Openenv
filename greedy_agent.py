"""
Greedy Agent Verification
=========================
For each product, evaluates ALL valid empty cells, picks the one with the
highest reward, and places it there. This simulates what a well-trained RL
model should converge toward. Shows step-by-step decisions + final grid.
"""

import copy

from my_env.server.your_environment import WarehouseEnvironment, EASY_TASK, MEDIUM_TASK, HARD_TASK

env = WarehouseEnvironment()


def get_empty_cells(mode):
    """Return all empty cell positions for the current grid."""
    cells = []
    if mode == "hard":
        rows, cols, levels = env._grid_shape
        for r in range(rows):
            for c in range(cols):
                for l in range(levels):
                    if env._grid[l][r][c] == 0:
                        cells.append([r, c, l])
    else:
        rows, cols = env._grid_shape
        for r in range(rows):
            for c in range(cols):
                if env._grid[r][c] == 0:
                    cells.append([r, c])
    return cells


def evaluate_position(product, pos, mode):
    """Temporarily place product, compute reward, then undo."""
    # Save state
    saved_grid = copy.deepcopy(env._grid)
    saved_queue = list(env._products_queue)
    saved_placed = list(env._placed_products)

    # Simulate placement
    if product in env._products_queue:
        env._products_queue.remove(product)
    env._set_cell(pos, product["id"])
    env._placed_products.append({"product": product, "position": pos})

    # Compute reward
    if mode == "easy":
        reward = env._reward_easy(pos)
    elif mode == "medium":
        reward = env._reward_medium(product, pos)
    else:
        reward = env._reward_hard(product, pos)

    # Restore state
    env._grid = saved_grid
    env._products_queue = saved_queue
    env._placed_products = saved_placed

    return reward


def print_grid_2d(grid, labels=None):
    """Print 2D grid with optional product labels."""
    for r, row in enumerate(grid):
        cells = []
        for c, val in enumerate(row):
            if val == -1:
                cells.append("  X ")
            elif val == 0:
                cells.append("  . ")
            else:
                if labels and val in labels:
                    cells.append(f"{labels[val]:>4s}")
                else:
                    cells.append(f"  {val} ")
        print("    " + "|".join(cells))
    print()


def print_grid_3d(grid, labels=None):
    """Print 3D grid level by level."""
    for lvl, level_grid in enumerate(grid):
        level_name = ["BOTTOM", "MIDDLE", "TOP"][lvl] if lvl < 3 else f"L{lvl}"
        print(f"    === Level {lvl} ({level_name}) ===")
        for r, row in enumerate(level_grid):
            cells = []
            for c, val in enumerate(row):
                if val == 0:
                    cells.append("  .  ")
                else:
                    if labels and val in labels:
                        cells.append(f"{labels[val]:>5s}")
                    else:
                        cells.append(f"  {val}  ")
            print("      " + "|".join(cells))
        print()


def run_greedy_episode(mode):
    """Run a full greedy episode, placing each product at the best position."""
    env.start(mode)
    env.reset()

    # Build short labels for products
    task = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK}[mode]
    labels = {}
    for p in task["products"]:
        name = p["name"]
        # Create short 3-4 char label
        if len(name) <= 4:
            labels[p["id"]] = name[:4]
        else:
            words = name.split()
            if len(words) > 1:
                labels[p["id"]] = (words[0][:2] + words[1][:2]).upper()
            else:
                labels[p["id"]] = name[:4].upper()

    total_reward = 0
    step_num = 0

    while env._products_queue and not env._done:
        product = env._products_queue[0]
        empty_cells = get_empty_cells(mode)

        # Evaluate all positions
        best_pos = None
        best_reward = -1
        for pos in empty_cells:
            r = evaluate_position(product, pos, mode)
            if r > best_reward:
                best_reward = r
                best_pos = pos

        # Place at best position
        obs, actual_reward, done, info = env.step({"position": best_pos})
        total_reward += actual_reward
        step_num += 1

        # Product properties for hard mode
        props = ""
        if mode == "hard":
            tags = []
            if product.get("fragile"):
                tags.append("FRAG")
            if product.get("flammable"):
                tags.append("FLAM")
            tags.append(product.get("size", "?"))
            props = f" [{','.join(tags)}]"

        print(f"  Step {step_num:2d}: {product['name']:20s}{props:20s} -> {str(best_pos):12s}  reward={actual_reward:.4f}")

    avg_reward = total_reward / step_num if step_num > 0 else 0
    print(f"\n  Total reward: {total_reward:.4f}  |  Avg per step: {avg_reward:.4f}")
    print(f"\n  FINAL GRID:")

    if mode == "hard":
        print_grid_3d(env._grid, labels)
    else:
        print_grid_2d(env._grid, labels)

    return total_reward, avg_reward


# =====================================================================
print("=" * 70)
print("  GREEDY AGENT — Picks highest-reward position for each product")
print("  This simulates optimal RL policy convergence")
print("=" * 70)

# --- EASY ---
print("\n" + "-" * 70)
print("  EASY MODE — Goal: Compact cluster, avoid fragmentation")
print("-" * 70)
total_e, avg_e = run_greedy_episode("easy")

# --- MEDIUM ---
print("\n" + "-" * 70)
print("  MEDIUM MODE — Goal: Related products adjacent + tight cluster")
print("-" * 70)
total_m, avg_m = run_greedy_episode("medium")

# --- HARD ---
print("\n" + "-" * 70)
print("  HARD MODE — Goal: Safety rules + grouping + compactness")
print("  Rules: Fragile->top, Flammable->away from zones, Big->bottom")
print("-" * 70)
total_h, avg_h = run_greedy_episode("hard")

# --- SUMMARY ---
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Easy:   avg_reward={avg_e:.4f}  total={total_e:.4f}  (5 products)")
print(f"  Medium: avg_reward={avg_m:.4f}  total={total_m:.4f}  (10 products)")
print(f"  Hard:   avg_reward={avg_h:.4f}  total={total_h:.4f}  (12 products)")
print()
print("  If grids show tight clusters (easy/medium) and correct zone")
print("  placement (hard), the reward system is guiding correctly.")
print("=" * 70)
