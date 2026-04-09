"""
app.py — HuggingFace Space entry point
=======================================
Serves:
  - Gradio UI  (mounted at /)
  - FastAPI endpoints required by the OpenEnv validator:
      POST /reset  → returns {"status": "ok"}
      GET  /health → returns {"status": "ok"}
"""

import time
import copy

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from my_env.server.your_environment import WarehouseEnvironment, EASY_TASK, MEDIUM_TASK, HARD_TASK

# ── FastAPI app (validator endpoints) ────────────────────────────────

api = FastAPI()

_env_global = WarehouseEnvironment()


@api.get("/health")
async def health():
    return {"status": "ok"}


@api.post("/reset")
async def reset():
    """
    OpenEnv validator pings this endpoint.
    Resets the environment and returns 200 OK.
    """
    try:
        _env_global.start("easy")
        _env_global.reset()
    except Exception:
        pass
    return JSONResponse(content={"status": "ok"}, status_code=200)


# ── Gradio helpers ────────────────────────────────────────────────────

def get_short_labels(task_config):
    labels = {}
    for p in task_config["products"]:
        name = p["name"]
        if len(name) <= 4:
            labels[p["id"]] = name[:4]
        else:
            words = name.split()
            if len(words) > 1:
                labels[p["id"]] = (words[0][:2] + words[1][:2]).upper()
            else:
                labels[p["id"]] = name[:4].upper()
    return labels


def format_grid_markdown(grid, mode, labels):
    if mode == "hard":
        output = ""
        for lvl, level_grid in enumerate(grid):
            level_name = ["BOTTOM (0)", "MIDDLE (1)", "TOP (2)"][lvl] if lvl < 3 else f"L{lvl}"
            output += f"### Level: {level_name}\n"
            real_table = "| " + " | ".join([f"C{i}" for i in range(len(level_grid[0]))]) + " |\n"
            real_table += "| " + " | ".join(["---"] * len(level_grid[0])) + " |\n"
            for row in level_grid:
                real_table += "| " + " | ".join(
                    [f"**{labels[v]}**" if v > 0 else ("❌" if v == -1 else "◻️") for v in row]
                ) + " |\n"
            output += real_table + "\n"
        return output
    else:
        table = "| " + " | ".join([f"C{i}" for i in range(len(grid[0]))]) + " |\n"
        table += "| " + " | ".join(["---"] * len(grid[0])) + " |\n"
        for row in grid:
            table += "| " + " | ".join(
                [f"**{labels[v]}**" if v > 0 else ("🚫" if v == -1 else "◻️") for v in row]
            ) + " |\n"
        return table


def simulate(mode):
    env = WarehouseEnvironment()
    env.start(mode)
    env.reset()

    task_configs = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK}
    labels = get_short_labels(task_configs[mode])

    total_reward = 0
    step_num = 0
    log_text = f"**Started {mode.upper()} Mode**\n\n"
    grid_view = format_grid_markdown(env._grid, mode, labels)
    yield grid_view, log_text, "0.00", "0.00"
    time.sleep(0.5)

    def get_empty_cells():
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

    def evaluate_position(product, pos):
        saved_grid = copy.deepcopy(env._grid)
        saved_queue = list(env._products_queue)
        saved_placed = list(env._placed_products)
        saved_step = env._step_count
        saved_done = env._done

        if product in env._products_queue:
            env._products_queue.remove(product)
        env._set_cell(pos, product["id"])
        env._placed_products.append({"product": product, "position": pos})

        if mode == "easy":
            r = env._reward_easy(pos)
        elif mode == "medium":
            r = env._reward_medium(product, pos)
        else:
            r = env._reward_hard(product, pos)

        env._grid = saved_grid
        env._products_queue = saved_queue
        env._placed_products = saved_placed
        env._step_count = saved_step
        env._done = saved_done
        return r

    while env._products_queue and not env._done:
        product = env._products_queue[0]
        empty_cells = get_empty_cells()

        best_pos, best_reward = None, -1
        for pos in empty_cells:
            r = evaluate_position(product, pos)
            if r > best_reward:
                best_reward, best_pos = r, pos

        obs, actual_reward, done, info = env.step({"position": best_pos})
        total_reward += actual_reward
        step_num += 1

        props = ""
        if mode == "hard":
            tags = []
            if product.get("fragile"):
                tags.append("FRAG")
            if product.get("flammable"):
                tags.append("FLAM")
            tags.append(product.get("size", "?"))
            props = f" ({','.join(tags)})"

        log_text += f"- **Step {step_num}:** Placed `{product['name']}` {props} at `{best_pos}` | Reward: `{actual_reward:.4f}`\n"
        grid_view = format_grid_markdown(env._grid, mode, labels)
        avg_reward = total_reward / step_num
        yield grid_view, log_text, f"{total_reward:.4f}", f"{avg_reward:.4f}"
        time.sleep(0.8)

    log_text += "\n**Simulation Complete!**"
    avg_reward = total_reward / step_num if step_num > 0 else 0
    yield grid_view, log_text, f"{total_reward:.4f}", f"{avg_reward:.4f}"


# ── Gradio UI ─────────────────────────────────────────────────────────

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")

with gr.Blocks(theme=theme, title="Warehouse RL Optimizer") as demo:
    gr.Markdown("# 📦 OpenEnv Warehouse RL Optimizer")
    gr.Markdown("Visualize the greedy agent converging to optimal placements in real-time.")

    with gr.Row():
        with gr.Column(scale=1):
            mode_selector = gr.Radio(["easy", "medium", "hard"], label="Task Mode", value="easy")
            run_btn = gr.Button("🚀 Run Simulation", variant="primary")
            total_rew = gr.Textbox(label="Total Reward", value="0.00")
            avg_rew = gr.Textbox(label="Avg Reward Per Step", value="0.00")
        with gr.Column(scale=2):
            grid_display = gr.Markdown("### Grid Layout\nPress run to start simulation.")

    gr.Markdown("### Step Log")
    log_display = gr.Markdown("> Awaiting simulation...")

    run_btn.click(
        fn=simulate,
        inputs=[mode_selector],
        outputs=[grid_display, log_display, total_rew, avg_rew],
    )

# ── Mount Gradio onto FastAPI ─────────────────────────────────────────

app = gr.mount_gradio_app(api, demo, path="/")
