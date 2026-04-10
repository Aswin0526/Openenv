#!/usr/bin/env python3
"""
inference.py — Warehouse RL Agent Inference Script
===================================================
OpenEnv-compatible inference script for the Warehouse Load Distribution Environment.

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from my_env.client import WarehouseEnv
from my_env.models import WarehouseAction

# ── Configuration ─────────────────────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("WAREHOUSE_TASK", "easy")
BENCHMARK = os.getenv("WAREHOUSE_BENCHMARK", "warehouse_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = 0.5


def _validate_env_vars() -> None:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not API_KEY:
        missing.append("API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert warehouse placement agent.
You must place products in a warehouse grid to maximize reward.

Rules:
- Easy mode: 2D grid [row, col]. Cluster products together for higher reward.
- Medium mode: 2D grid [row, col]. Place related products adjacent to each other.
- Hard mode: 3D grid [row, col, level]. Follow safety rules:
  * Fragile items -> level 2 (top rack)
  * Big items -> level 0 (bottom rack)
  * Flammable items -> away from heat/electrical zones

Respond with ONLY a JSON array representing the position.
Examples: [0, 1] for 2D, [1, 2, 0] for 3D.
No explanation, no markdown, just the array.
""").strip()


# ── Logging helpers ───────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent logic ───────────────────────────────────────────────────────

def build_user_prompt(obs: dict, step: int, history: List[str]) -> str:
    mode = obs.get("mode", "easy")
    current = obs.get("current_product", {})
    remaining = obs.get("products_remaining", 0)
    grid = obs.get("grid", [])
    history_block = "\n".join(history[-4:]) if history else "None"

    prompt = f"Mode: {mode}\nStep: {step}\nProducts remaining: {remaining}\n"
    prompt += f"Current product: {current}\n"
    prompt += f"Grid: {grid}\n"

    if mode == "medium":
        prompt += f"Related products: {obs.get('related_products', [])}\n"
        prompt += f"Placed products: {[(p['product']['name'], p['position']) for p in obs.get('placed_products', [])]}\n"
    elif mode == "hard":
        prompt += f"Heat zones: {obs.get('heat_zones', [])}\n"
        prompt += f"Electrical zones: {obs.get('electrical_zones', [])}\n"
        prompt += f"Safety rules: {obs.get('safety_rules', [])}\n"

    prompt += f"\nRecent history:\n{history_block}\n\nRespond with the position array only."
    return prompt


def get_action(client: OpenAI, obs: dict, step: int, history: List[str], mode: str) -> list:
    """Ask the LLM for a placement position. Falls back to greedy on failure."""
    import json
    import copy

    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse JSON array from response
        position = json.loads(text)
        if isinstance(position, list) and len(position) in (2, 3):
            return [int(x) for x in position]
    except Exception as exc:
        print(f"[DEBUG] LLM parse failed: {exc}", flush=True)

    # Greedy fallback: pick first empty cell
    return _greedy_fallback(obs, mode)


def _greedy_fallback(obs: dict, mode: str) -> list:
    """Return the first available empty cell."""
    grid = obs.get("grid", [])
    if mode == "hard":
        for level_idx, level in enumerate(grid):
            for r, row in enumerate(level):
                for c, val in enumerate(row):
                    if val == 0:
                        return [r, c, level_idx]
    else:
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 0:
                    return [r, c]
    return [0, 0] if mode != "hard" else [0, 0, 0]


# ── Main episode loop ─────────────────────────────────────────────────

async def run_episode(task_mode: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_mode, env=BENCHMARK, model=MODEL_NAME)
    try:
        _validate_env_vars()
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] OpenAI initialization failed: {exc}", flush=True)
        log_end(success=False, steps=steps_taken, score=score, rewards=rewards)
        return

    base_url = os.getenv("WAREHOUSE_SERVER_URL", "http://localhost:8000")

    async with WarehouseEnv(base_url=base_url) as env:
        try:
            await env.start(mode=task_mode)
            obs_obj = await env.reset()
            obs = obs_obj.__dict__

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                position = get_action(client, obs, step, history, task_mode)
                action = WarehouseAction(position=position)

                result = await env.step(action)
                obs = result["observation"].__dict__
                reward = result.get("reward", 0.0) or 0.0
                done = result.get("done", False)
                error = obs.get("message") if reward < 0 else None

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=str(position),
                    reward=reward,
                    done=done,
                    error=error,
                )
                history.append(f"Step {step}: pos={position} reward={reward:+.2f}")

                if done:
                    break

            # Normalize score: sum of rewards / max possible
            max_reward = steps_taken * 1.0 if steps_taken > 0 else 1.0
            score = sum(rewards) / max_reward if max_reward > 0 else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    await run_episode(TASK_NAME)


if __name__ == "__main__":
    asyncio.run(main())
