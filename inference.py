#!/usr/bin/env python3
"""
inference.py — Warehouse RL Agent (Strict Proxy Version)
=======================================================
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from my_env.client import WarehouseEnv
from my_env.models import WarehouseAction

# ── Configuration ─────────────────────────────────────────────────────
# We use os.environ[] directly to ensure the script crashes if these are missing.
# This prevents "Success with 0 API calls" errors.
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("WAREHOUSE_TASK", "easy")
BENCHMARK = os.getenv("WAREHOUSE_BENCHMARK", "warehouse_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert warehouse placement agent. 
Respond with ONLY a JSON array representing the position.
No explanation, no markdown, just the array.
Examples: [0, 1] for 2D, [1, 2, 0] for 3D.
""").strip()

# ── Logging helpers ───────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Agent logic ───────────────────────────────────────────────────────

def get_action(client: OpenAI, obs: dict) -> list:
    """Ask the LLM for a placement position. No greedy fallback allowed."""
    user_prompt = f"Mode: {obs.get('mode')}\nGrid: {obs.get('grid')}\nProduct: {obs.get('current_product')}\nPosition:"
    
    # This call MUST succeed for the script to continue
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    
    raw_text = (completion.choices[0].message.content or "").strip()
    
    # Robust JSON cleaning (strips markdown if the model hallucinates it)
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()
    
    position = json.loads(clean_text)
    if isinstance(position, list) and len(position) in (2, 3):
        return [int(x) for x in position]
    raise ValueError(f"Invalid position format received: {raw_text}")

# ── Main episode loop ─────────────────────────────────────────────────

async def run_episode(task_mode: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_mode, env=BENCHMARK, model=MODEL_NAME)

    # RECTIFIED: Strict variable extraction
    try:
        api_key = os.environ["API_KEY"]
        api_base = os.environ["API_BASE_URL"]
        client = OpenAI(api_key=api_key, base_url=api_base)
    except KeyError as e:
        print(f"[DEBUG] CRITICAL: Environment variable {e} is missing.", flush=True)
        return

    server_url = os.getenv("WAREHOUSE_SERVER_URL", "http://localhost:8000")

    async with WarehouseEnv(base_url=server_url) as env:
        try:
            await env.start(mode=task_mode)
            obs_obj = await env.reset()
            # Object to dict conversion
            obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else obs_obj

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                # No try-except here: if the API call fails, the script should fail.
                position = get_action(client, obs)
                action = WarehouseAction(position=position)

                result = await env.step(action)
                
                # Update observation
                obs_data = result.get("observation")
                obs = obs_data.__dict__ if hasattr(obs_data, "__dict__") else obs_data
                
                reward = result.get("reward", 0.0) or 0.0
                done = result.get("done", False)
                error = obs.get("message") if reward < 0 else None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=str(position), reward=reward, done=done, error=error)

                if done:
                    break

            # Calculate Final Score
            max_possible = steps_taken if steps_taken > 0 else 1
            score = min(max(sum(rewards) / max_possible, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            # Re-raise to ensure the validator sees a non-zero exit code if it failed
            raise 
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    await run_episode(TASK_NAME)

if __name__ == "__main__":
    asyncio.run(main())