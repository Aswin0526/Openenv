#!/usr/bin/env python3
import asyncio
import os
import textwrap
import json
import sys
from typing import List, Optional

from openai import OpenAI
from my_env.client import WarehouseEnv
from my_env.models import WarehouseAction

# ── Configuration ─────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy_key")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.environ.get("WAREHOUSE_TASK", "easy")
BENCHMARK = "warehouse_env"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
TEMPERATURE = 0.3
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = "You are a warehouse agent. Reply ONLY with a JSON array [row, col] or [row, col, level]. No markdown."

# ── Logging helpers ───────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)

# ── Agent logic ───────────────────────────────────────────────────────

def get_action_llm(client: OpenAI, obs: dict) -> list:
    """Strict LLM call to get position."""
    prompt = f"Mode: {obs.get('mode')}\nGrid: {obs.get('grid')}\nProduct: {obs.get('current_product')}"
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    
    res = completion.choices[0].message.content.strip()
    # Remove markdown formatting if present
    clean_res = res.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_res)

# ── Main logic ────────────────────────────────────────────────────────

async def run_episode():
    # 1. Initialize Client
    if not API_KEY or not API_BASE_URL:
        print("[WARN] API_KEY or API_BASE_URL looks empty; proceeding anyway")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # 2. PROXY CHECK: Force a single API call before starting (Like your friend's code)
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "check"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"[DEBUG] Proxy connection check failed: {e}")

    # 3. Environment Setup
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    server_url = os.environ.get("WAREHOUSE_SERVER_URL", "http://localhost:8000")
    
    rewards = []
    steps_taken = 0
    final_success = False
    final_score = 0.0

    async with WarehouseEnv(base_url=server_url) as env:
        try:
            await env.start(mode=TASK_NAME)
            obs_obj = await env.reset()
            obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else obs_obj

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                try:
                    pos = get_action_llm(client, obs)
                except Exception:
                    # Very simple fallback if LLM produces garbage JSON
                    grid = obs.get("grid", [])
                    pos = [0, 0, 0] if TASK_NAME == "hard" else [0, 0]

                action = WarehouseAction(position=pos)
                result = await env.step(action)
                
                # Unwrap observation
                obs_data = result.get("observation")
                obs = obs_data.__dict__ if hasattr(obs_data, "__dict__") else obs_data
                
                curr_reward = result.get("reward", 0.0)
                done = result.get("done", False)
                
                rewards.append(curr_reward)
                steps_taken = step
                
                log_step(step, str(pos), curr_reward, done, obs.get("message") if curr_reward < 0 else None)
                
                if done:
                    break

            # Calculate score (normalized)
            total_r = sum(rewards)
            final_score = min(max(total_r / (steps_taken or 1), 0.0), 1.0)
            final_success = final_score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[DEBUG] Runtime error: {e}")
        finally:
            log_end(final_success, steps_taken, final_score, rewards)

if __name__ == "__main__":
    asyncio.run(run_episode())