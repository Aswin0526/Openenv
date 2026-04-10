#!/usr/bin/env python3
"""
inference.py — LLM-powered agent for the Warehouse Load Distribution Environment.

Runs all 3 task modes (easy, medium, hard) so the validator sees ≥3 graded tasks.
Each task's score is clamped to the strict open interval (0, 1).
"""

import asyncio
import os
import json
import sys
from typing import List, Optional

from openai import OpenAI
from my_env.client import WarehouseEnv
from my_env.models import WarehouseAction
from graders.easy_grader import grade as grade_easy
from graders.medium_grader import grade as grade_medium
from graders.hard_grader import grade as grade_hard

# ── Configuration ─────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy_key")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "warehouse_env"
MAX_STEPS_MAP = {"easy": 20, "medium": 40, "hard": 60}
TEMPERATURE = 0.3
MAX_TOKENS = 256

SYSTEM_PROMPT = "You are a warehouse agent. Reply ONLY with a JSON array [row, col] or [row, col, level]. No markdown."

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


# ── Helpers ───────────────────────────────────────────────────────────

def clamp_strict(value: float) -> float:
    """Clamp to open interval (0, 1)."""
    return max(0.01, min(0.99, value))


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(task, success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)


# ── Agent logic ───────────────────────────────────────────────────────

def get_action_llm(client: OpenAI, obs: dict, task_name: str) -> list:
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


# ── Single episode ────────────────────────────────────────────────────

async def run_single_task(client: OpenAI, task_name: str, server_url: str) -> dict:
    """Run one episode for a given task mode. Returns episode data for grading."""
    max_steps = MAX_STEPS_MAP.get(task_name, 20)
    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards = []
    steps_taken = 0
    final_success = False
    final_score = 0.0

    async with WarehouseEnv(base_url=server_url) as env:
        try:
            await env.start(mode=task_name)
            obs_obj = await env.reset()
            obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else obs_obj

            for step in range(1, max_steps + 1):
                if obs.get("done", False):
                    break

                try:
                    pos = get_action_llm(client, obs, task_name)
                except Exception:
                    # Fallback if LLM produces garbage JSON
                    pos = [0, 0, 0] if task_name == "hard" else [0, 0]

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

            # Calculate score
            if rewards and steps_taken > 0:
                avg_r = sum(rewards) / len(rewards)
                final_score = clamp_strict(avg_r)
                final_success = final_score >= 0.5
            else:
                final_score = 0.01
                final_success = False

        except Exception as e:
            print(f"[DEBUG] Runtime error in {task_name}: {e}")
            final_score = 0.01
        finally:
            log_end(task_name, final_success, steps_taken, final_score, rewards)

    return {
        "task": task_name,
        "rewards": rewards,
        "steps": steps_taken,
        "success": final_success,
        "score": final_score,
    }


# ── Main: run all 3 tasks ────────────────────────────────────────────

async def run_all_episodes():
    # 1. Initialize LLM Client
    if not API_KEY or not API_BASE_URL:
        print("[WARN] API_KEY or API_BASE_URL looks empty; proceeding anyway")

    print(f"[CONFIG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", flush=True)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # 2. Proxy warmup call
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "check"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"[DEBUG] Proxy warmup failed: {e}")

    # 3. Server URL
    server_url = os.environ.get("WAREHOUSE_SERVER_URL", "http://localhost:8000")

    # 4. Run each task mode
    task_modes = ["easy", "medium", "hard"]
    results = {}

    for task in task_modes:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running task: {task.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        episode_data = await run_single_task(client, task, server_url)

        # Grade with the appropriate grader
        grader = GRADERS[task]
        graded_score = grader(episode_data)
        graded_score = clamp_strict(graded_score)

        results[task] = {
            "score": graded_score,
            "steps": episode_data["steps"],
            "success": episode_data["success"],
        }

        print(f"[GRADE] task={task} graded_score={graded_score:.3f}", flush=True)

    # 5. Summary
    print(f"\n{'='*60}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task, r in results.items():
        print(f"  {task:8s} → score={r['score']:.3f}  steps={r['steps']}  success={r['success']}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    asyncio.run(run_all_episodes())