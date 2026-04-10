"""
easy_grader.py — Grader for the Easy warehouse task.

Scores the agent's performance on compact placement in a 2D grid.
Returns a float strictly in (0, 1).
"""


def _clamp_strict(value: float) -> float:
    """Clamp to the open interval (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, value))


def grade(episode_data: dict) -> float:
    """
    Grade an easy-mode episode.

    Parameters
    ----------
    episode_data : dict
        Must contain:
          - "rewards"  : list[float]  — per-step rewards
          - "steps"    : int          — total steps taken
          - "success"  : bool         — whether the episode was successful

    Returns
    -------
    float  in (0, 1)
    """
    rewards = episode_data.get("rewards", [])
    steps = episode_data.get("steps", 0)
    success = episode_data.get("success", False)

    if not rewards or steps == 0:
        return 0.01  # minimum non-zero score

    avg_reward = sum(rewards) / len(rewards)

    # Bonus for completing all placements successfully
    completion_bonus = 0.1 if success else 0.0

    score = avg_reward + completion_bonus
    return _clamp_strict(score)
