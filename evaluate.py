"""Repeatable combat evaluation, also used to select training checkpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces

from arena_env import tinyBlasterEnv


DEFAULT_MODEL = Path(__file__).resolve().with_name("arena_ai_v2.zip")


def load_model(path, env):
    model = PPO.load(path, device="cpu")
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
    except ValueError as exc:
        raise ValueError(
            "This checkpoint uses incompatible actions/observations. "
            "Run train.py to train a v2 model with strafing and combat observations."
        ) from exc
    return model


def evaluate_model(model, env, *, episodes=20, seed=10_000):
    """Use the exact same spawn/opponent seeds for every candidate policy."""
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes must be positive and seed must be nonnegative")
    rounds = []
    for index in range(episodes):
        observation, _ = env.reset(seed=seed + index)
        total_reward = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                rounds.append({"seed": seed + index, "reward": total_reward, **info})
                break
    outcomes = Counter(row["outcome"] for row in rounds)
    result = {
        "episodes": episodes, "seed": seed,
        "win_rate": outcomes["win"] / episodes,
        "loss_rate": outcomes["loss"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "draw_rate": outcomes["draw"] / episodes,
    }
    for key in ("reward", "damage_dealt", "damage_taken", "shots_fired", "sim_seconds"):
        result[f"mean_{key}"] = float(np.mean([row[key] for row in rounds]))
    result["by_opponent"] = {}
    for opponent in sorted({row["opponent"] for row in rounds}):
        group = [row for row in rounds if row["opponent"] == opponent]
        result["by_opponent"][opponent] = {
            "episodes": len(group),
            "win_rate": sum(row["is_success"] for row in group) / len(group),
        }
    result["rounds"] = rounds
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000,
                        help="Use seeds separate from training's checkpoint selection.")
    parser.add_argument("--bot-style", choices=("stationary", "strafe", "mixed"), default="mixed")
    parser.add_argument("--output", type=Path, help="Optional JSON report including every round.")
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error("--episodes must be positive and --seed must be nonnegative")
    env = tinyBlasterEnv(bot_style=args.bot_style, shaping=False)
    try:
        try:
            model = load_model(args.model, env)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        result = evaluate_model(model, env, episodes=args.episodes, seed=args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
