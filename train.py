import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from arena_env import ACTION_REPEAT, ENV_VERSION, FRAME_GAMMA, tinyBlasterEnv
from evaluate import DEFAULT_MODEL, evaluate_model, load_model


EPISODE_METRICS = (
    "damage_dealt", "damage_taken", "shots_fired", "is_success", "outcome", "opponent"
)


class DamageMetricsCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                for key in ("damage_dealt", "damage_taken", "shots_fired"):
                    self.logger.record_mean(f"rollout/{key}_mean", info[key])
                self.logger.record_mean("rollout/timeout_rate", info["outcome"] == "timeout")
        return True


class CombatEvalCallback(BaseCallback):
    """Select by wins first, then damage balance, using unshaped combat rewards."""

    def __init__(self, env, log_dir, *, eval_freq, episodes, seed):
        super().__init__()
        self.env = env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.next_eval = eval_freq
        self.best_score = None

    def evaluate(self):
        result = evaluate_model(self.model, self.env, episodes=self.episodes, seed=self.seed)
        result["timesteps"] = self.num_timesteps
        for key, value in result.items():
            if isinstance(value, (int, float)):
                self.logger.record(f"eval/{key}", value)
        score = (result["win_rate"], result["mean_damage_dealt"] - result["mean_damage_taken"],
                 -result["timeout_rate"])
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.model.save(self.log_dir / "best_model.zip")
            (self.log_dir / "best_eval.json").write_text(json.dumps(result, indent=2) + "\n")
        with (self.log_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)
        print(f"Combat evaluation at {self.num_timesteps:,}: "
              f"wins {result['win_rate']:.0%}, damage {result['mean_damage_dealt']:.2f}/"
              f"{result['mean_damage_taken']:.2f}, timeouts {result['timeout_rate']:.0%}", flush=True)

    def _on_step(self):
        # num_timesteps counts transitions across ALL environments.
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval = self.num_timesteps + self.eval_freq
        return True

    def _on_training_end(self):
        # Periodic evaluation runs during collection, before PPO's gradient update.
        # Always evaluate the updated weights, even at the same transition count.
        self.evaluate()


def training_stages(total_timesteps):
    practice_steps = total_timesteps // 5
    return (
        ("target_practice", practice_steps,
         dict(bot_fires=False, bot_style="stationary", easy_start=True)),
        ("randomized_practice", practice_steps,
         dict(bot_fires=False, bot_style="mixed")),
        ("sparring", practice_steps, dict(bot_style="mixed", bot_skill=0.5)),
        ("combat", total_timesteps - 3 * practice_steps, dict(bot_style="mixed")),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Tiny Blast with strafing and a combat curriculum.")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Policy transitions across all environments; 20/20/20/40%% curriculum.")
    parser.add_argument("--gamma", type=float, default=FRAME_GAMMA ** ACTION_REPEAT,
                        help="Discount per policy decision (default: 0.999 ** 4).")
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout length per environment.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--test-seed", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--log-dir", type=Path,
                        default=Path("training_logs") / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.steps < 5:
        parser.error("--steps must be at least 5")
    if not 0 < args.gamma < 1 or not 0 < args.gae_lambda <= 1:
        parser.error("--gamma must be in (0, 1) and --gae-lambda in (0, 1]")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0 or not 0 < args.learning_rate < 1:
        parser.error("--ent-coef must be finite and nonnegative; --learning-rate must be in (0, 1)")
    if min(args.n_envs, args.n_steps, args.torch_threads, args.eval_freq, args.eval_episodes) <= 0:
        parser.error("Environment, rollout, thread, and evaluation counts must be positive")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size < 2 or rollout_size < args.batch_size or rollout_size % args.batch_size:
        parser.error("--batch-size must be at least 2 and divide n-envs * n-steps")
    if min(args.seed, args.eval_seed, args.test_seed) < 0:
        parser.error("Seeds must be nonnegative")
    if abs(args.test_seed - args.eval_seed) < args.eval_episodes:
        parser.error("Validation and final test seed ranges must not overlap")
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("--log-dir must be new or empty to preserve previous experiments")

    torch.set_num_threads(args.torch_threads)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stages = training_stages(args.steps)
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment_version": ENV_VERSION, "action_repeat": ACTION_REPEAT,
        "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
        "numpy": np.__version__, "stages": stages,
    }
    config_path = args.log_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    check = tinyBlasterEnv(bot_style="mixed", reward_gamma=args.gamma)
    try:
        check_env(check)
    finally:
        check.close()
    eval_env = tinyBlasterEnv(bot_style="mixed", shaping=False)
    try:
        eval_callback = CombatEvalCallback(
            eval_env, args.log_dir, eval_freq=args.eval_freq,
            episodes=args.eval_episodes, seed=args.eval_seed,
        )
        callbacks = [DamageMetricsCallback(), eval_callback]
        model = None
        for index, (name, steps, settings) in enumerate(stages):
            env = make_vec_env(
                tinyBlasterEnv, n_envs=args.n_envs, seed=args.seed + index * args.n_envs,
                env_kwargs={**settings, "reward_gamma": args.gamma},
                monitor_dir=str(args.log_dir / name),
                monitor_kwargs={"info_keywords": EPISODE_METRICS},
            )
            try:
                print(f"\nTraining {name}: {steps:,} transitions", flush=True)
                if model is None:
                    model = PPO(
                        "MlpPolicy", env, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        learning_rate=args.learning_rate, ent_coef=args.ent_coef,
                        n_steps=args.n_steps, batch_size=args.batch_size,
                        target_kl=0.03, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                        verbose=1, device="cpu", seed=args.seed,
                    )
                    model.set_logger(configure(str(args.log_dir), ["stdout", "csv"]))
                else:
                    model.set_env(env, force_reset=True)
                    model.ep_info_buffer.clear()
                    model.ep_success_buffer.clear()
                model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callbacks)
                model.save(args.log_dir / name / "model.zip")
            finally:
                env.close()
        model.save(args.log_dir / "final_model.zip")
        # Publish the best validated checkpoint, retaining the final policy for comparison.
        best = load_model(args.log_dir / "best_model.zip", eval_env)
        best.save(args.output)
        test_result = evaluate_model(best, eval_env, episodes=args.eval_episodes, seed=args.test_seed)
        (args.log_dir / "test_eval.json").write_text(json.dumps(test_result, indent=2) + "\n")
        config["actual_timesteps"] = model.num_timesteps
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved best policy to {args.output}; final policy and reports in {args.log_dir}")
        print(f"Held-out combat win rate: {test_result['win_rate']:.0%}")
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
