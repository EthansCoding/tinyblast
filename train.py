from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from gym import tinyBlasterEnv # Your gym.py file and environment class


def main():
    env = tinyBlasterEnv()

    # Check observation shapes, types, and return values.
    check_env(env)

    # Track episode rewards and lengths.
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=100_000)
        model.save("arena_ai")
    finally:
        env.close()


if __name__ == "__main__":
    main()
