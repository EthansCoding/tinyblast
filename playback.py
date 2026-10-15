import argparse
from pathlib import Path

import pygame
from stable_baselines3 import PPO

from arena_env import tinyBlasterEnv
from game import DT, HEIGHT, WIDTH, draw_bullet, draw_player


def draw_frame(screen, font, env, episode, total_reward, paused, result):
    screen.fill((25, 25, 30))
    for player, label, color in (
        (env.MLPlayer, "AI", "skyblue"),
        (env.SCRIPTPlayer, "BOT", "orange"),
    ):
        if player in env.handler.players:
            draw_player(screen, player)
            pygame.draw.circle(screen, color, player.position, player.radius + 4, 2)
            text = font.render(label, True, color)
            screen.blit(text, text.get_rect(center=(player.position.x, player.position.y - 28)))

    for bullet in env.handler.bullets:
        draw_bullet(screen, bullet)

    lines = [
        f"Round {episode}   AI HP: {max(0, env.MLPlayer.health)}   "
        f"BOT HP: {max(0, env.SCRIPTPlayer.health)}   Reward: {total_reward:.2f}",
        "Space: pause/resume   R: restart   Esc: quit",
    ]
    if result:
        lines.append(result + " — next round shortly")
    if paused:
        lines.append("Paused")
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, "white"), (12, 12 + index * 24))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Watch the trained Tiny Blast model play.")
    parser.add_argument(
        "model", nargs="?", type=Path,
        default=Path(__file__).resolve().with_name("arena_ai.zip"),
        help="Model archive (default: arena_ai.zip beside this script).",
    )
    parser.add_argument(
        "--fps", type=int, default=round(1 / DT),
        help="Playback steps per second (default: 60).",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    env = tinyBlasterEnv()
    try:
        try:
            model = PPO.load(args.model, env=env, device="cpu")
        except FileNotFoundError:
            parser.error(f"Model not found: {args.model}. Train a model with train.py first.")

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tiny Blast — trained model playback")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

        observation, _ = env.reset()
        episode = 1
        total_reward = 0.0
        paused = False
        result = ""
        restart_at = 0

        while True:
            clock.tick(args.fps)
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        restart = True

            if restart or (result and not paused and pygame.time.get_ticks() >= restart_at):
                observation, _ = env.reset()
                episode += 1
                total_reward = 0.0
                result = ""

            if not paused and not result:
                action, _ = model.predict(observation, deterministic=True)
                # env.step advances the same fixed timestep used during training.
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    if terminated:
                        if not env.handler.players:
                            result = "Draw"
                        elif env.MLPlayer in env.handler.players:
                            result = "AI wins"
                        else:
                            result = "Bot wins"
                    else:
                        result = "Time limit reached"
                    print(f"Round {episode}: {result} | reward {total_reward:.2f}")
                    restart_at = pygame.time.get_ticks() + 1500

            draw_frame(screen, font, env, episode, total_reward, paused, result)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
