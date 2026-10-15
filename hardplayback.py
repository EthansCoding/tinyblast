"""Watch two players controlled by the same trained Tiny Blast policy."""
import argparse
from pathlib import Path

import numpy as np
import pygame

from arena_env import ACTION_REPEAT, tinyBlasterEnv
from evaluate import DEFAULT_MODEL, load_model
from game import DT, HEIGHT, WIDTH, draw_bullet, draw_player


def predict_actions(model, env):
    # Both decisions see the same world state, each from its own perspective.
    observations = np.stack([
        env.observation_for(env.MLPlayer),
        env.observation_for(env.SCRIPTPlayer),
    ])
    actions, _ = model.predict(observations, deterministic=True)
    return actions


def draw_frame(screen, font, env, episode, paused, result):
    screen.fill((25, 25, 30))
    for player, label, color in (
        (env.MLPlayer, "AI 1", "skyblue"),
        (env.SCRIPTPlayer, "AI 2", "orange"),
    ):
        if player in env.handler.players:
            draw_player(screen, player)
            pygame.draw.circle(screen, color, player.position, player.radius + 4, 2)
            text = font.render(label, True, color)
            screen.blit(text, text.get_rect(center=(player.position.x, player.position.y - 28)))
    for bullet in env.handler.bullets:
        draw_bullet(screen, bullet)
    lines = [
        f"Round {episode}   AI 1 HP: {max(0, env.MLPlayer.health)}   "
        f"AI 2 HP: {max(0, env.SCRIPTPlayer.health)}   Time: {env.physics_steps * DT:.1f}s",
        "Same policy for both players   Space: pause   R: restart   Esc: quit",
    ]
    if result:
        lines.append(result + " — next round shortly")
    if paused:
        lines.append("Paused")
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, "white"), (12, 12 + index * 24))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", type=Path, default=DEFAULT_MODEL,
        help="Checkpoint used by BOTH players (default: arena_ai_v2.zip).",
    )
    parser.add_argument(
        "--fps", type=int, default=round(1 / DT),
        help="Physics frames per second (default: 60; four frames per AI decision).",
    )
    parser.add_argument("--seed", type=int, default=50_000)
    args = parser.parse_args()
    if args.fps <= 0 or args.seed < 0:
        parser.error("--fps must be positive and --seed must be nonnegative")

    env = tinyBlasterEnv(action_repeat=1, shaping=False)
    try:
        try:
            model = load_model(args.model, env)
        except FileNotFoundError:
            parser.error(f"Model not found: {args.model}. Train a model with train.py first.")
        except ValueError as exc:
            parser.error(str(exc))

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tiny Blast — AI vs AI")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

        env.reset(seed=args.seed)
        decision_frame = 0
        episode = 1
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
                env.reset(seed=args.seed + episode)
                decision_frame = 0
                episode += 1
                result = ""

            if not paused and not result:
                if decision_frame % ACTION_REPEAT == 0:
                    actions = predict_actions(model, env)
                _, _, terminated, truncated, info = env.step(
                    actions[0], opponent_action=actions[1],
                )
                decision_frame += 1
                if terminated or truncated:
                    result = {
                        "win": "AI 1 wins", "loss": "AI 2 wins",
                        "draw": "Draw", "timeout": "Time limit reached",
                    }[info["outcome"]]
                    print(f"Round {episode}: {result} | {info['sim_seconds']:.1f}s", flush=True)
                    restart_at = pygame.time.get_ticks() + 1500

            draw_frame(screen, font, env, episode, paused, result)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
