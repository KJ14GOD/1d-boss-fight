from __future__ import annotations

import argparse

import numpy as np

from game import BossArenaEnv, GameConfig, apply_level


def parse_args():
    parser = argparse.ArgumentParser(description="Play BossArenaEnv yourself with keyboard controls.")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2], help="Boss difficulty level.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def movement_to_index(up: bool, down: bool, left: bool, right: bool) -> int:
    if up and left and not down and not right:
        return 5
    if up and right and not down and not left:
        return 6
    if down and left and not up and not right:
        return 7
    if down and right and not up and not left:
        return 8
    if up and not down and not left and not right:
        return 1
    if down and not up and not left and not right:
        return 2
    if left and not right and not up and not down:
        return 3
    if right and not left and not up and not down:
        return 4
    return 0


def print_controls() -> None:
    print("Controls:")
    print("  Move: WASD or Arrow Keys")
    print("  Shoot: Space")
    print("  Reset episode: R")
    print("  Quit: Esc or close window")


def main():
    args = parse_args()

    cfg = GameConfig()
    cfg.render_fps = max(1, int(args.fps))
    apply_level(cfg, args.level)

    env = BossArenaEnv(cfg=cfg, render_mode="human")
    obs, info = env.reset(seed=args.seed)

    print(f"Human play started | level={args.level} | seed={args.seed}")
    print_controls()
    print(
        f"Spawn | player={info.get('player_pos')} | "
        f"boss={info.get('boss_pos')} | level_id={info.get('level_id')}"
    )

    # Ensure pygame display/event system is initialized before reading events.
    env.render()

    try:
        import pygame
    except Exception as exc:
        env.close()
        raise ImportError("pygame is required for human play. Install with: pip install pygame") from exc

    running = True
    while running:
        reset_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_requested = True

        if not running:
            break

        if reset_requested:
            obs, info = env.reset()
            print(
                f"Manual reset | player={info.get('player_pos')} | "
                f"boss={info.get('boss_pos')} | level_id={info.get('level_id')}"
            )
            env.render()
            continue

        keys = pygame.key.get_pressed()
        up = bool(keys[pygame.K_w] or keys[pygame.K_UP])
        down = bool(keys[pygame.K_s] or keys[pygame.K_DOWN])
        left = bool(keys[pygame.K_a] or keys[pygame.K_LEFT])
        right = bool(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        shoot = bool(keys[pygame.K_SPACE])

        move_idx = movement_to_index(up=up, down=down, left=left, right=right)
        action = np.array([move_idx, int(shoot)], dtype=np.int64)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(
                f"Episode done | level={info.get('level_id')} | win={info.get('win')} | "
                f"player_hp={info.get('player_hp'):.1f} | boss_hp={info.get('boss_hp'):.1f}"
            )
            obs, info = env.reset()
            print(
                f"Respawn | player={info.get('player_pos')} | "
                f"boss={info.get('boss_pos')} | level_id={info.get('level_id')}"
            )

    env.close()


if __name__ == "__main__":
    main()
