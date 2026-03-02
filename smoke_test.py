from __future__ import annotations

import argparse
import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from game import BossArenaEnv, GameConfig


def run_smoke_test(episodes: int, seed: int) -> None:
    cfg = GameConfig()
    env = BossArenaEnv(cfg=cfg, render_mode=None)

    # Gymnasium API validation (headless-safe).
    check_env(env, skip_render_check=True)

    wins = 0
    total_steps = 0
    ep_lengths: list[int] = []
    ep_rewards: list[float] = []

    obs, info = env.reset(seed=seed)
    assert obs.shape == (cfg.obs_dim,), f"Bad obs shape at reset: {obs.shape}"
    assert np.isfinite(obs).all(), "Non-finite value in reset observation."

    for ep in range(episodes):
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_steps += 1
            ep_reward += float(reward)

            assert obs.shape == (cfg.obs_dim,), f"Bad obs shape at step: {obs.shape}"
            assert np.isfinite(obs).all(), "Non-finite value in step observation."
            assert math.isfinite(reward), "Non-finite reward value."

        wins += int(bool(info.get("win", False)))
        ep_lengths.append(steps)
        ep_rewards.append(ep_reward)

        if ep < episodes - 1:
            obs, info = env.reset()
            assert obs.shape == (cfg.obs_dim,), f"Bad obs shape after reset: {obs.shape}"
            assert np.isfinite(obs).all(), "Non-finite value in post-episode reset observation."

    env.close()

    avg_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    win_rate = (wins / episodes) * 100.0 if episodes > 0 else 0.0

    print("Smoke test passed.")
    print(f"Episodes: {episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average episode length: {avg_len:.2f}")
    print(f"Average episode reward: {avg_reward:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless smoke test for BossArenaEnv.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed for first episode.")
    args = parser.parse_args()

    run_smoke_test(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
