from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from game import BossArenaEnv, GameConfig, apply_level


def find_vecnormalize_path(model_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"VecNormalize file not found: {p}")
        return p

    direct = model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")
    if direct.exists():
        return direct

    candidates = [
        model_path.parent / "latest_vecnormalize.pkl",
        model_path.parent / "latest_final_vecnormalize.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    globbed = list(model_path.parent.glob("*vecnormalize*.pkl"))
    if not globbed:
        return None
    globbed.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return globbed[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on BossArenaEnv.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/ppo/best_model.zip"))
    parser.add_argument("--vecnormalize-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic.")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2], help="Boss difficulty level.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    cfg = GameConfig()
    apply_level(cfg, args.level)
    base_seed = args.seed

    def _make_env():
        env = BossArenaEnv(cfg=cfg, render_mode=None)
        env.reset(seed=base_seed)
        return env

    raw_env = DummyVecEnv([_make_env])
    raw_env = VecMonitor(raw_env)

    vecnorm_path = find_vecnormalize_path(model_path, args.vecnormalize_path)
    if vecnorm_path is not None:
        eval_env = VecNormalize.load(str(vecnorm_path), raw_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"Loaded VecNormalize stats: {vecnorm_path}")
    else:
        eval_env = raw_env
        print("No VecNormalize stats found; evaluating without normalization wrapper.")

    model = PPO.load(str(model_path), env=eval_env, device=args.device)
    deterministic = not args.stochastic

    wins = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    episode_damage_dealt: list[float] = []
    episode_damage_taken: list[float] = []
    time_to_win_steps: list[int] = []

    obs = eval_env.reset()
    for ep in range(args.episodes):
        done = False
        steps = 0
        ep_reward = 0.0
        dealt = 0.0
        taken = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = eval_env.step(action)
            done = bool(dones[0])
            info = infos[0]
            last_info = info

            steps += 1
            ep_reward += float(rewards[0])
            dealt += float(info.get("damage_dealt_step", 0.0))
            taken += float(info.get("damage_taken_step", 0.0))

        won = bool(last_info.get("win", False))
        wins += int(won)
        if won:
            time_to_win_steps.append(steps)

        episode_lengths.append(steps)
        episode_rewards.append(ep_reward)
        episode_damage_dealt.append(dealt)
        episode_damage_taken.append(taken)

        if ep < args.episodes - 1:
            obs = eval_env.reset()

    eval_env.close()

    win_rate = 100.0 * wins / max(1, args.episodes)
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    avg_dealt = float(np.mean(episode_damage_dealt)) if episode_damage_dealt else 0.0
    avg_taken = float(np.mean(episode_damage_taken)) if episode_damage_taken else 0.0
    avg_ttw = float(np.mean(time_to_win_steps)) if time_to_win_steps else float("nan")

    print("=== Evaluation Summary ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic policy: {deterministic}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Avg episode length (steps): {avg_len:.2f}")
    print(f"Avg episode reward: {avg_reward:.3f}")
    print(f"Avg damage dealt: {avg_dealt:.3f}")
    print(f"Avg damage taken: {avg_taken:.3f}")
    if time_to_win_steps:
        print(f"Avg time-to-win (steps, wins only): {avg_ttw:.2f}")
    else:
        print("Avg time-to-win (steps, wins only): N/A (no wins)")


if __name__ == "__main__":
    main()
