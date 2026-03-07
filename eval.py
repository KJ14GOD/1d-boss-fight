from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
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
    parser.add_argument(
        "--benchmark-levels",
        type=str,
        default=None,
        help="Comma-separated level ids to benchmark in one run, e.g. 0,1,2. If omitted, only --level is evaluated.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Optional output path for eval JSON. If omitted, a default file is written next to the checkpoint.",
    )
    return parser.parse_args()


def parse_level_ids(level_spec: str | None, fallback_level: int) -> list[int]:
    if level_spec is None:
        return [int(fallback_level)]

    levels: list[int] = []
    for raw in level_spec.split(","):
        token = raw.strip()
        if not token:
            continue
        level_id = int(token)
        if level_id not in (0, 1, 2):
            raise ValueError(f"Invalid benchmark level: {level_id}")
        if level_id not in levels:
            levels.append(level_id)
    if not levels:
        raise ValueError("--benchmark-levels produced an empty level list.")
    return levels


def build_eval_env(model_path: Path, vecnorm_path: Path | None, level_id: int, seed: int):
    cfg = GameConfig()
    apply_level(cfg, level_id)

    def _make_env():
        env = BossArenaEnv(cfg=cfg, render_mode=None)
        env.reset(seed=seed)
        return env

    raw_env = DummyVecEnv([_make_env])
    raw_env = VecMonitor(raw_env)

    if vecnorm_path is not None:
        eval_env = VecNormalize.load(str(vecnorm_path), raw_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = raw_env
    return eval_env


def evaluate_level(model, eval_env, episodes: int, deterministic: bool) -> dict:
    wins = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    episode_damage_dealt: list[float] = []
    episode_damage_taken: list[float] = []
    time_to_win_steps: list[int] = []
    episode_results: list[dict] = []

    obs = eval_env.reset()
    for ep in range(episodes):
        done = False
        steps = 0
        ep_reward = 0.0
        dealt = 0.0
        taken = 0.0
        last_info = {}

        while not done:
            action_masks = get_action_masks(eval_env)
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
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
        episode_results.append(
            {
                "episode": ep + 1,
                "win": won,
                "steps": steps,
                "reward": ep_reward,
                "damage_dealt": dealt,
                "damage_taken": taken,
            }
        )

        if ep < episodes - 1:
            obs = eval_env.reset()

    win_rate = 100.0 * wins / max(1, episodes)
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    avg_dealt = float(np.mean(episode_damage_dealt)) if episode_damage_dealt else 0.0
    avg_taken = float(np.mean(episode_damage_taken)) if episode_damage_taken else 0.0
    avg_ttw = float(np.mean(time_to_win_steps)) if time_to_win_steps else float("nan")

    return {
        "episodes": episodes,
        "wins": wins,
        "win_rate": win_rate,
        "avg_episode_length_steps": avg_len,
        "avg_episode_reward": avg_reward,
        "avg_damage_dealt": avg_dealt,
        "avg_damage_taken": avg_taken,
        "avg_time_to_win_steps": None if np.isnan(avg_ttw) else avg_ttw,
        "episode_results": episode_results,
    }


def default_json_path(model_path: Path, levels: list[int], benchmark_mode: bool) -> Path:
    out_dir = model_path.parent / "eval_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if benchmark_mode:
        level_label = "_".join(f"lvl{level}" for level in levels)
        return out_dir / f"{model_path.stem}_benchmark_{level_label}.json"
    return out_dir / f"{model_path.stem}_level{levels[0]}.json"


def print_level_summary(result: dict, deterministic: bool) -> None:
    print(f"Level: {result['level_id']}")
    print(f"Deterministic policy: {deterministic}")
    print(f"Win rate: {result['win_rate']:.2f}%")
    print(f"Avg episode length (steps): {result['avg_episode_length_steps']:.2f}")
    print(f"Avg episode reward: {result['avg_episode_reward']:.3f}")
    print(f"Avg damage dealt: {result['avg_damage_dealt']:.3f}")
    print(f"Avg damage taken: {result['avg_damage_taken']:.3f}")
    if result["avg_time_to_win_steps"] is None:
        print("Avg time-to-win (steps, wins only): N/A (no wins)")
    else:
        print(f"Avg time-to-win (steps, wins only): {result['avg_time_to_win_steps']:.2f}")


def main():
    args = parse_args()
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    level_ids = parse_level_ids(args.benchmark_levels, args.level)
    benchmark_mode = len(level_ids) > 1
    vecnorm_path = find_vecnormalize_path(model_path, args.vecnormalize_path)

    if vecnorm_path is not None:
        print(f"Loaded VecNormalize stats: {vecnorm_path}")
    else:
        print("No VecNormalize stats found; evaluating without normalization wrapper.")

    deterministic = not args.stochastic
    all_results: list[dict] = []
    model = None

    for idx, level_id in enumerate(level_ids):
        eval_env = build_eval_env(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            level_id=level_id,
            seed=args.seed + idx,
        )
        try:
            if model is None:
                model = MaskablePPO.load(str(model_path), env=eval_env, device=args.device)
            else:
                model.set_env(eval_env)

            level_result = evaluate_level(
                model=model,
                eval_env=eval_env,
                episodes=args.episodes,
                deterministic=deterministic,
            )
            level_result["level_id"] = level_id
            all_results.append(level_result)
        finally:
            eval_env.close()

    if benchmark_mode:
        print("=== Per-Level Benchmark ===")
        for result in all_results:
            print_level_summary(result, deterministic=deterministic)
            print("---")
    else:
        print("=== Evaluation Summary ===")
        print(f"Model: {model_path}")
        print(f"Episodes: {args.episodes}")
        print_level_summary(all_results[0], deterministic=deterministic)

    payload = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path) if vecnorm_path is not None else None,
        "device": args.device,
        "deterministic": deterministic,
        "seed": args.seed,
        "episodes_per_level": args.episodes,
        "benchmark_mode": benchmark_mode,
        "results": all_results,
    }
    json_path = args.json_path or default_json_path(model_path, level_ids, benchmark_mode)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved eval JSON: {json_path}")


if __name__ == "__main__":
    main()
