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
    parser = argparse.ArgumentParser(description="Render PPO policy in BossArenaEnv.")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/ppo/best_model.zip"))
    parser.add_argument("--vecnormalize-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic.")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional output video path, e.g. videos/boss_run.mp4. If omitted, opens human window.",
    )
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2], help="Boss difficulty level.")
    return parser.parse_args()


def save_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames captured; nothing to write.")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise ImportError(
            "imageio is required to save videos. Install with: python -m pip install imageio imageio-ffmpeg"
        ) from exc

    with imageio.get_writer(str(video_path), fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    args = parse_args()
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    render_mode = "rgb_array" if args.video_path is not None else "human"
    cfg = GameConfig()
    apply_level(cfg, args.level)
    base_seed = args.seed

    def _make_env():
        env = BossArenaEnv(cfg=cfg, render_mode=render_mode)
        env.reset(seed=base_seed)
        return env

    raw_env = DummyVecEnv([_make_env])
    raw_env = VecMonitor(raw_env)
    base_env = raw_env.envs[0]

    vecnorm_path = find_vecnormalize_path(model_path, args.vecnormalize_path)
    if vecnorm_path is not None:
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats: {vecnorm_path}")
    else:
        env = raw_env
        print("No VecNormalize stats found; rendering without normalization wrapper.")

    model = PPO.load(str(model_path), env=env, device=args.device)
    deterministic = not args.stochastic
    print(f"Render mode: {render_mode} | deterministic_policy={deterministic}")

    obs = env.reset()
    frames: list[np.ndarray] = []
    wins = 0
    lengths: list[int] = []
    frame_count = 0

    for ep in range(args.episodes):
        done = False
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            last_info = infos[0]
            steps += 1

            frame = base_env.render()
            if args.video_path is not None and frame is not None:
                frames.append(frame)
                frame_count += 1
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    done = True

        won = bool(last_info.get("win", False))
        wins += int(won)
        lengths.append(steps)
        print(f"Episode {ep + 1}/{args.episodes} | win={won} | steps={steps}")

        if ep < args.episodes - 1:
            obs = env.reset()

    env.close()

    win_rate = 100.0 * wins / max(1, args.episodes)
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    print(f"Summary | win_rate={win_rate:.2f}% | avg_steps={avg_len:.2f}")

    if args.video_path is not None:
        save_video(args.video_path, frames, args.video_fps)
        print(f"Saved video: {args.video_path}")


if __name__ == "__main__":
    main()
