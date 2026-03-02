from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from game import BossArenaEnv, GameConfig, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize


class LatestCheckpointCallback(BaseCallback):
    """Save rolling latest model (and optional VecNormalize stats)."""

    def __init__(self, save_freq: int, save_path: Path, name_prefix: str = "latest", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, save_freq)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        self.save_path.mkdir(parents=True, exist_ok=True)
        model_path = self.save_path / f"{self.name_prefix}.zip"
        self.model.save(str(model_path))

        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(self.save_path / f"{self.name_prefix}_vecnormalize.pkl"))
        return True


def build_level0_config(spawn_jitter: float = 0.0, min_spawn_distance: float = 6.0) -> GameConfig:
    # Easiest curriculum level.
    cfg = GameConfig()
    cfg.enable_phases = False
    cfg.enable_ring = False
    cfg.enable_leap = False
    cfg.fan_count = 5
    cfg.boss_speed = 0.20
    cfg.spawn_jitter = max(0.0, float(spawn_jitter))
    cfg.min_spawn_distance = max(0.0, float(min_spawn_distance))
    return cfg


def create_train_env(
    num_envs: int,
    seed: int,
    cfg: GameConfig,
    normalize: bool,
    start_method: str,
):
    env_fns = [make_env(seed=seed + i, cfg=cfg, render_mode=None) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(vec_env)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def create_eval_env(
    seed: int,
    cfg: GameConfig,
    normalize: bool,
):
    def _make_eval():
        env = BossArenaEnv(cfg=cfg, render_mode=None)
        env.reset(seed=seed)
        return env

    eval_env = DummyVecEnv([_make_eval])
    eval_env = VecMonitor(eval_env)
    if normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    return eval_env


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel PPO training for BossArenaEnv.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument(
        "--train-spawn-jitter",
        type=float,
        default=0.0,
        help="Random start jitter radius (world units) used during training resets.",
    )
    parser.add_argument(
        "--eval-spawn-jitter",
        type=float,
        default=0.0,
        help="Random start jitter radius (world units) used during eval callback resets.",
    )
    parser.add_argument(
        "--min-spawn-distance",
        type=float,
        default=6.0,
        help="Minimum initial distance between player and boss when spawn jitter is enabled.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout steps per env before update.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--normalize", action="store_true", help="Enable VecNormalize on observations/rewards.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate and save best every N environment steps (global steps).",
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save periodic checkpoint every N environment steps (global steps).",
    )
    parser.add_argument(
        "--latest-freq",
        type=int,
        default=10_000,
        help="Overwrite latest checkpoint every N environment steps (global steps).",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="SubprocVecEnv start method. Use spawn on macOS for stability.",
    )
    return parser.parse_args()


def sync_eval_vecnormalize(train_env, eval_env):
    train_vecnorm = train_env if isinstance(train_env, VecNormalize) else None
    eval_vecnorm = eval_env if isinstance(eval_env, VecNormalize) else None
    if train_vecnorm is None or eval_vecnorm is None:
        return
    eval_vecnorm.obs_rms = train_vecnorm.obs_rms
    eval_vecnorm.ret_rms = train_vecnorm.ret_rms


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = build_level0_config(
        spawn_jitter=args.train_spawn_jitter,
        min_spawn_distance=args.min_spawn_distance,
    )
    eval_cfg = build_level0_config(
        spawn_jitter=args.eval_spawn_jitter,
        min_spawn_distance=args.min_spawn_distance,
    )
    train_env = create_train_env(
        num_envs=args.num_envs,
        seed=args.seed,
        cfg=train_cfg,
        normalize=args.normalize,
        start_method=args.start_method,
    )
    eval_env = create_eval_env(
        seed=args.seed + 10_000,
        cfg=eval_cfg,
        normalize=args.normalize,
    )
    sync_eval_vecnormalize(train_env, eval_env)

    has_tensorboard = importlib.util.find_spec("tensorboard") is not None
    tensorboard_log = str(args.log_dir / "tb") if has_tensorboard else None

    # Convert env-step frequencies to callback-step frequencies
    # (callback is called once per vec step).
    periodic_freq = max(1, args.checkpoint_freq // args.num_envs)
    latest_freq = max(1, args.latest_freq // args.num_envs)
    eval_freq = max(1, args.eval_freq // args.num_envs)

    periodic_cb = CheckpointCallback(
        save_freq=periodic_freq,
        save_path=str(args.checkpoint_dir / "periodic"),
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=args.normalize,
    )
    latest_cb = LatestCheckpointCallback(
        save_freq=latest_freq,
        save_path=args.checkpoint_dir,
        name_prefix="latest",
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(args.checkpoint_dir),
        log_path=str(args.log_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )

    run_config = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "normalize": args.normalize,
        "learning_rate": args.learning_rate,
        "train_spawn_jitter": args.train_spawn_jitter,
        "eval_spawn_jitter": args.eval_spawn_jitter,
        "min_spawn_distance": args.min_spawn_distance,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "eval_freq_env_steps": args.eval_freq,
        "checkpoint_freq_env_steps": args.checkpoint_freq,
        "latest_freq_env_steps": args.latest_freq,
        "start_method": args.start_method,
    }
    (args.log_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[periodic_cb, latest_cb, eval_cb],
        )
    finally:
        model.save(str(args.checkpoint_dir / "latest_final.zip"))
        vecnorm = model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(args.checkpoint_dir / "latest_final_vecnormalize.pkl"))

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
