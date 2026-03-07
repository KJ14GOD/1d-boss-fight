from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from game import BossArenaEnv, GameConfig, make_env, apply_level
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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


class CurriculumCallback(BaseCallback):
    """Every eval_freq steps, check win rate. If >= threshold, promote to next level."""

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int,
        win_rate_threshold: float,
        num_envs: int,
        seed: int,
        min_spawn_distance: float,
        normalize: bool,
        start_method: str,
        checkpoint_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_freq = max(1, eval_freq)
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.num_envs = num_envs
        self.seed = seed
        self.min_spawn_distance = min_spawn_distance
        self.normalize = normalize
        self.start_method = start_method
        self.checkpoint_dir = Path(checkpoint_dir)
        self.current_level = 0
        self.max_level = 2
        self.best_mean_reward = -float("inf")
        self.promoted = False
        self._eval_env = None

    def _init_eval_env(self):
        if self._eval_env is not None:
            self._eval_env.close()
        cfg = build_config(self.current_level, self.min_spawn_distance)
        self._eval_env = create_eval_env(
            seed=self.seed + 10_000, cfg=cfg, normalize=self.normalize,
        )

    def _on_training_start(self):
        self._init_eval_env()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        sync_eval_vecnormalize(self.model.get_env(), self._eval_env)
        win_rate, mean_reward, mean_ep_len = evaluate(self.model, self._eval_env, self.eval_episodes)

        print(f"\n  [Curriculum] Level {self.current_level} | Win rate: {win_rate:.0%} | "
              f"Mean reward: {mean_reward:.2f} | Mean ep length: {mean_ep_len:.0f}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.checkpoint_dir / "best_model.zip"))
            print(f"  [Curriculum] New best reward: {mean_reward:.2f} — saved best_model.zip")

        if win_rate >= self.win_rate_threshold and self.current_level < self.max_level:
            save_path = self.checkpoint_dir / f"level{self.current_level}_pass.zip"
            self.model.save(str(save_path))
            print(f"  [Curriculum] Level {self.current_level} PASSED! Saved {save_path}")

            self.current_level += 1
            self.promoted = True
            print(f"  [Curriculum] Promoting to level {self.current_level} — stopping to swap envs...")
            return False  # stop model.learn() so main() can swap envs safely

        elif win_rate >= self.win_rate_threshold and self.current_level >= self.max_level:
            save_path = self.checkpoint_dir / f"level{self.current_level}_pass.zip"
            self.model.save(str(save_path))
            print(f"  [Curriculum] ALL LEVELS COMPLETE! Saved {save_path}")

        return True

    def _on_training_end(self):
        if self._eval_env is not None:
            self._eval_env.close()


def build_config(level_id: int, min_spawn_distance: float = 6.0) -> GameConfig:
    cfg = GameConfig()
    cfg.min_spawn_distance = max(0.0, float(min_spawn_distance))
    apply_level(cfg, level_id)
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
    parser.add_argument("--win-rate-threshold", type=float, default=0.70,
                        help="Win rate to promote to next curriculum level.")
    return parser.parse_args()


def sync_eval_vecnormalize(train_env, eval_env):
    train_vecnorm = train_env if isinstance(train_env, VecNormalize) else None
    eval_vecnorm = eval_env if isinstance(eval_env, VecNormalize) else None
    if train_vecnorm is None or eval_vecnorm is None:
        return
    eval_vecnorm.obs_rms = train_vecnorm.obs_rms
    eval_vecnorm.ret_rms = train_vecnorm.ret_rms


def evaluate(model, eval_env, n_episodes: int = 50) -> tuple:
    """Returns (win_rate, mean_reward, mean_ep_length)."""
    wins = 0
    total_rewards = []
    total_lengths = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action_masks = get_action_masks(eval_env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, rewards, dones, infos = eval_env.step(action)
            ep_reward += rewards[0]
            ep_length += 1
            if dones[0]:
                done = True
                if infos[0].get("win", False):
                    wins += 1
        total_rewards.append(ep_reward)
        total_lengths.append(ep_length)
    win_rate = wins / n_episodes
    mean_reward = sum(total_rewards) / n_episodes
    mean_ep_length = sum(total_lengths) / n_episodes
    return win_rate, mean_reward, mean_ep_length


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = build_config(level_id=0, min_spawn_distance=args.min_spawn_distance)
    train_env = create_train_env(
        num_envs=args.num_envs,
        seed=args.seed,
        cfg=train_cfg,
        normalize=args.normalize,
        start_method=args.start_method,
    )

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
    curriculum_cb = CurriculumCallback(
        eval_freq=eval_freq,
        eval_episodes=args.eval_episodes,
        win_rate_threshold=args.win_rate_threshold,
        num_envs=args.num_envs,
        seed=args.seed,
        min_spawn_distance=args.min_spawn_distance,
        normalize=args.normalize,
        start_method=args.start_method,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = MaskablePPO(
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
        while True:
            remaining = max(0, args.total_timesteps - model.num_timesteps)
            if remaining == 0:
                break
            model.learn(
                total_timesteps=remaining,
                callback=[periodic_cb, latest_cb, curriculum_cb],
                reset_num_timesteps=False,
            )

            if not curriculum_cb.promoted:
                break  # training finished normally (hit total_timesteps or all levels done)

            # swap envs for the new level
            curriculum_cb.promoted = False
            train_env.close()

            cfg = build_config(curriculum_cb.current_level, args.min_spawn_distance)
            train_env = create_train_env(
                num_envs=args.num_envs, seed=args.seed, cfg=cfg,
                normalize=args.normalize, start_method=args.start_method,
            )
            model.set_env(train_env)
            curriculum_cb._init_eval_env()

    finally:
        model.save(str(args.checkpoint_dir / "latest_final.zip"))
        vecnorm = model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(args.checkpoint_dir / "latest_final_vecnormalize.pkl"))

        train_env.close()


if __name__ == "__main__":
    main()
