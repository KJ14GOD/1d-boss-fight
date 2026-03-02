from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import pygame
except Exception:  # pragma: no cover - optional dependency
    pygame = None


@dataclass
class GameConfig:
    # Arena + episode
    width: float = 20.0
    height: float = 12.0
    max_steps: int = 1200
    spawn_jitter: float = 0.0
    min_spawn_distance: float = 6.0

    # Player
    player_hp: float = 100.0
    player_speed: float = 0.35
    dash_speed: float = 1.1
    dash_cd: int = 10
    melee_range: float = 1.6
    melee_damage: float = 8.0
    melee_cd: int = 12

    # Boss
    boss_hp: float = 150.0
    boss_speed: float = 0.24
    swipe_range: float = 1.4
    swipe_damage: float = 6.0
    swipe_cd: int = 18
    fan_count: int = 5
    fan_cd: int = 28
    fan_spread_rad: float = 0.9
    fan_speed: float = 0.46
    fan_life: int = 85
    projectile_radius: float = 0.22
    projectile_damage: float = 4.0
    enable_ring: bool = False
    ring_count: int = 12
    ring_speed: float = 0.36
    ring_life: int = 85
    ring_cd: int = 55
    enable_leap: bool = False
    leap_cd: int = 75
    leap_radius: float = 1.5
    leap_damage: float = 9.0
    enable_phases: bool = False

    # Reward (taken penalty stronger to discourage reckless trading)
    r_damage_dealt: float = 0.08
    r_damage_taken: float = -0.12
    r_time: float = -0.002
    r_win: float = 20.0
    r_loss: float = -20.0
    r_draw: float = -2.0

    # Observation
    k_projectiles: int = 4  # nearest K in obs
    obs_dim: int = 53

    # Rendering
    render_width: int = 960
    render_height: int = 576
    render_fps: int = 60


@dataclass
class Projectile:
    pos: np.ndarray  # (2,)
    vel: np.ndarray  # (2,)
    life: int
    radius: float
    damage: float


class BossArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    _MOVE_DIRS = np.array(
        [
            [0.0, 0.0],  # idle
            [0.0, -1.0],  # up
            [0.0, 1.0],  # down
            [-1.0, 0.0],  # left
            [1.0, 0.0],  # right
            [-0.70710677, -0.70710677],  # up-left
            [0.70710677, -0.70710677],  # up-right
            [-0.70710677, 0.70710677],  # down-left
            [0.70710677, 0.70710677],  # down-right
        ],
        dtype=np.float32,
    )

    def __init__(self, cfg: Optional[GameConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or GameConfig()

        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete([9, 2, 2])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.obs_dim,), dtype=np.float32
        )

        self._screen = None
        self._clock = None
        self._surface = None

        self._init_state()

    def _init_state(self) -> None:
        self.step_count = 0
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_hp = self.cfg.player_hp
        self.player_dash_cd = 0
        self.player_melee_cd = 0

        self.boss_pos = np.zeros(2, dtype=np.float32)
        self.boss_vel = np.zeros(2, dtype=np.float32)
        self.boss_hp = self.cfg.boss_hp
        self.boss_phase = 1
        self.boss_swipe_cd = 0
        self.boss_fan_cd = 0
        self.boss_ring_cd = 0
        self.boss_leap_cd = 0

        self.projectiles: List[Projectile] = []

        self._damage_dealt_step = 0.0
        self._damage_taken_step = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_state()

        # deterministic starts
        base_player = np.array([self.cfg.width * 0.25, self.cfg.height * 0.5], dtype=np.float32)
        base_boss = np.array([self.cfg.width * 0.75, self.cfg.height * 0.5], dtype=np.float32)
        self.player_pos[:] = base_player
        self.boss_pos[:] = base_boss

        jitter = float(max(0.0, self.cfg.spawn_jitter))
        if jitter > 0.0:
            # Randomized starts for generalization testing.
            for _ in range(24):
                p = base_player + self.np_random.uniform(-jitter, jitter, size=2).astype(np.float32)
                b = base_boss + self.np_random.uniform(-jitter, jitter, size=2).astype(np.float32)
                p[0] = float(np.clip(p[0], 0.0, self.cfg.width))
                p[1] = float(np.clip(p[1], 0.0, self.cfg.height))
                b[0] = float(np.clip(b[0], 0.0, self.cfg.width))
                b[1] = float(np.clip(b[1], 0.0, self.cfg.height))
                if float(np.linalg.norm(b - p)) >= self.cfg.min_spawn_distance:
                    self.player_pos[:] = p
                    self.boss_pos[:] = b
                    break

        obs = self._get_obs()
        info = {"phase": self.boss_phase}
        return obs, info

    def step(self, action):
        self.step_count += 1
        self._damage_dealt_step = 0.0
        self._damage_taken_step = 0.0

        self._apply_player_action(action)
        self._update_boss_ai()
        self._damage_taken_step += self._update_projectiles()

        self.player_hp = float(max(0.0, self.player_hp))
        self.boss_hp = float(max(0.0, self.boss_hp))

        terminated = (self.player_hp <= 0.0) or (self.boss_hp <= 0.0)
        truncated = self.step_count >= self.cfg.max_steps
        reward = self._compute_reward(terminated, truncated)

        obs = self._get_obs()
        info = {
            "win": self.boss_hp <= 0.0 and self.player_hp > 0.0,
            "boss_hp": float(self.boss_hp),
            "player_hp": float(self.player_hp),
            "damage_dealt_step": float(self._damage_dealt_step),
            "damage_taken_step": float(self._damage_taken_step),
            "phase": int(self.boss_phase),
            "projectiles": int(len(self.projectiles)),
        }
        return obs, reward, terminated, truncated, info

    def _apply_player_action(self, action) -> None:
        move_idx = int(action[0])
        dash = int(action[1]) == 1
        melee = int(action[2]) == 1

        self.player_dash_cd = max(0, self.player_dash_cd - 1)
        self.player_melee_cd = max(0, self.player_melee_cd - 1)

        speed = self.cfg.player_speed
        if dash and self.player_dash_cd == 0:
            speed = self.cfg.dash_speed
            self.player_dash_cd = self.cfg.dash_cd

        direction = self._MOVE_DIRS[move_idx]
        self.player_vel = direction * speed
        self.player_pos += self.player_vel
        self.player_pos[0] = float(np.clip(self.player_pos[0], 0.0, self.cfg.width))
        self.player_pos[1] = float(np.clip(self.player_pos[1], 0.0, self.cfg.height))

        if melee and self.player_melee_cd == 0:
            dist = float(np.linalg.norm(self.boss_pos - self.player_pos))
            if dist <= self.cfg.melee_range:
                self.boss_hp -= self.cfg.melee_damage
                self._damage_dealt_step += self.cfg.melee_damage
            self.player_melee_cd = self.cfg.melee_cd

    def _update_boss_ai(self) -> None:
        self.boss_swipe_cd = max(0, self.boss_swipe_cd - 1)
        self.boss_fan_cd = max(0, self.boss_fan_cd - 1)
        self.boss_ring_cd = max(0, self.boss_ring_cd - 1)
        self.boss_leap_cd = max(0, self.boss_leap_cd - 1)

        if self.cfg.enable_phases:
            hp_ratio = self.boss_hp / max(1e-6, self.cfg.boss_hp)
            if hp_ratio <= 0.33:
                self.boss_phase = 3
            elif hp_ratio <= 0.66:
                self.boss_phase = 2
            else:
                self.boss_phase = 1
        else:
            self.boss_phase = 1

        delta = self.player_pos - self.boss_pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            self.boss_vel = (delta / dist) * self.cfg.boss_speed
            self.boss_pos += self.boss_vel
            self.boss_pos[0] = float(np.clip(self.boss_pos[0], 0.0, self.cfg.width))
            self.boss_pos[1] = float(np.clip(self.boss_pos[1], 0.0, self.cfg.height))
        else:
            self.boss_vel[:] = 0.0

        if dist <= self.cfg.swipe_range and self.boss_swipe_cd == 0:
            self.player_hp -= self.cfg.swipe_damage
            self._damage_taken_step += self.cfg.swipe_damage
            self.boss_swipe_cd = self.cfg.swipe_cd

        if self.boss_fan_cd == 0:
            self._spawn_fan_projectiles()
            self.boss_fan_cd = self.cfg.fan_cd

        if self.cfg.enable_ring and self.boss_phase >= 2 and self.boss_ring_cd == 0:
            self._spawn_ring_projectiles()
            self.boss_ring_cd = self.cfg.ring_cd

        if self.cfg.enable_leap and self.boss_phase >= 3 and self.boss_leap_cd == 0:
            self._do_leap_slam()
            self.boss_leap_cd = self.cfg.leap_cd

    def _spawn_fan_projectiles(self) -> None:
        delta = self.player_pos - self.boss_pos
        base = float(np.arctan2(delta[1], delta[0]))
        n = max(1, int(self.cfg.fan_count))
        if n == 1:
            angles = [base]
        else:
            angles = np.linspace(
                base - self.cfg.fan_spread_rad / 2.0,
                base + self.cfg.fan_spread_rad / 2.0,
                n,
            )

        for angle in angles:
            vel = np.array(
                [np.cos(angle) * self.cfg.fan_speed, np.sin(angle) * self.cfg.fan_speed],
                dtype=np.float32,
            )
            self.projectiles.append(
                Projectile(
                    pos=self.boss_pos.copy(),
                    vel=vel,
                    life=int(self.cfg.fan_life),
                    radius=float(self.cfg.projectile_radius),
                    damage=float(self.cfg.projectile_damage),
                )
            )

    def _spawn_ring_projectiles(self) -> None:
        n = max(1, int(self.cfg.ring_count))
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        for angle in angles:
            vel = np.array(
                [np.cos(angle) * self.cfg.ring_speed, np.sin(angle) * self.cfg.ring_speed],
                dtype=np.float32,
            )
            self.projectiles.append(
                Projectile(
                    pos=self.boss_pos.copy(),
                    vel=vel,
                    life=int(self.cfg.ring_life),
                    radius=float(self.cfg.projectile_radius),
                    damage=float(self.cfg.projectile_damage),
                )
            )

    def _do_leap_slam(self) -> None:
        offset = np.array(self.np_random.uniform(-0.7, 0.7, size=2), dtype=np.float32)
        target = self.player_pos + offset
        target[0] = float(np.clip(target[0], 0.0, self.cfg.width))
        target[1] = float(np.clip(target[1], 0.0, self.cfg.height))
        self.boss_pos[:] = target

        dist = float(np.linalg.norm(self.player_pos - self.boss_pos))
        if dist <= self.cfg.leap_radius:
            self.player_hp -= self.cfg.leap_damage
            self._damage_taken_step += self.cfg.leap_damage

    def _update_projectiles(self) -> float:
        damage_taken = 0.0
        alive: List[Projectile] = []

        for proj in self.projectiles:
            proj.pos = proj.pos + proj.vel
            proj.life -= 1

            if proj.life <= 0:
                continue
            if proj.pos[0] < -1.0 or proj.pos[0] > self.cfg.width + 1.0:
                continue
            if proj.pos[1] < -1.0 or proj.pos[1] > self.cfg.height + 1.0:
                continue

            dist = float(np.linalg.norm(self.player_pos - proj.pos))
            if dist <= proj.radius:
                self.player_hp -= proj.damage
                damage_taken += proj.damage
                continue

            alive.append(proj)

        self.projectiles = alive
        return float(damage_taken)

    def _compute_reward(self, terminated: bool, truncated: bool) -> float:
        reward = self.cfg.r_time
        reward += self.cfg.r_damage_dealt * self._damage_dealt_step
        reward += self.cfg.r_damage_taken * self._damage_taken_step

        if terminated:
            if self.boss_hp <= 0.0 and self.player_hp > 0.0:
                reward += self.cfg.r_win
            elif self.player_hp <= 0.0 and self.boss_hp > 0.0:
                reward += self.cfg.r_loss
            else:
                reward += self.cfg.r_draw
        elif truncated:
            reward += self.cfg.r_draw

        return float(reward)

    @staticmethod
    def _scale01_to_m11(value_01: float) -> float:
        return float(np.clip(value_01 * 2.0 - 1.0, -1.0, 1.0))

    def _norm_pos(self, p: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip((p[0] / self.cfg.width) * 2.0 - 1.0, -1.0, 1.0),
                np.clip((p[1] / self.cfg.height) * 2.0 - 1.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _norm_vel(self, v: np.ndarray) -> np.ndarray:
        vmax = max(self.cfg.dash_speed, self.cfg.fan_speed, self.cfg.ring_speed, 1e-6)
        return np.array(np.clip(v / vmax, -1.0, 1.0), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        diag = float(np.sqrt(self.cfg.width ** 2 + self.cfg.height ** 2))
        delta = self.boss_pos - self.player_pos
        dist = float(np.linalg.norm(delta))

        player_block = [
            *self._norm_pos(self.player_pos),
            *self._norm_vel(self.player_vel),
            self._scale01_to_m11(self.player_hp / max(1e-6, self.cfg.player_hp)),
            self._scale01_to_m11(self.player_dash_cd / max(1, self.cfg.dash_cd)),
            self._scale01_to_m11(self.player_melee_cd / max(1, self.cfg.melee_cd)),
            self._scale01_to_m11(float(np.linalg.norm(self.player_vel)) / max(1e-6, self.cfg.dash_speed)),
        ]

        phase_norm = (self.boss_phase - 1) / 2.0
        boss_block = [
            *self._norm_pos(self.boss_pos),
            *self._norm_vel(self.boss_vel),
            self._scale01_to_m11(self.boss_hp / max(1e-6, self.cfg.boss_hp)),
            self._scale01_to_m11(phase_norm),
            self._scale01_to_m11(dist / max(1e-6, diag)),
        ]

        relative_block = [
            float(np.clip(delta[0] / self.cfg.width, -1.0, 1.0)),
            float(np.clip(delta[1] / self.cfg.height, -1.0, 1.0)),
            float(np.clip((self.boss_vel[0] - self.player_vel[0]) / max(1e-6, self.cfg.dash_speed), -1.0, 1.0)),
            float(np.clip((self.boss_vel[1] - self.player_vel[1]) / max(1e-6, self.cfg.dash_speed), -1.0, 1.0)),
        ]

        global_block = [
            self._scale01_to_m11(self.step_count / max(1, self.cfg.max_steps)),
            self._scale01_to_m11(min(1.0, len(self.projectiles) / 25.0)),
        ]

        obs = list(player_block) + list(boss_block) + list(relative_block) + list(global_block)
        assert len(obs) == 21

        sorted_projectiles = sorted(
            self.projectiles,
            key=lambda p: float(np.linalg.norm(p.pos - self.player_pos)),
        )

        max_life = float(max(self.cfg.fan_life, self.cfg.ring_life, 1))
        for i in range(self.cfg.k_projectiles):
            if i < len(sorted_projectiles):
                p = sorted_projectiles[i]
                rel = p.pos - self.player_pos
                obs.extend(
                    [
                        float(np.clip(rel[0] / self.cfg.width, -1.0, 1.0)),
                        float(np.clip(rel[1] / self.cfg.height, -1.0, 1.0)),
                        float(np.clip(p.vel[0] / max(1e-6, self.cfg.dash_speed), -1.0, 1.0)),
                        float(np.clip(p.vel[1] / max(1e-6, self.cfg.dash_speed), -1.0, 1.0)),
                        self._scale01_to_m11(p.life / max_life),
                        self._scale01_to_m11(min(1.0, p.radius / 1.0)),
                        self._scale01_to_m11(min(1.0, p.damage / 15.0)),
                        1.0,  # presence mask
                    ]
                )
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0])

        out = np.array(obs, dtype=np.float32)
        if out.shape[0] != self.cfg.obs_dim:
            raise ValueError(f"Observation size mismatch: got {out.shape[0]}, expected {self.cfg.obs_dim}")
        return np.clip(out, -1.0, 1.0)

    def _world_to_screen(self, world_pos: np.ndarray) -> tuple[int, int]:
        sx = int((world_pos[0] / self.cfg.width) * self.cfg.render_width)
        sy = int((world_pos[1] / self.cfg.height) * self.cfg.render_height)
        return sx, sy

    def _draw_scene(self, surface) -> None:
        surface.fill((22, 24, 30))

        arena_rect = pygame.Rect(0, 0, self.cfg.render_width, self.cfg.render_height)
        pygame.draw.rect(surface, (35, 39, 48), arena_rect, width=4)

        player_xy = self._world_to_screen(self.player_pos)
        boss_xy = self._world_to_screen(self.boss_pos)
        player_r = max(4, int((0.35 / self.cfg.width) * self.cfg.render_width))
        boss_r = max(6, int((0.52 / self.cfg.width) * self.cfg.render_width))

        pygame.draw.circle(surface, (80, 200, 255), player_xy, player_r)
        pygame.draw.circle(surface, (255, 90, 90), boss_xy, boss_r)

        for p in self.projectiles:
            px, py = self._world_to_screen(p.pos)
            pr = max(2, int((p.radius / self.cfg.width) * self.cfg.render_width))
            pygame.draw.circle(surface, (255, 195, 75), (px, py), pr)

        hp_h = 14
        pad = 12
        p_ratio = float(np.clip(self.player_hp / max(1e-6, self.cfg.player_hp), 0.0, 1.0))
        b_ratio = float(np.clip(self.boss_hp / max(1e-6, self.cfg.boss_hp), 0.0, 1.0))
        pygame.draw.rect(surface, (60, 60, 60), pygame.Rect(pad, pad, 240, hp_h))
        pygame.draw.rect(surface, (80, 200, 255), pygame.Rect(pad, pad, int(240 * p_ratio), hp_h))
        pygame.draw.rect(
            surface,
            (60, 60, 60),
            pygame.Rect(self.cfg.render_width - 252, pad, 240, hp_h),
        )
        pygame.draw.rect(
            surface,
            (255, 90, 90),
            pygame.Rect(self.cfg.render_width - 252, pad, int(240 * b_ratio), hp_h),
        )

    def render(self):
        if self.render_mode is None:
            return None
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        if self.render_mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode((self.cfg.render_width, self.cfg.render_height))
                pygame.display.set_caption("Boss Arena Preview")
                self._clock = pygame.time.Clock()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            self._draw_scene(self._screen)
            pygame.display.flip()
            self._clock.tick(self.cfg.render_fps)
            return None

        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.cfg.render_width, self.cfg.render_height))
        self._draw_scene(self._surface)
        frame = pygame.surfarray.array3d(self._surface).transpose(1, 0, 2)
        return frame

    def close(self):
        if pygame is not None:
            if self._screen is not None or self._surface is not None:
                pygame.quit()
        self._screen = None
        self._surface = None
        self._clock = None


def make_env(seed: int, cfg: Optional[GameConfig] = None, render_mode: Optional[str] = None):
    # for SubprocVecEnv
    def _thunk():
        env = BossArenaEnv(cfg=cfg, render_mode=render_mode)
        env.reset(seed=seed)
        return env

    return _thunk
