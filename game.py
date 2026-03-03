from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
try:
    import pygame
except Exception: 
    pygame = None


@dataclass
class GameConfig:
    width: float = 20.0
    height: float = 20.0
    max_steps: int = 10000
    spawn_jitter: float = 0.0
    min_spawn_distance: float = 6.0

    #player
    player_hp: float = 100.0
    player_speed: float = 0.34
    player_shoot_cd: int = 12
    player_projectile_speed: float = 0.62
    player_projectile_radius: float = 0.20
    player_projectile_damage: float = 6.0

    #boss
    boss_hp: float = 220.0
    boss_speed: float = 0.20
    boss_shoot_cd: int = 8
    boss_projectile_speed: float = 0.78
    boss_projectile_radius: float = 0.26
    boss_projectile_damage: float = 9.0

    #reward
    r_damage_dealt: float = 0.08
    r_damage_taken: float = -0.12
    r_time: float = -0.002
    r_win: float = 20.0
    r_loss: float = -20.0
    r_draw: float = -2.0

    #observation
    k_projectiles: int = 4
    obs_dim: int = 53

    #rendering
    render_width: int = 960
    render_height: int = 576
    render_fps: int = 60


@dataclass
class Projectile:
    pos: np.ndarray  #(x,y)
    vel: np.ndarray  #(vx,vy)
    radius: float #radius of the projectile
    damage: float #damage dealt by the projectile
    owner: int  # 1=player projectile, -1=boss projectile


class BossArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    move_dirs = np.array(
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
        self.cfg = cfg or GameConfig()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode

        # move directions + shoot
        self.action_space = spaces.MultiDiscrete([9, 2])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.obs_dim,),
            dtype=np.float32,
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
        self.player_shoot_cd = 0

        self.boss_pos = np.zeros(2, dtype=np.float32)
        self.boss_vel = np.zeros(2, dtype=np.float32)
        self.boss_hp = self.cfg.boss_hp
        self.boss_shoot_cd = 0

        self.projectiles: List[Projectile] = []
        self._damage_dealt_step = 0.0
        self._damage_taken_step = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_state()

        base_player = np.array([self.cfg.width * 0.25, self.cfg.height * 0.5], dtype=np.float32)
        base_boss = np.array([self.cfg.width * 0.75, self.cfg.height * 0.5], dtype=np.float32)
        self.player_pos[:] = base_player
        self.boss_pos[:] = base_boss

        jitter = float(max(0.0, self.cfg.spawn_jitter))
        if jitter > 0.0:
            for _ in range(24):
                p = base_player + self.np_random.uniform(-jitter, jitter, size=2).astype(np.float32) # add randomness to player pos
                b = base_boss + self.np_random.uniform(-jitter, jitter, size=2).astype(np.float32) # add randomness to boss pos
                p[0] = float(np.clip(p[0], 0.0, self.cfg.width))  # clip to arena width
                p[1] = float(np.clip(p[1], 0.0, self.cfg.height))  # clip to arena height
                b[0] = float(np.clip(b[0], 0.0, self.cfg.width))  # clip to arena width
                b[1] = float(np.clip(b[1], 0.0, self.cfg.height))  # clip to arena height
                if float(np.linalg.norm(b - p)) >= self.cfg.min_spawn_distance:
                    self.player_pos[:] = p
                    self.boss_pos[:] = b
                    break

        obs = self.get_obs()
        info = {"projectiles": 0}
        return obs, info

    def step(self, action):
        self.step_count += 1
        self._damage_dealt_step = 0.0
        self._damage_taken_step = 0.0

        self.apply_player_action(action)
        self.update_boss_ai()
        self.update_projectiles()

        self.player_hp = float(max(0.0, self.player_hp))
        self.boss_hp = float(max(0.0, self.boss_hp))

        terminated = (self.player_hp <= 0.0) or (self.boss_hp <= 0.0)
        truncated = self.step_count >= self.cfg.max_steps
        
        reward = self.compute_reward(terminated, truncated)

        obs = self.get_obs()
        info = {
            "win": self.boss_hp <= 0.0 and self.player_hp > 0.0,
            "boss_hp": float(self.boss_hp),
            "player_hp": float(self.player_hp),
            "damage_dealt_step": float(self._damage_dealt_step),
            "damage_taken_step": float(self._damage_taken_step),
            "projectiles": int(len(self.projectiles)),
        }
        return obs, reward, terminated, truncated, info

    def apply_player_action(self, action) -> None:
        move_idx = int(action[0])
        if int(action[1]) == 1:
            shoot = True
        else:
            shoot = False

        self.player_shoot_cd = max(0, self.player_shoot_cd - 1)

        direction = self.move_dirs[move_idx]
        self.player_vel = direction * self.cfg.player_speed
        self.player_pos += self.player_vel
        self.player_pos[0] = float(np.clip(self.player_pos[0], 0.0, self.cfg.width)) #clip to arena width
        self.player_pos[1] = float(np.clip(self.player_pos[1], 0.0, self.cfg.height)) #clip to arena height
 
        if shoot and self.player_shoot_cd == 0:
            self.spawn_projectile(
                origin=self.player_pos,
                target=self.boss_pos,
                speed=self.cfg.player_projectile_speed,
                radius=self.cfg.player_projectile_radius,
                damage=self.cfg.player_projectile_damage,
                owner=1,
            )
            self.player_shoot_cd = self.cfg.player_shoot_cd

    def update_boss_ai(self) -> None:
        self.boss_shoot_cd = max(0, self.boss_shoot_cd - 1)

        delta = self.player_pos - self.boss_pos
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            self.boss_vel = (delta / dist) * self.cfg.boss_speed
            self.boss_pos += self.boss_vel
            self.boss_pos[0] = float(np.clip(self.boss_pos[0], 0.0, self.cfg.width))
            self.boss_pos[1] = float(np.clip(self.boss_pos[1], 0.0, self.cfg.height))
        else:
            self.boss_vel[:] = 0.0

        if self.boss_shoot_cd == 0:
            self.spawn_projectile(
                origin=self.boss_pos,
                target=self.player_pos,
                speed=self.cfg.boss_projectile_speed,
                radius=self.cfg.boss_projectile_radius,
                damage=self.cfg.boss_projectile_damage,
                owner=-1,
            )
            self.boss_shoot_cd = self.cfg.boss_shoot_cd

    def spawn_projectile(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        speed: float,
        radius: float,
        damage: float,
        owner: int,
    ) -> None:
        delta = target - origin
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = np.array(delta / norm, dtype=np.float32)
        vel = direction * float(speed)
        self.projectiles.append(
            Projectile(
                pos=origin.copy(),
                vel=vel.astype(np.float32),
                radius=float(radius),
                damage=float(damage),
                owner=int(owner),
            )
        )

    def update_projectiles(self) -> None:
        alive: List[Projectile] = []
        for proj in self.projectiles:
            proj.pos = proj.pos + proj.vel

            if proj.pos[0] < -1.0 or proj.pos[0] > self.cfg.width + 1.0:
                continue
            if proj.pos[1] < -1.0 or proj.pos[1] > self.cfg.height + 1.0:
                continue

            if proj.owner == 1:
                # Player projectile hits boss.
                if float(np.linalg.norm(self.boss_pos - proj.pos)) <= proj.radius:
                    self.boss_hp -= proj.damage
                    self._damage_dealt_step += proj.damage
                    continue
            else:
                # Boss projectile hits player.
                if float(np.linalg.norm(self.player_pos - proj.pos)) <= proj.radius:
                    self.player_hp -= proj.damage
                    self._damage_taken_step += proj.damage
                    continue

            alive.append(proj)

        self.projectiles = alive

    def compute_reward(self, terminated: bool, truncated: bool) -> float:
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
        vmax = max(
            self.cfg.player_speed,
            self.cfg.boss_speed,
            self.cfg.player_projectile_speed,
            self.cfg.boss_projectile_speed,
            1e-6,
        )
        return np.array(np.clip(v / vmax, -1.0, 1.0), dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        diag = float(np.sqrt(self.cfg.width ** 2 + self.cfg.height ** 2))
        delta = self.boss_pos - self.player_pos
        dist = float(np.linalg.norm(delta))
        max_proj_dmg = float(max(self.cfg.player_projectile_damage, self.cfg.boss_projectile_damage, 1.0))
        max_proj_radius = float(max(self.cfg.player_projectile_radius, self.cfg.boss_projectile_radius, 1e-6))

        player_block = [
            *self._norm_pos(self.player_pos),
            *self._norm_vel(self.player_vel),
            self._scale01_to_m11(self.player_hp / max(1e-6, self.cfg.player_hp)),
            self._scale01_to_m11(self.player_shoot_cd / max(1, self.cfg.player_shoot_cd)),
            self._scale01_to_m11(float(np.linalg.norm(self.player_vel)) / max(1e-6, self.cfg.player_speed)),
            1.0 if self.player_shoot_cd == 0 else -1.0,
        ]

        boss_block = [
            *self._norm_pos(self.boss_pos),
            *self._norm_vel(self.boss_vel),
            self._scale01_to_m11(self.boss_hp / max(1e-6, self.cfg.boss_hp)),
            self._scale01_to_m11(self.boss_shoot_cd / max(1, self.cfg.boss_shoot_cd)),
            self._scale01_to_m11(dist / max(1e-6, diag)),
        ]

        relative_block = [
            float(np.clip(delta[0] / self.cfg.width, -1.0, 1.0)),
            float(np.clip(delta[1] / self.cfg.height, -1.0, 1.0)),
            float(
                np.clip(
                    (self.boss_vel[0] - self.player_vel[0]) / max(1e-6, self.cfg.player_projectile_speed),
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (self.boss_vel[1] - self.player_vel[1]) / max(1e-6, self.cfg.player_projectile_speed),
                    -1.0,
                    1.0,
                )
            ),
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

        vel_norm_ref = max(self.cfg.player_projectile_speed, self.cfg.boss_projectile_speed, 1e-6)
        for i in range(self.cfg.k_projectiles):
            if i < len(sorted_projectiles):
                p = sorted_projectiles[i]
                rel = p.pos - self.player_pos
                obs.extend(
                    [
                        float(np.clip(rel[0] / self.cfg.width, -1.0, 1.0)),
                        float(np.clip(rel[1] / self.cfg.height, -1.0, 1.0)),
                        float(np.clip(p.vel[0] / vel_norm_ref, -1.0, 1.0)),
                        float(np.clip(p.vel[1] / vel_norm_ref, -1.0, 1.0)),
                        self._scale01_to_m11(p.radius / max_proj_radius),
                        self._scale01_to_m11(p.damage / max_proj_dmg),
                        float(np.clip(p.owner, -1.0, 1.0)),
                        1.0,  # presence mask
                    ]
                )
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0])

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
            color = (120, 255, 120) if p.owner == 1 else (255, 195, 75)
            pygame.draw.circle(surface, color, (px, py), pr)

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
            for _event in pygame.event.get():
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
        if pygame is not None and (self._screen is not None or self._surface is not None):
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
