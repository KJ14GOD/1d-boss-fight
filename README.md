# Boss Fight RL Project Roadmap

This repo is for a fast, headless-first RL boss-fight system with parallel training and separate video rendering.

## 1) End Goal

Build and train an agent that can:
- Survive and defeat Boss 1 in a 2D arena.
- Progress through a curriculum of increasing boss difficulty.
- Run training headless and fast (parallel envs).
- Produce visual demo videos in a separate render script.

---

## 2) Core Rules (Do This First)

- Headless simulation first (no pygame in training loop).
- Keep environment deterministic per seed.
- Use Gymnasium API:
  - `reset() -> obs, info`
  - `step() -> obs, reward, terminated, truncated, info`
  - `done = terminated or truncated`
- Separate game logic from rendering logic.

`terminated` vs `truncated`:
- `terminated`: true game ending (player dead or boss dead).
- `truncated`: time/step limit reached (`max_steps`).

---

## 3) Project File Plan

- `game.py`
  - `GameConfig`
  - `Projectile` dataclass
  - `BossArenaEnv`
  - `make_env(seed, cfg)`
- `train.py`
  - SB3 PPO training script (vectorized envs)
- `eval.py`
  - Deterministic evaluation loop (50 episodes)
- `render.py`
  - Load trained model and render one env with pygame
- `preview.py`
  - Open a pygame window with random actions to visually debug game state before RL
- `checkpoints/`
  - `latest.zip`
  - `best.zip`
  - periodic checkpoints (`ckpt_*.zip`)
- `logs/`
  - TensorBoard + eval logs

---

## 4) Phase Checklist

## Phase A: Environment Core (Headless)

- [ ] Implement `GameConfig` constants.
- [ ] Implement environment state:
  - [ ] player position/velocity/hp/cooldowns
  - [ ] boss position/velocity/hp/phase
  - [ ] projectile list
- [ ] Implement deterministic `reset(seed=...)`:
  - [ ] fixed start positions
  - [ ] clear cooldowns
  - [ ] clear projectiles
- [ ] Implement `action_space = MultiDiscrete([9,2,2])`:
  - [ ] move direction (8-dir + idle)
  - [ ] dash on/off
  - [ ] melee on/off
- [ ] Implement boss behavior:
  - [ ] chase + swipe
  - [ ] projectile fan
  - [ ] ring burst (phase-gated)
  - [ ] leap slam (phase-gated)
- [ ] Implement projectile simulation:
  - [ ] velocity integration
  - [ ] lifetime removal
  - [ ] collision with player
- [ ] Implement observation vector (~53-d):
  - [ ] player features
  - [ ] boss features
  - [ ] relative features
  - [ ] global/cooldown/phase features
  - [ ] K nearest projectile features (+ mask)
- [ ] Implement reward:
  - [ ] positive for damage dealt
  - [ ] negative for damage taken
  - [ ] small time penalty each step
  - [ ] terminal win/loss bonus
- [ ] Implement episode endings:
  - [ ] `terminated` for win/loss
  - [ ] `truncated` for `max_steps`
- [ ] Smoke test:
  - [ ] run random policy for 1000 episodes with no crashes

Definition of done:
- Stable headless env, deterministic reset, valid Gymnasium outputs, random rollout works.

---

## Phase B: Parallel Experience Collection

- [ ] Implement `make_env(seed)` thunk.
- [ ] Plug into `SubprocVecEnv` with 8 envs.
- [ ] Test 16 envs if stable.
- [ ] Keep training env headless only.
- [ ] Measure samples/sec vs single env.

Definition of done:
- Throughput is significantly higher than single env and stable over long runs.

---

## Phase C: PPO Baseline (Level 0)

- [ ] Setup SB3 PPO (MLP policy, vector obs).
- [ ] Choose device:
  - [ ] try MPS
  - [ ] fallback CPU if needed
- [ ] Add checkpoint callbacks:
  - [ ] `latest`
  - [ ] `best` (by eval win-rate)
  - [ ] periodic numbered checkpoints
- [ ] Run baseline training on easiest config.

Definition of done:
- Agent survival time increases, then first consistent wins on Level 0.

---

## Phase D: Evaluation Loop (Ground Truth)

- [ ] Build `eval.py`:
  - [ ] load model checkpoint
  - [ ] run 50 headless episodes (single env)
  - [ ] compute win-rate
  - [ ] avg time-to-win
  - [ ] avg damage dealt/taken
- [ ] Save/print eval summary each checkpoint cycle.

Definition of done:
- Promotion decisions are based on eval metrics, not noisy train reward.

---

## Phase E: Curriculum

Level 0:
- [ ] only moves 1-2
- [ ] no phases
- [ ] fan=5
- [ ] slower boss

Promotion rule:
- [ ] evaluate 50 episodes
- [ ] if win-rate >= 70%, move to next level

Next levels:
- [ ] enable ring burst
- [ ] enable phases 1-2
- [ ] increase projectile count/speed
- [ ] enable leap slam + phase 3
- [ ] mix in previous levels to avoid forgetting

Definition of done:
- Agent climbs difficulty levels without restarting from scratch.

---

## Phase F: Render + Video (Separate From Training)

- [ ] Build `render.py`:
  - [ ] load `best` checkpoint
  - [ ] run one env with pygame
  - [ ] optional frame capture to mp4
- [ ] Ensure rendering path does not affect training throughput.

Definition of done:
- You can generate videos of trained policies beating bosses.

---

## Phase G: Add Boss 2+

- [ ] Create new boss pattern sets (timings/moves).
- [ ] Keep same action and observation interface.
- [ ] Reuse training + curriculum + eval pipeline.

Definition of done:
- Multiple boss variants are trainable under same RL stack.

---

## 5) Milestone Order (Recommended)

1. Minimal env that resets and returns valid 53-d obs.
2. Add minimal `step()` with movement + timeout truncation.
3. Add combat + projectiles + rewards.
4. Run 1000 random episodes stability test.
5. Add vectorized training in `train.py`.
6. Add `eval.py` and checkpoint policy selection.
7. Add curriculum promotion loop.
8. Add renderer/video.

---

## 6) Common Failure Checks

- [ ] Observation shape/dtype mismatch.
- [ ] Non-deterministic reset.
- [ ] Accidentally returning `done` instead of `terminated/truncated`.
- [ ] Projectiles never cleaned up (memory leak).
- [ ] Reward scale too extreme (training unstable).
- [ ] Rendering accidentally running during training.

---

## 7) Current Immediate Next Step

In `game.py`, do this first:
- Implement `GameConfig`, `BossArenaEnv.__init__`, `_init_state`, `reset`, and `_get_obs` (dummy zero vector with correct shape).

Then:
- Implement a minimal `step` with:
  - step counter
  - trivial movement update
  - `terminated=False`
  - `truncated = step_count >= max_steps`

Once that runs, add combat and projectile logic incrementally.

---

## 8) Quick Visual Preview

To see the game before training:

```bash
python /Users/kj16/Desktop/game_rl/preview.py
```

If pygame is missing:

```bash
python -m pip install pygame
```

---

## 9) Train / Eval / Render Commands

Train (parallel PPO, 8 envs):

```bash
python /Users/kj16/Desktop/game_rl/train.py \
  --total-timesteps 500000 \
  --num-envs 8 \
  --normalize \
  --device cpu
```

Train with randomized starts (recommended for generalization):

```bash
python /Users/kj16/Desktop/game_rl/train.py \
  --total-timesteps 500000 \
  --num-envs 8 \
  --normalize \
  --train-spawn-jitter 1.5 \
  --eval-spawn-jitter 0.0 \
  --device cpu
```

Evaluate best checkpoint (50 episodes):

```bash
python /Users/kj16/Desktop/game_rl/eval.py \
  --model-path /Users/kj16/Desktop/game_rl/checkpoints/ppo/best_model.zip \
  --episodes 50
```

Evaluate with randomized starts:

```bash
python /Users/kj16/Desktop/game_rl/eval.py \
  --model-path /Users/kj16/Desktop/game_rl/checkpoints/ppo/best_model.zip \
  --episodes 50 \
  --spawn-jitter 1.5
```

Render in a window:

```bash
python /Users/kj16/Desktop/game_rl/render.py \
  --model-path /Users/kj16/Desktop/game_rl/checkpoints/ppo/best_model.zip \
  --episodes 1
```

Render and save mp4:

```bash
python /Users/kj16/Desktop/game_rl/render.py \
  --model-path /Users/kj16/Desktop/game_rl/checkpoints/ppo/best_model.zip \
  --episodes 1 \
  --video-path /Users/kj16/Desktop/game_rl/videos/boss_run.mp4
```

Render 50 randomized episodes and save mp4:

```bash
python /Users/kj16/Desktop/game_rl/render.py \
  --model-path /Users/kj16/Desktop/game_rl/checkpoints/ppo/best_model.zip \
  --episodes 50 \
  --spawn-jitter 1.5 \
  --video-path /Users/kj16/Desktop/game_rl/videos/boss_50eps_randomized.mp4
```

If you save video and are missing dependencies:

```bash
python -m pip install imageio imageio-ffmpeg
```
