import time
from game import BossArenaEnv, GameConfig


def main():
    cfg = GameConfig()
    env = BossArenaEnv(cfg=cfg, render_mode="human")
    obs, info = env.reset(seed=0)
    print("Preview started. Close the window or Ctrl+C to stop.")

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                print(
                    f"Episode done | win={info.get('win')} | "
                    f"player_hp={info.get('player_hp'):.1f} boss_hp={info.get('boss_hp'):.1f}"
                )
                obs, info = env.reset()
                time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
