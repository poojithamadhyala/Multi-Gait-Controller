import gymnasium as gym
import numpy as np
import imageio
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from multi_gait_env import MultiGaitWrapper

os.makedirs("videos", exist_ok=True)

# Load trained model
model = PPO.load("multigait_final")

GAIT_NAMES = {0: "CAUTIOUS", 1: "FAST", 2: "EFFICIENT"}
GAIT_COLORS = {0: "🟣", 1: "🟡", 2: "🟢"}

all_frames = []

for mode in [0, 1, 2]:
    print(f"{GAIT_COLORS[mode]} Recording Mode {mode}: {GAIT_NAMES[mode]}...")

    vec_env = DummyVecEnv([lambda m=mode: MultiGaitWrapper(
        gym.make("HalfCheetah-v5", render_mode="rgb_array"),
        mode=m, randomize_mode=False
    )])
    vec_env = VecNormalize.load("vec_normalize_multigait.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    frames = []

    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
        frame = vec_env.render()
        # Add mode label to frame
        frames.append(frame)
        if done[0]:
            obs = vec_env.reset()

    all_frames.extend(frames)
    vec_env.close()
    print(f"   Captured {len(frames)} frames")

# Save combined video — all 3 gaits back to back
imageio.mimwrite("videos/multigait_demo.mp4", all_frames, fps=30)
print(f"\n✅ Saved videos/multigait_demo.mp4")
print(f"   {len(all_frames)} total frames — Mode 0 → Mode 1 → Mode 2")
