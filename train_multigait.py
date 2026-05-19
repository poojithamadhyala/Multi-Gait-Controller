import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from multi_gait_env import MultiGaitWrapper

LOG_DIR  = "./logs/"
SAVE_DIR = "./checkpoints/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def make_env(rank, mode=None):
    def _init():
        env = gym.make("HalfCheetah-v5")
        env = MultiGaitWrapper(env, mode=mode, randomize_mode=(mode is None))
        return env
    return _init

if __name__ == '__main__':

    # Training env — randomizes gait mode each episode
    train_env = SubprocVecEnv([make_env(i) for i in range(4)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Eval envs — one per gait mode so we track each separately
    eval_env_0 = VecNormalize(SubprocVecEnv([make_env(0, mode=0)]), norm_obs=True, norm_reward=False, training=False)
    eval_env_1 = VecNormalize(SubprocVecEnv([make_env(0, mode=1)]), norm_obs=True, norm_reward=False, training=False)
    eval_env_2 = VecNormalize(SubprocVecEnv([make_env(0, mode=2)]), norm_obs=True, norm_reward=False, training=False)

    eval_cb_0 = EvalCallback(eval_env_0, best_model_save_path=SAVE_DIR+"mode0/",
                              log_path=LOG_DIR+"mode0/", eval_freq=20000, n_eval_episodes=5)
    eval_cb_1 = EvalCallback(eval_env_1, best_model_save_path=SAVE_DIR+"mode1/",
                              log_path=LOG_DIR+"mode1/", eval_freq=20000, n_eval_episodes=5)
    eval_cb_2 = EvalCallback(eval_env_2, best_model_save_path=SAVE_DIR+"mode2/",
                              log_path=LOG_DIR+"mode2/", eval_freq=20000, n_eval_episodes=5)

    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=SAVE_DIR,
        name_prefix="multigait"
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("🚀 Training Multi-Gait Controller...")
    print("   Mode 0: Cautious  |  Mode 1: Fast  |  Mode 2: Efficient")
    print(f"   Observation space: {train_env.observation_space.shape}\n")

    model.learn(
        total_timesteps=3_000_000,
        callback=[eval_cb_0, eval_cb_1, eval_cb_2, ckpt_cb],
        progress_bar=True
    )

    model.save("multigait_final")
    train_env.save("vec_normalize_multigait.pkl")
    print("✅ Training complete!")
