import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiGaitWrapper(gym.Wrapper):
    """
    Wraps HalfCheetah-v5 with a gait mode variable injected into observations.

    Mode 0 — PROWL:   slow, low to ground, stealthy
    Mode 1 — SPRINT:  maximum forward velocity, full power
    Mode 2 — TROT:    smooth efficient gait, low energy
    """

    N_MODES = 3

    def __init__(self, env, mode=None, randomize_mode=True):
        super().__init__(env)
        self.mode = mode
        self.randomize_mode = randomize_mode
        self.current_mode = 0

        orig = env.observation_space
        low  = np.concatenate([orig.low,  np.zeros(self.N_MODES)])
        high = np.concatenate([orig.high, np.ones(self.N_MODES)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _one_hot(self, mode):
        oh = np.zeros(self.N_MODES, dtype=np.float32)
        oh[mode] = 1.0
        return oh

    def _augment_obs(self, obs):
        return np.concatenate([obs, self._one_hot(self.current_mode)])

    def reset(self, **kwargs):
        if self.mode is not None:
            self.current_mode = self.mode
        elif self.randomize_mode:
            self.current_mode = np.random.randint(self.N_MODES)

        obs, info = self.env.reset(**kwargs)
        info["gait_mode"] = self.current_mode
        return self._augment_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        x_vel     = info.get("x_velocity", 0.0)
        ctrl_cost = info.get("reward_ctrl", 0.0)

        if self.current_mode == 0:
            # PROWL — slow and controlled, like stalking prey
            reward = (
                1.0  * max(x_vel, 0)        +   # gentle forward motion
                -1.5 * abs(x_vel - 1.0)     +   # target speed ~1.0 m/s
                3.0  * ctrl_cost                 # very smooth movements
            )

        elif self.current_mode == 1:
            # SPRINT — full speed, push velocity to max
            reward = (
                4.0  * max(x_vel, 0)        +   # max speed reward
                0.5  * ctrl_cost                 # light energy penalty
            )

        else:
            # TROT — efficient, smooth, sustainable pace
            reward = (
                2.0  * max(x_vel, 0)        +   # move forward
                5.0  * ctrl_cost                 # heavy energy penalty
            )

        info["gait_mode"]     = self.current_mode
        info["custom_reward"] = reward
        return self._augment_obs(obs), reward, terminated, truncated, info
