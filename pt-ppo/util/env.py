import os
import gym
from gym.spaces.box import Box
import numpy as np

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

def make_env(env_id, rank):
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        if is_atari:
            env = wrap_deepmind(env)
        env.seed(rank)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        # (84, 84, 1) Only black and white
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)
        return env
    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        # (84, 84, 1) -> (1, 84, 84)
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=np.uint8
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
