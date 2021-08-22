"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import gym
from gym import wrappers

from utils.seed import set_global_seeds

class ProcessFrameRGB(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def extract_rgb(self, obs):
        return obs[..., :3]

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.extract_rgb(o), r, d, i
    
    def reset(self):
        return self.extract_rgb(self.env.reset())

def get_env(game_name, config, seed, idx_to_save_video=tuple()):
    env = gym.make(game_name, config=config)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './videos'
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda x: (x in idx_to_save_video))

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
