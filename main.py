import numpy as np
from itertools import count
import random
import gym.spaces
from tqdm import tqdm
import gym
from utils.gym import get_env, get_wrapper_by_name
from utils.plot import plot_line
from manipulation_main.common import io_utils
import logging
logging.getLogger().setLevel(logging.DEBUG) 

# Hyperparameters
config = io_utils.load_yaml("config/gripper_grasp.yaml")
env = get_env("gripper-env-v0", config=config, seed=0, idx_to_save_video=(0,))
gamma=0.99
total_timestep=100_000

assert type(env.observation_space) == gym.spaces.Box
assert type(env.action_space)      == gym.spaces.Box

###############
# BUILD MODEL #
###############
img_h, img_w, img_c = 64, 64, 3
action_shape = env.action_space.shape
action_min = env.action_space.low
action_max = env.action_space.high

###############
# RUN ENV     #
###############
num_param_updates = 0
mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')
last_obs = env.reset()
log_every_n_steps = 1000
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

with tqdm(total=total_timestep) as pbar:
    for t in count():
        # update progress bar
        pbar.n = t
        pbar.refresh()

        ### Check stopping criterion
        if t >= total_timestep:
            break

        action = np.random.rand(5)
        action = action * (action_max - action_min) + action_min
        obs, reward, done, _ = env.step(action)
        # normalize rewards between -1 and 1, -200 < reward < 10000
        reward = reward / 10000.0
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % log_every_n_steps == 0:
            tqdm.write("=========================================")
            tqdm.write("Timestep %d" % (t,))
            tqdm.write("mean reward (100 episodes) %f" % mean_episode_reward)
            tqdm.write("best mean reward %f" % best_mean_episode_reward)
            tqdm.write("episodes %d" % len(episode_rewards))
            tqdm.write("=========================================")
            # Test Q-values over validation memory
            metrics['rewards'].append([np.mean(episode_rewards[-5:])])
            metrics['steps'].append(t)

            plot_line(metrics['steps'], metrics['rewards'], 'Reward')
