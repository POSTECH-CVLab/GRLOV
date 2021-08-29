'''
Code modified from: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
'''

import argparse
from itertools import count

import math
import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer():
    def __init__(self, max_size=1000000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.buffer.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        x, y, a, r, d = [], [], [], [], []

        for i in ind:
            X, Y, A, R, D = self.buffer[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = x * self.max_action 
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, config=None,
                 gamma=0.99, tau=0.005, batch_size=32):
        if config is None:
            self.config = None
            self.gamma = gamma
            self.tau = tau
            self.batch_size = batch_size
        else:
            self.config = config
            self.gamma = config['gamma']
            self.tau = config['tau']
            self.batch_size = config['batch_size']

        self.actor_agent = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_agent = Critic(state_dim, action_dim).to(device)

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor_agent.state_dict())
        self.critic_target.load_state_dict(self.critic_agent.state_dict())

        self.replay_buffer = ReplayBuffer()

        self.actor_optimizer = optim.Adam(self.actor_agent.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic_agent.parameters(), lr=1e-3)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def update_epsilone(self, epoch, EPS_START=10, EPS_END=0, EPS_DECAY=10000):
        if self.config == None:
            self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * epoch / EPS_DECAY)
        else:
            self.epsilon = (self.config['epsilon_start'] - self.config['epsilon_end'])
            self.epsilon *= math.exp(-1. * epoch / self.config['epsilon_decay'])
            self.epsilon += self.config['epsilon_end']
        return

    def epsilon_greedy_action(self, state, dim, low = -1, high = 1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor_agent(state)
        action = action.cpu().data.numpy().flatten()
        action += np.random.normal(0, self.epsilon, size=dim)
        action = action.clip(low, high)
        return action

    def update(self, update_iteration):
        for it in range(update_iteration):
            # Sample from replay buffer
            x, y, a, r, d = self.replay_buffer.sample(self.batch_size)
            curr_state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(a).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            ## Training critic
            self.critic_optimizer.zero_grad()

            critic_Q = self.critic_agent(curr_state, action)
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            critic_loss = F.mse_loss(critic_Q, target_Q)
            critic_loss.backward()
            self.critic_optimizer.step()

            ## Training actor
            self.actor_optimizer.zero_grad()

            actor_loss = self.critic_agent(curr_state, self.actor_agent(curr_state))
            actor_loss = -actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the (frozen) target models
            for agent_param, target_param in zip(self.critic_agent.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * agent_param.data + (1 - self.tau) * target_param.data)

            for agent_param, target_param in zip(self.actor_agent.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * agent_param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def load_weight(self, model_dir="./checkpoints/DDPG_tmp.pth"):
        model = torch.load(model_dir, map_location=device)
        self.actor_agent.load_state_dict(model['actor_agent'])
        self.critic_agent.load_state_dict(model['critic_agent'])
        self.actor_target.load_state_dict(model['actor_target'])
        self.critic_target.load_state_dict(model['critic_target'])
        return

    def save_weight(self, model_dir="./checkpoints/DDPG_tmp.pth"):
        checkpoint = {
            'actor_agent': self.actor_agent.state_dict(),
            'critic_agent': self.critic_agent.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }
        torch.save(checkpoint, model_dir)
        return

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    #agent.load_weight()
    
    total_step, max_reward = 0, -float("inf")
    max_episode = 300
    eval_steps, eval_rewards = [], []
    for i in range(max_episode):
        step, total_reward = 0, 0
        curr_state = env.reset()
        for t in count():
            action = agent.epsilon_greedy_action(curr_state, action_dim, env.action_space.low, env.action_space.high)

            next_state, reward, done, info = env.step(action)
            if total_step % 100 == 0 : env.render()
            agent.replay_buffer.push((curr_state, next_state, action, reward, np.float(done)))

            curr_state = next_state
            if done: break

            step += 1
            total_reward += reward
        total_step += step+1
        eval_steps.append(total_step)
        eval_rewards.append(total_reward)
        agent.update(2000)
        
        if max_reward < total_reward:
            max_reward = total_reward
            agent.save_weight()
            print(f"Total T:{total_step} Episode: \t{i} Total Reward: \t{total_reward:0.2f}")
    
    '''
    plt.figure(figsize=(15, 15))
    plt.title('reward')
    plt.plot(eval_steps, eval_rewards, 'r')
    plt.savefig('./demo/ddpg_example.png)
    '''