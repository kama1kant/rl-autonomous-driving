import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import highway_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 100)
        self.action_head = nn.Linear(100, num_actions)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    def __init__(self):
        super(PPO, self).__init__()

    def init_hyp(self):
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_time = 10
        self.buffer_capacity = 1000
        self.batch_size = 32
        self.gamma = 0.99
        self.render = True
        self.seed = 1
        self.log_interval = 10
        self.checkpoint_path = "single-agent/env-highway/ppo/checkpoints/ppo-highway.pth"

    def init_env(self):
        self.env = gym.make('highway-v0')
        self.num_state = self.env.observation_space.shape[0]
        self.num_action = self.env.action_space.n
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        print("obs space = {} action space = {} ".format(self.env.observation_space.shape,
              self.env.action_space.n))

    def create_model(self):
        self.actor_net = Actor(
            np.prod(self.env.observation_space.shape), self.env.action_space.n)
        self.critic_net = Critic(
            np.prod(self.env.observation_space.shape), self.env.action_space.n)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), 3e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor(
            [t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(
                        i_ep, self.training_step))

                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor_net(state[index]).gather(
                    1, action[index])

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                    1 + self.clip_param) * advantage

                action_loss = -torch.min(surr1, surr2).mean()

                print("loss= {}, step = {}".format(
                    action_loss.item(), self.training_step))

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]

    def save_model(self):
        torch.save(self.critic_net, self.checkpoint_path)


def main():
    Transition = namedtuple(
        'Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

    agent = PPO()
    agent.init_hyp()
    agent.init_env()
    agent.create_model()
    for i_epoch in range(1000):
        state = agent.env.reset()
        if agent.render:
            agent.env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if agent.render:
                agent.env.render()
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                    agent.save_model()
                break


# standalone function evaluate model on the environment
def evaluate():
    checkpoint_path = "single-agent/env-highway/ppo/checkpoints/ppo-highway.pth"
    model = torch.load(checkpoint_path)
    env = gym.make("highway-v0")
    state = env.reset()
    done = False

    while done != True:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = model(state)
        c = Categorical(action_prob)
        action = c.sample()
        state, reward, done, info = env.step(action.item())
        env.render()


if __name__ == '__main__':
    main()
    # evaluate()
