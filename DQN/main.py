import sys; sys.path.append('./')

import random
from itertools import count
from copy import deepcopy
from typing import Union

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.optim import Adam
from torchsummary import summary

from blocks import Agent, Environment, ReplayBuffer
from utils import plot_returns, test


# Hyperparameters -> This part can later be changed to get inputs from argparser
NAME = 'DQN'

# Environment
ENV_NAME = "CartPole-v0"
RENDER_MODE = None

# Agent
REPLAY_SIZE = 100000
TAU = 0.1
EPS = 1
EPS_DECAY = 0.999
EPS_MIN = 0.001
HIDDEN1 = 400
HIDDEN2 = 300

# Training
EPISODES = 10000
BATCH_SIZE = 500
GAMMA = 0.99
LR = 0.0001

# CUDA
DEVICE = 'cuda' if cuda.is_available() else 'cpu'


class QNet(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()

        self.layer1 = nn.Linear(n_in, HIDDEN1)
        self.layer2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.layer3 = nn.Linear(HIDDEN2, n_out)
        self.activation = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)

        return x

class AgentDQN(Agent):
    def __init__(self, ob_space: Box, ac_space: Union[Box, Discrete]) -> None:
        super().__init__(ob_space, ac_space)

        assert type(ac_space) == Discrete, "action space not discrete"

        self.q = QNet(self.dim_ob, self.n_ac).to(DEVICE)
        self.q_t = deepcopy(self.q).to(DEVICE)
        self.q_optim = Adam(self.q.parameters(), lr=LR)

        self.replaybuffer = ReplayBuffer(REPLAY_SIZE)

    def get_ac(self, ob):
        """Return action according to e-greedy policy."""

        ac_logits : torch.Tensor = self.q(ob)
        max_ac = torch.argmax(ac_logits)

        rand_val = random.random()
        if rand_val >= EPS:
            return torch.tensor([max_ac])
        else:
            return torch.randint(0, self.n_ac, (1,))
        
    def q_update(self):
        ob, ac, rwd, next_ob, _ \
            = self.replaybuffer.sample(BATCH_SIZE)

        q = self.q(ob).gather(1, ac)
        q_t_max, _ = torch.max(self.q_t(next_ob), 1)
        target = rwd + GAMMA * q_t_max.unsqueeze(1)

        loss = torch.mean(torch.square(target.detach() - q), 0)
        
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()

    def q_t_update(self):
        q_t_sd = self.q_t.state_dict()
        q_sd = self.q.state_dict()
        for key in q_sd:
            q_t_sd[key] = q_sd[key] * TAU + q_t_sd[key] * (1 - TAU)
        self.q_t.load_state_dict(q_t_sd)


if __name__ == '__main__':

    env = Environment(ENV_NAME, render_mode=RENDER_MODE)
    agent = AgentDQN(env.ob_space, env.ac_space)

    ob = env.reset()

    ep_returns = []
    for episode in range(EPISODES):

        ep_rwds = []
        for t in count():
            ac = agent.get_ac(ob)
            next_ob, rwd, done = env.step(ac)
            ep_rwds.append(rwd)

            agent.replaybuffer.push(ob, ac, rwd, next_ob, done)
            ob = next_ob

            if done:
                if len(agent.replaybuffer) >= BATCH_SIZE:
                    agent.q_update()
                    agent.q_t_update()

                ep_returns.append(sum(ep_rwds))
                plot_returns(torch.as_tensor(ep_returns))

                if EPS > EPS_MIN: EPS *= EPS_DECAY

                env.reset()
                break

    plot_returns(torch.as_tensor(ep_returns), show_result=True)

    test_env = Environment(ENV_NAME, render_mode='human')
    test(agent, env, 10)