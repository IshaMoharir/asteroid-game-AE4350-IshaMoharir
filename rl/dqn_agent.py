import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    '''A simple Deep Q-Network (DQN) model for reinforcement learning.'''

    # Parameters:
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    ''' A Deep Q-Network (DQN) agent for reinforcement learning tasks.'''

    # Parameters:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # or 0.996
        self.epsilon_min = 0.05
        # self.epsilon_decay = 1 - (1 - self.epsilon_min) / 500  # decay to min over ~800 eps

        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.update_target()

        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 64

    # Methods:
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state).argmax().item()

    # Store a transition in the replay memory.
    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    # Update the target network with the current model's weights.
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # Perform a training step using a batch from the replay memory.
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.float32)
        s_ = torch.tensor(s_, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        q_values = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target(s_).max(1)[0]
        target = r + self.gamma * next_q * (1 - d)

        loss = self.loss_fn(q_values, target.detach())
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
