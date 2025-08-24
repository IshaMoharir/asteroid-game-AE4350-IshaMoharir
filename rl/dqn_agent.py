import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -----------------------------
# DQN Model
# -----------------------------
class DQN(nn.Module):
    '''A simple Deep Q-Network (DQN) model for reinforcement learning.'''

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, state_dim, action_dim):
        """
        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Number of discrete actions.
        """
        super(DQN, self).__init__()
        # 2-layer MLP with ReLU activations mapping state -> Q-values for each action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    # -----------------------------
    # Forward Pass
    # -----------------------------
    def forward(self, x):
        """
        Args:
            x (Tensor): Batch of states, shape [B, state_dim].
        Returns:
            Tensor: Q-values per action, shape [B, action_dim].
        """
        return self.net(x)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    ''' A Deep Q-Network (DQN) agent for reinforcement learning tasks.'''

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        """
        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Number of discrete actions.
            gamma (float): Discount factor.
            lr (float): Learning rate for Adam optimizer.
        """
        self.action_dim = action_dim
        self.gamma = gamma

        # Epsilon-greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # or 0.996
        self.epsilon_min = 0.05
        # self.epsilon_decay = 1 - (1 - self.epsilon_min) / 500  # decay to min over ~800 eps

        # Online and target networks
        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.update_target()  # Sync target with online network

        # Replay memory and optimisation setup
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 64

    # -----------------------------
    # Action Selection (Epsilon-Greedy)
    # -----------------------------
    def act(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state (array-like): Current state.
        Returns:
            int: Chosen action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state).argmax().item()

    # -----------------------------
    # Store Transition
    # -----------------------------
    def store(self, s, a, r, s_, done):
        """
        Store a transition tuple in the replay buffer.

        Args:
            s: state
            a: action
            r: reward
            s_: next state
            done (bool): terminal flag
        """
        self.memory.append((s, a, r, s_, done))

    # -----------------------------
    # Target Network Update
    # -----------------------------
    def update_target(self):
        """
        Hard-update target network to match the online network.
        """
        self.target.load_state_dict(self.model.state_dict())

    # -----------------------------
    # Training Step
    # -----------------------------
    def train_step(self):
        """
        Sample a batch from replay memory and perform one optimisation step.

        Notes:
            - Gradient clipping is called before backward() below, which results in no effect.
              If intended to clip gradients, move the clipping call to *after* loss.backward().

        """
        if len(self.memory) < self.batch_size:
            return

        # Sample random minibatch
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        # Convert to tensors
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.float32)
        s_ = torch.tensor(s_, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        # Q(s,a) for taken actions
        q_values = self.model(s).gather(1, a.unsqueeze(1)).squeeze()

        # Max_a' Q_target(s', a')
        next_q = self.target(s_).max(1)[0]

        # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        target = r + self.gamma * next_q * (1 - d)

        # Compute loss
        loss = self.loss_fn(q_values, target.detach())

        # Gradient clipping (note: has no effect here since grads not computed yet)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optional epsilon decay
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
