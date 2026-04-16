"""
DQN Agent for policy clause auditing.

Architecture: fully connected network (no CNN — input is a 772-dim float vector).
    772 → 512 (BN + ReLU) → 256 (BN + ReLU) → 128 (ReLU) → 2

Key design notes:
- BatchNorm1d requires explicit train()/eval() switching around inference.
  The target network is always in eval() mode.
- Huber loss (SmoothL1) is used instead of MSE for stability with the
  asymmetric reward structure (+2.0, -0.1, -0.5, 0.0).
- Target network is hard-updated every TARGET_UPDATE_STEPS environment steps.
- Epsilon uses linear decay to maintain exploration across the full training run.
"""

import os
import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.replay_buffer import ReplayBuffer
from src.env.reward import SKIP, COMPARE

# Default hyperparameters
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 64
TARGET_UPDATE_STEPS = 100

STATE_DIM = 772
N_ACTIONS = 2


class QNetwork(nn.Module):
    """
    Fully connected Q-network for the 772-dim clause pair state space.

    Network: 772 → 512 (BN+ReLU) → 256 (BN+ReLU) → 128 (ReLU) → 2
    """

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    DQN agent with experience replay, target network, and linear epsilon decay.

    Args:
        state_dim: Input dimension (default 772).
        n_actions: Number of discrete actions (default 2).
        lr: Adam learning rate.
        gamma: Discount factor.
        device: 'cpu', 'cuda', or 'mps'.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay_steps: Number of gradient update steps over which to decay epsilon.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = DEFAULT_LR,
        gamma: float = DEFAULT_GAMMA,
        device: str = "cpu",
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()  # Target net is always in eval mode

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self._learn_steps = 0
        self._env_steps = 0  # total environment steps taken
        self.epsilon = epsilon_start

    def _update_epsilon(self):
        """Linear epsilon decay based on learn steps."""
        progress = min(self._learn_steps / max(self.epsilon_decay_steps, 1), 1.0)
        self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions given a single state.

        Switches q_net to eval() for inference to ensure BatchNorm uses running
        statistics rather than batch statistics. Restores train() mode afterward.

        Returns:
            np.ndarray of shape (n_actions,), dtype float32.
        """
        self.q_net.eval()
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(state_t).cpu().numpy().flatten()
        self.q_net.train()
        return q

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: np.ndarray of shape (state_dim,).

        Returns:
            Action int (0=SKIP, 1=COMPARE).
        """
        self._env_steps += 1
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.get_q_values(state)))

    def learn(self, replay_buffer: ReplayBuffer, batch_size: int = DEFAULT_BATCH_SIZE) -> float:
        """
        Perform one gradient update step from the replay buffer.

        Args:
            replay_buffer: ReplayBuffer with enough transitions.
            batch_size: Mini-batch size.

        Returns:
            Scalar float loss value.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Current Q-values: Q(s, a)
        self.q_net.train()
        q_values = self.q_net(states)  # (batch, n_actions)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states)  # (batch, n_actions)
            max_next_q = next_q.max(dim=1).values  # (batch,)
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        self._update_epsilon()

        # Hard target network update
        if self._learn_steps % TARGET_UPDATE_STEPS == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        """Hard copy weights from q_net to target_net."""
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    def save(self, path: str):
        """Save agent state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self._learn_steps,
                "env_steps": self._env_steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._learn_steps = checkpoint.get("learn_steps", 0)
        self._env_steps = checkpoint.get("env_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.target_net.eval()
        self.q_net.train()

    def reset(self):
        """No per-episode state to reset for basic DQN."""
        pass
