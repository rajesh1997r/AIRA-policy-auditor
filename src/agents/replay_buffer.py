"""
Experience replay buffer for the DQN agent.

Stores (state, action, reward, next_state, done) transitions as numpy arrays.
Returns torch tensors on sample() for direct use in gradient updates.
"""

import collections
import random
from typing import Tuple

import numpy as np
import torch

# Minimum buffer size before sampling is allowed
MIN_REPLAY_SIZE = 500


class ReplayBuffer:
    """
    Fixed-capacity circular buffer for experience replay.

    States are stored as float32 numpy arrays and converted to tensors
    on sample() using zero-copy torch.from_numpy() where possible.

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimension of each state vector (default 772).
        device: Torch device for returned tensors.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        state_dim: int = 772,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = torch.device(device)
        self._buffer = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer."""
        self._buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
            on self.device.

        Raises:
            RuntimeError if buffer has fewer than MIN_REPLAY_SIZE transitions.
        """
        if len(self) < MIN_REPLAY_SIZE:
            raise RuntimeError(
                f"Buffer has {len(self)} transitions — need at least "
                f"{MIN_REPLAY_SIZE} before sampling."
            )

        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self._buffer)

    def ready(self) -> bool:
        """Returns True if the buffer has enough transitions to sample."""
        return len(self) >= MIN_REPLAY_SIZE
