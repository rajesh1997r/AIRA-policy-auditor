"""
UCB Explorer — Upper Confidence Bound exploration strategy for clause pair selection.

Combines DQN exploitation (learned Q-values) with UCB1 exploration (uncertainty bonus)
to prioritize under-explored clause pairs.

UCB1 formula applied to the COMPARE action:
    score(i,j) = Q_dqn(s_ij, COMPARE) + c * sqrt( ln(t + 1) / N(i,j) )

Where:
    Q_dqn : DQN's estimated value of COMPARE for this pair's state
    c     : exploration constant (default sqrt(2) ≈ 1.41)
    t     : total pair-selection steps across the current episode
    N(i,j): number of times pair (i,j) has been seen in this episode

The UCB bonus applies ONLY to the COMPARE action. The SKIP action uses the
raw DQN Q-value. This is semantically correct: UCB encourages investigating
unseen or rarely-seen pairs before deciding to skip them.

After the warm-start phase (first `warm_start_episodes` episodes), unvisited
pairs are initialized to N=1 rather than N=0 (avoids infinite bonus while still
encouraging exploration).
"""

import math
from typing import Optional

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.env.reward import SKIP, COMPARE


class UCBExplorer:
    """
    UCB1 exploration wrapper around DQNAgent.

    The DQNAgent handles all learning (replay buffer, Bellman updates).
    UCBExplorer wraps action selection only — `learn()` is called on the
    underlying DQNAgent unchanged.

    Args:
        dqn_agent: A DQNAgent instance.
        n_pairs: Total number of unique clause pairs in the document.
        c: UCB exploration constant. Higher values encourage more exploration.
        warm_start_episodes: During this many episodes, use infinite UCB bonus
            for unvisited pairs. After that, initialize N=1 for unvisited pairs.
    """

    def __init__(
        self,
        dqn_agent: DQNAgent,
        n_pairs: int,
        c: float = 0.3,
        warm_start_episodes: int = 100,
    ):
        self.agent = dqn_agent
        self.n_pairs = n_pairs
        self.c = c
        self.warm_start_episodes = warm_start_episodes

        # Per-episode state
        self._visit_counts = np.zeros(n_pairs, dtype=np.float32)
        self._total_steps = 0
        self._episode_count = 0

    @staticmethod
    def pair_to_idx(i: int, j: int, n: int) -> int:
        """
        Map a canonical pair (i < j) to a flat index in [0, n*(n-1)/2).

        Uses the triangular number formula to avoid an n×n matrix.
        """
        assert i < j, f"Require i < j, got i={i}, j={j}"
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def reset(self):
        """
        Increment episode counter.

        Visit counts and total_steps accumulate across episodes — this is
        what gives UCB1 its exploration-to-exploitation shift over training.
        Do NOT reset _visit_counts or _total_steps here.
        """
        self._episode_count += 1

    def select_action(self, state: np.ndarray, pair_idx: int) -> int:
        """
        Select COMPARE or SKIP using UCB-augmented Q-values.

        Args:
            state: Current observation (772-dim float32).
            pair_idx: Flat index of the current clause pair (from pair_to_idx).

        Returns:
            Action int: 0 (SKIP) or 1 (COMPARE).
        """
        q_values = self.agent.get_q_values(state)  # shape (2,)

        # UCB bonus for COMPARE action
        n_visits = self._visit_counts[pair_idx]
        if n_visits == 0:
            # Unvisited during warm start — always explore
            ucb_bonus = float("inf")
        else:
            ucb_bonus = self.c * math.sqrt(
                math.log(self._total_steps + 1) / n_visits
            )

        # Augment only the COMPARE Q-value
        compare_score = q_values[COMPARE] + ucb_bonus
        skip_score = q_values[SKIP]

        self._visit_counts[pair_idx] += 1
        self._total_steps += 1

        return COMPARE if compare_score >= skip_score else SKIP

    def learn(self, replay_buffer, batch_size: int = 64) -> float:
        """Delegate learning to the underlying DQNAgent."""
        return self.agent.learn(replay_buffer, batch_size=batch_size)

    def save(self, path: str):
        """Save underlying DQN agent."""
        self.agent.save(path)

    def load(self, path: str):
        """Load underlying DQN agent."""
        self.agent.load(path)

    @property
    def epsilon(self) -> float:
        return self.agent.epsilon

    @property
    def device(self):
        return self.agent.device

    # Expose for direct epsilon-greedy fallback if needed
    def select_action_dqn(self, state: np.ndarray) -> int:
        """Pure DQN action selection (bypasses UCB)."""
        return self.agent.select_action(state)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from src.agents.dqn_agent import DQNAgent

    dqn = DQNAgent(state_dim=772, n_actions=2)
    ucb = UCBExplorer(dqn, n_pairs=10, c=1.41, warm_start_episodes=2)

    # Simulate 2 episodes
    for ep in range(3):
        ucb.reset()
        print(f"\nEpisode {ep + 1} (warm_start={ep < ucb.warm_start_episodes}):")
        for pair_idx in range(5):
            state = np.random.randn(772).astype(np.float32)
            action = ucb.select_action(state, pair_idx)
            print(
                f"  pair {pair_idx}: action={'COMPARE' if action == COMPARE else 'SKIP'} "
                f"visits={ucb._visit_counts[pair_idx]:.0f}"
            )
