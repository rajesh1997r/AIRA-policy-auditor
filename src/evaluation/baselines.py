"""
Baseline agents for comparison against the trained DQN+UCB system.

RandomAgent          : randomly COMPARE or SKIP, calibrated to the budget fraction.
ExhaustiveAgent      : always COMPARE (max recall, max resource usage).
CosineSimilarityBaseline: ranks all pairs by embedding cosine similarity,
                          COMPAREs the top-k (k = budget). Non-RL baseline.
"""

import numpy as np
from src.env.reward import SKIP, COMPARE


class RandomAgent:
    """
    Random baseline: selects COMPARE with probability `compare_prob`.

    By default compare_prob = budget_fraction so the expected number of
    comparisons matches the DQN's budget, enabling fair efficiency comparison.
    """

    def __init__(self, compare_prob: float = 0.15, seed: int = 42):
        self.compare_prob = compare_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> int:
        return COMPARE if self.rng.random() < self.compare_prob else SKIP

    def reset(self):
        pass  # stateless


class ExhaustiveAgent:
    """
    Always-COMPARE baseline: reviews every pair.

    Establishes the theoretical upper bound on recall.
    Efficiency = 1.0 (compares 100% of pairs — worst case).
    """

    def select_action(self, state: np.ndarray) -> int:
        return COMPARE

    def reset(self):
        pass  # stateless


class CosineSimilarityBaseline:
    """
    Non-RL baseline: ranks all clause pairs by cosine similarity of their
    embeddings and COMPAREs the top-k most similar pairs (k = budget).

    Rationale: contradictory clauses often use similar vocabulary (same topic,
    opposite rules), so high cosine similarity can indicate conflict. This
    baseline represents the best simple threshold approach without learning.

    Unlike RandomAgent, this agent pre-computes all pairwise similarities
    and selects exactly `budget` pairs — it does not use the env step interface
    for action selection but integrates with run_agent_episode via a pre-built
    compare set.
    """

    def __init__(self, budget_fraction: float = 0.15):
        self.budget_fraction = budget_fraction
        self._compare_set: set = set()

    def prepare(self, env) -> None:
        """
        Pre-compute pairwise cosine similarities and select the top-k pairs.
        Call this after env.reset() and before stepping.
        """
        embeddings = env.embeddings  # shape (N, 384), already normalized
        n = len(env.clauses)
        budget = env._budget

        # Compute all pairwise dot products (= cosine sim for normalized vectors)
        sims = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                pair_key = frozenset({env.clauses[i]["id"], env.clauses[j]["id"]})
                sims[pair_key] = sim

        # Select top-k by cosine similarity
        ranked = sorted(sims.items(), key=lambda x: -x[1])
        self._compare_set = {pair for pair, _ in ranked[:budget]}

    def select_action(self, state: np.ndarray) -> int:
        # Fallback: used only if called without prepare(); always SKIP
        return SKIP

    def reset(self):
        self._compare_set = set()


def run_cosine_episode(agent: "CosineSimilarityBaseline", env, seed: int = None) -> dict:
    """
    Run one episode with CosineSimilarityBaseline.

    Differs from run_agent_episode: the agent pre-computes similarity ranks
    before stepping, then COMPAREs exactly the pre-selected top-k pairs.
    """
    obs, info = env.reset(seed=seed)
    agent.prepare(env)

    episode_log = []
    while True:
        # Identify current pair by index
        i, j = env._pair_queue[env._queue_idx]
        pair_key = frozenset({env.clauses[i]["id"], env.clauses[j]["id"]})
        action = COMPARE if pair_key in agent._compare_set else SKIP

        obs, reward, terminated, truncated, step_info = env.step(action)
        episode_log.append((action, reward, step_info["is_contradiction"]))
        if terminated or truncated:
            break

    from src.evaluation.metrics import summarize_episode
    summary = summarize_episode(episode_log)
    summary["efficiency"] = summary["compare_count"] / max(info["n_pairs"], 1)
    return summary


def run_agent_episode(agent, env, seed: int = None) -> dict:
    """
    Run one episode with the given agent and return episode metrics.

    Args:
        agent: Any agent with a select_action(state) -> int method.
        env: ClauseAuditorEnv instance.
        seed: Optional seed for reproducibility.

    Returns:
        dict with total_reward, steps, compare_count, TP, FP, FN,
              precision, recall, f1, efficiency.
    """
    from src.evaluation.metrics import summarize_episode

    obs, info = env.reset(seed=seed)
    episode_log = []

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        episode_log.append((action, reward, step_info["is_contradiction"]))
        if terminated or truncated:
            break

    summary = summarize_episode(episode_log)
    summary["efficiency"] = summary["compare_count"] / max(info["n_pairs"], 1)
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    import numpy as np
    from src.preprocessing.segmenter import segment_policy
    from src.env.clause_auditor_env import ClauseAuditorEnv

    with open("data/raw/synthetic_policy.txt") as f:
        text = f.read()

    clauses = segment_policy(text, "SYN")
    n = len(clauses)
    rng = np.random.default_rng(0)
    embs = rng.random((n, 384)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    # Use the annotated ground truth
    import json
    with open("data/annotated/synthetic_contradictions.json") as f:
        ann = json.load(f)
    gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

    env = ClauseAuditorEnv(clauses, embs, gt_pairs, budget_fraction=0.15)

    # Run each agent for 10 episodes and average
    random_agent = RandomAgent(compare_prob=0.15, seed=0)
    exhaustive_agent = ExhaustiveAgent()

    N_EVAL = 10
    for name, agent in [("Random", random_agent), ("Exhaustive", exhaustive_agent)]:
        rewards, recalls, efficiencies = [], [], []
        for ep in range(N_EVAL):
            result = run_agent_episode(agent, env, seed=ep)
            rewards.append(result["total_reward"])
            recalls.append(result["recall"])
            efficiencies.append(result["efficiency"])

        print(f"\n{name} Agent (avg over {N_EVAL} episodes):")
        print(f"  Reward:     {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Recall:     {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
        print(f"  Efficiency: {np.mean(efficiencies):.2f} ± {np.std(efficiencies):.2f}")
