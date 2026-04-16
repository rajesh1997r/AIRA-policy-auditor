"""
Multi-Seed Statistical Validation — AIRA DQN+UCB

Trains the DQN+UCB agent from scratch on 5 independent seeds and reports
mean ± std for Precision, Recall, F1, and Efficiency over the last 50 episodes.
Also generates a shaded learning curve figure showing confidence bands.

Output:
    experiments/multi_seed_results.json
    results/figures/multi_seed_curve.png

Usage:
    python experiments/multi_seed_eval.py
    python experiments/multi_seed_eval.py --seeds 42 7 123
    python experiments/multi_seed_eval.py --episodes 300  # faster run for testing
    python experiments/multi_seed_eval.py --use-random-embeddings
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing.segmenter import segment_policy
from src.env.clause_auditor_env import ClauseAuditorEnv
from src.env.reward import SKIP, COMPARE
from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_agent import DQNAgent
from src.agents.ucb_explorer import UCBExplorer

# ─── Configuration ──────────────────────────────────────────────────────────

DEFAULT_SEEDS = [42, 7, 123, 999, 2024]
DEFAULT_EPISODES = 500
N_EVAL_EPISODES = 50

HP = {
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "gamma": 0.99,
    "lr": 1e-4,
    "buffer_capacity": 50_000,
    "budget_fraction": 0.15,
    "curriculum_start_budget": 0.30,
    "curriculum_episodes": 200,
    "ucb_c": 0.3,
    "ucb_warm_start": 10,
}


def load_env(policy_path: str, annotations_path: str, doc_prefix: str,
             use_random_embeddings: bool = False) -> ClauseAuditorEnv:
    with open(policy_path) as f:
        text = f.read()
    clauses = segment_policy(text, doc_prefix)

    with open(annotations_path) as f:
        ann = json.load(f)
    gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

    if use_random_embeddings:
        rng = np.random.default_rng(42)
        embs = rng.random((len(clauses), 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    else:
        from src.preprocessing.embedder import ClauseEmbedder
        embedder = ClauseEmbedder(cache_dir="data/processed")
        doc_name = os.path.splitext(os.path.basename(policy_path))[0]
        embs = embedder.embed_clauses(clauses, doc_name=doc_name)

    return ClauseAuditorEnv(clauses, embs, gt_pairs, budget_fraction=HP["budget_fraction"])


def curriculum_budget(episode: int, n_episodes: int) -> float:
    start = HP["curriculum_start_budget"]
    end = HP["budget_fraction"]
    progress = min(episode / HP["curriculum_episodes"], 1.0)
    return start - progress * (start - end)


def train_one_seed(seed: int, env: ClauseAuditorEnv, episodes: int) -> dict:
    """
    Train DQN+UCB from scratch for `episodes` episodes using the given seed.
    Returns per-episode history and final aggregated metrics.
    """
    import torch

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed everything for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    epsilon_decay_steps = episodes * max(env._budget, 1)
    dqn = DQNAgent(
        state_dim=772,
        n_actions=2,
        lr=HP["lr"],
        gamma=HP["gamma"],
        device=device,
        epsilon_start=HP["epsilon_start"],
        epsilon_end=HP["epsilon_end"],
        epsilon_decay_steps=epsilon_decay_steps,
    )
    agent = UCBExplorer(dqn, n_pairs=env._n_pairs, c=HP["ucb_c"],
                        warm_start_episodes=HP["ucb_warm_start"])
    buffer = ReplayBuffer(capacity=HP["buffer_capacity"], device=device)

    reward_history, precision_history, recall_history = [], [], []

    for episode in range(episodes):
        curr_budget = curriculum_budget(episode, episodes)
        env.curriculum_budget = curr_budget
        agent.reset()
        obs, info = env.reset(seed=seed * 10000 + episode)  # unique per seed + episode

        ep_reward = 0.0
        while True:
            i, j = env._pair_queue[env._queue_idx]
            n = len(env.clauses)
            flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
            action = agent.select_action(obs, flat_idx)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            ep_reward += reward

            if buffer.ready():
                agent.learn(buffer, batch_size=HP["batch_size"])

            obs = next_obs
            if done:
                break

        decisions = env.get_decisions()
        n_compared = sum(1 for a in decisions.values() if a == COMPARE)
        tp = sum(1 for k, a in decisions.items() if a == COMPARE and k in env.ground_truth_pairs)
        fn = sum(1 for k in env.ground_truth_pairs if decisions.get(k) != COMPARE)
        fp = n_compared - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        reward_history.append(ep_reward)
        precision_history.append(precision)
        recall_history.append(recall)

    final_precision = float(np.mean(precision_history[-N_EVAL_EPISODES:]))
    final_recall = float(np.mean(recall_history[-N_EVAL_EPISODES:]))
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)

    return {
        "seed": seed,
        "episodes": episodes,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": float(final_f1),
        "reward_history": [float(r) for r in reward_history],
        "precision_history": [float(p) for p in precision_history],
        "recall_history": [float(r) for r in recall_history],
    }


def smooth(values: list, window: int = 20) -> np.ndarray:
    arr = np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_multi_seed_curve(seed_results: list, out_path: str):
    """Generate shaded learning curves (mean ± 1 std) across seeds."""
    rewards_all = np.array([smooth(r["reward_history"], 20) for r in seed_results])
    recalls_all = np.array([smooth(r["recall_history"], 20) for r in seed_results])
    n_points = rewards_all.shape[1]
    x = np.arange(n_points) + 20  # offset for smoothing window

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DQN+UCB Learning Curves — Multi-Seed Validation (5 seeds)", fontsize=13)

    for ax, data, ylabel, color in [
        (ax1, rewards_all, "Episode Reward (smoothed)", "#2196F3"),
        (ax2, recalls_all, "Recall (smoothed)", "#4CAF50"),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color, label="± 1 std")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed statistical validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--policy", default="data/raw/synthetic_policy.txt")
    parser.add_argument("--annotations", default="data/annotated/synthetic_contradictions.json")
    parser.add_argument("--doc-prefix", default="SYN")
    parser.add_argument("--use-random-embeddings", action="store_true")
    args = parser.parse_args()

    print(f"Multi-seed evaluation: {len(args.seeds)} seeds × {args.episodes} episodes")
    print(f"Seeds: {args.seeds}")

    env = load_env(args.policy, args.annotations, args.doc_prefix, args.use_random_embeddings)
    print(f"Environment: {len(env.clauses)} clauses, {env._n_pairs} pairs, "
          f"{len(env.ground_truth_pairs)} contradictions")

    seed_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n[Seed {seed}] ({i+1}/{len(args.seeds)})")
        t0 = time.time()
        result = train_one_seed(seed, env, args.episodes)
        elapsed = time.time() - t0
        print(f"  F1={result['final_f1']:.3f}  Recall={result['final_recall']:.3f}  "
              f"Precision={result['final_precision']:.3f}  ({elapsed:.0f}s)")
        seed_results.append(result)

    # Aggregate across seeds
    metrics = {
        "precision": [r["final_precision"] for r in seed_results],
        "recall": [r["final_recall"] for r in seed_results],
        "f1": [r["final_f1"] for r in seed_results],
    }
    summary = {}
    for k, vals in metrics.items():
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    output = {
        "seeds": args.seeds,
        "episodes": args.episodes,
        "n_eval_episodes": N_EVAL_EPISODES,
        "summary": summary,
        "per_seed": seed_results,
    }

    out_path = "experiments/multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Plot
    plot_multi_seed_curve(seed_results, "results/figures/multi_seed_curve.png")

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Values'}")
    print("-" * 55)
    for k in ["precision", "recall", "f1"]:
        vals_str = "  ".join(f"{v:.3f}" for v in metrics[k])
        print(f"{k:<15} {summary[f'{k}_mean']:>10.3f} {summary[f'{k}_std']:>10.3f}  [{vals_str}]")
    print("=" * 55)


if __name__ == "__main__":
    main()
