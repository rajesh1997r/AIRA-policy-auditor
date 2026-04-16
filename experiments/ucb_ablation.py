"""
UCB Exploration Constant Ablation — AIRA

Sweeps the UCB exploration constant c over [0.0, 0.1, 0.3, 1.0, 1.41, 3.0]
and trains for 300 episodes at each value. c=0.0 is equivalent to pure DQN
(no UCB bonus), providing a useful baseline within the ablation.

Produces:
    experiments/ucb_ablation_results.json
    results/figures/ucb_ablation.png (3-panel: Recall, F1, Efficiency vs c)

Usage:
    python experiments/ucb_ablation.py
    python experiments/ucb_ablation.py --episodes 200  # faster test
    python experiments/ucb_ablation.py --use-random-embeddings
"""

import argparse
import json
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

UCB_C_VALUES = [0.0, 0.1, 0.3, 1.0, 1.41, 3.0]
DEFAULT_EPISODES = 300
N_EVAL_EPISODES = 50
SEED = 42

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


def curriculum_budget(episode: int) -> float:
    start = HP["curriculum_start_budget"]
    end = HP["budget_fraction"]
    progress = min(episode / HP["curriculum_episodes"], 1.0)
    return start - progress * (start - end)


def train_one_c(c_value: float, env: ClauseAuditorEnv, episodes: int) -> dict:
    """Train DQN+UCB with a fixed c value for `episodes` episodes."""
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    epsilon_decay_steps = episodes * max(env._budget, 1)
    dqn = DQNAgent(
        state_dim=772, n_actions=2, lr=HP["lr"], gamma=HP["gamma"], device=device,
        epsilon_start=HP["epsilon_start"], epsilon_end=HP["epsilon_end"],
        epsilon_decay_steps=epsilon_decay_steps,
    )

    # c=0.0 is pure DQN (UCB bonus is always 0)
    agent = UCBExplorer(dqn, n_pairs=env._n_pairs, c=c_value,
                        warm_start_episodes=HP["ucb_warm_start"] if c_value > 0.0 else 0)
    buffer = ReplayBuffer(capacity=HP["buffer_capacity"], device=device)

    precision_history, recall_history, pairs_compared_history = [], [], []

    for episode in range(episodes):
        env.curriculum_budget = curriculum_budget(episode)
        agent.reset()
        obs, info = env.reset(seed=SEED * 10000 + episode)

        while True:
            i, j = env._pair_queue[env._queue_idx]
            n = len(env.clauses)
            flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
            action = agent.select_action(obs, flat_idx)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            if buffer.ready():
                agent.learn(buffer, batch_size=HP["batch_size"])
            obs = next_obs
            if done:
                break

        decisions = env.get_decisions()
        n_compared = sum(1 for a in decisions.values() if a == COMPARE)
        tp = sum(1 for k, a in decisions.items() if a == COMPARE and k in env.ground_truth_pairs)
        fp = n_compared - tp
        fn = sum(1 for k in env.ground_truth_pairs if decisions.get(k) != COMPARE)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_history.append(precision)
        recall_history.append(recall)
        pairs_compared_history.append(n_compared)

    final_precision = float(np.mean(precision_history[-N_EVAL_EPISODES:]))
    final_recall = float(np.mean(recall_history[-N_EVAL_EPISODES:]))
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)
    final_efficiency = float(np.mean(pairs_compared_history[-N_EVAL_EPISODES:]) / env._n_pairs)

    return {
        "c": c_value,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": float(final_f1),
        "final_efficiency": final_efficiency,
        "recall_history": [float(r) for r in recall_history],
        "precision_history": [float(p) for p in precision_history],
    }


def plot_ablation(ablation_results: list, out_path: str):
    """3-panel plot: Recall, F1, Efficiency vs UCB c value."""
    c_values = [r["c"] for r in ablation_results]
    recalls = [r["final_recall"] for r in ablation_results]
    f1s = [r["final_f1"] for r in ablation_results]
    efficiencies = [r["final_efficiency"] for r in ablation_results]

    # Highlight the chosen c=0.3
    chosen_idx = c_values.index(0.3) if 0.3 in c_values else None
    c_labels = [str(c) for c in c_values]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("UCB Exploration Constant Ablation (c = 0.0 is pure DQN)", fontsize=12)

    panels = [
        (axes[0], recalls, "Recall", "#4CAF50"),
        (axes[1], f1s, "F1 Score", "#2196F3"),
        (axes[2], efficiencies, "Efficiency (lower = better)", "#FF9800"),
    ]

    for ax, values, ylabel, color in panels:
        bars = ax.bar(c_labels, values, color=color, alpha=0.7, edgecolor="gray")
        if chosen_idx is not None:
            bars[chosen_idx].set_edgecolor("black")
            bars[chosen_idx].set_linewidth(2.5)
            bars[chosen_idx].set_alpha(1.0)
        ax.set_xlabel("UCB constant c")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        ax.grid(axis="y", alpha=0.3)
        # Annotate chosen value
        if chosen_idx is not None:
            ax.annotate("chosen", xy=(chosen_idx, values[chosen_idx]),
                        xytext=(chosen_idx, values[chosen_idx] + max(values) * 0.08),
                        ha="center", fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="UCB c parameter ablation study")
    parser.add_argument("--c-values", type=float, nargs="+", default=UCB_C_VALUES)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--policy", default="data/raw/synthetic_policy.txt")
    parser.add_argument("--annotations", default="data/annotated/synthetic_contradictions.json")
    parser.add_argument("--doc-prefix", default="SYN")
    parser.add_argument("--use-random-embeddings", action="store_true")
    args = parser.parse_args()

    print(f"UCB ablation: c ∈ {args.c_values}, {args.episodes} episodes each")

    env = load_env(args.policy, args.annotations, args.doc_prefix, args.use_random_embeddings)
    print(f"Environment: {len(env.clauses)} clauses, {env._n_pairs} pairs, "
          f"{len(env.ground_truth_pairs)} contradictions")

    ablation_results = []
    for i, c_val in enumerate(args.c_values):
        label = "pure DQN" if c_val == 0.0 else f"c={c_val}"
        print(f"\n[{label}] ({i+1}/{len(args.c_values)})")
        t0 = time.time()
        result = train_one_c(c_val, env, args.episodes)
        elapsed = time.time() - t0
        print(f"  Recall={result['final_recall']:.3f}  F1={result['final_f1']:.3f}  "
              f"Efficiency={result['final_efficiency']:.3f}  ({elapsed:.0f}s)")
        ablation_results.append(result)

    output = {
        "c_values": args.c_values,
        "episodes": args.episodes,
        "n_eval_episodes": N_EVAL_EPISODES,
        "seed": SEED,
        "results": ablation_results,
    }

    out_path = "experiments/ucb_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plot_ablation(ablation_results, "results/figures/ucb_ablation.png")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'c value':<10} {'Recall':>10} {'F1':>10} {'Efficiency':>12} {'Notes'}")
    print("-" * 65)
    for r in ablation_results:
        note = " ← chosen" if r["c"] == 0.3 else (" ← pure DQN" if r["c"] == 0.0 else "")
        print(f"{r['c']:<10.2f} {r['final_recall']:>10.3f} {r['final_f1']:>10.3f} "
              f"{r['final_efficiency']:>12.3f}{note}")
    print("=" * 65)


if __name__ == "__main__":
    main()
