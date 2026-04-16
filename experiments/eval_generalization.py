"""
Generalization Evaluation — AIRA Zero-Shot Cross-Document Performance

Loads the trained DQN+UCB checkpoint (no retraining) and evaluates on all three
policy documents: synthetic (training domain), NEU, and MIT (zero-shot transfer).

Computes:
  - Per-document Precision / Recall / F1 / Efficiency for DQN+UCB, Random, Cosine
  - Per-contradiction-type breakdown for the DQN+UCB agent on each document

Output:
    experiments/generalization_results.json

Usage:
    python experiments/eval_generalization.py
    python experiments/eval_generalization.py --n-eval 30
    python experiments/eval_generalization.py --use-random-embeddings
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.preprocessing.segmenter import segment_policy
from src.env.clause_auditor_env import ClauseAuditorEnv
from src.env.reward import SKIP, COMPARE
from src.agents.dqn_agent import DQNAgent
from src.agents.ucb_explorer import UCBExplorer
from src.evaluation.baselines import RandomAgent, CosineSimilarityBaseline, run_agent_episode, run_cosine_episode
from src.evaluation.metrics import compute_per_type_metrics

# ─── Document configs ───────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "name": "Synthetic",
        "split": "train",
        "policy_path": "data/raw/synthetic_policy.txt",
        "annotations_path": "data/annotated/synthetic_contradictions.json",
        "doc_prefix": "SYN",
    },
    {
        "name": "Northeastern",
        "split": "zero-shot",
        "policy_path": "data/raw/northeastern_ai_policy.txt",
        "annotations_path": "data/annotated/northeastern_contradictions.json",
        "doc_prefix": "NEU",
    },
    {
        "name": "MIT",
        "split": "zero-shot",
        "policy_path": "data/raw/mit_ai_policy.txt",
        "annotations_path": "data/annotated/mit_contradictions.json",
        "doc_prefix": "MIT",
    },
]

CHECKPOINT = "experiments/exp3_dqn_with_ucb/checkpoints/final.pt"
N_EVAL = 20
SEED_OFFSET = 1000  # Avoid overlap with training seeds (0-499)


def load_env(cfg: dict, budget_fraction: float = 0.15, use_random_embeddings: bool = False) -> ClauseAuditorEnv:
    with open(cfg["policy_path"]) as f:
        text = f.read()
    clauses = segment_policy(text, cfg["doc_prefix"])

    with open(cfg["annotations_path"]) as f:
        ann = json.load(f)
    gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

    if use_random_embeddings:
        rng = np.random.default_rng(42)
        embs = rng.random((len(clauses), 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    else:
        from src.preprocessing.embedder import ClauseEmbedder
        embedder = ClauseEmbedder(cache_dir="data/processed")
        doc_name = os.path.splitext(os.path.basename(cfg["policy_path"]))[0]
        embs = embedder.embed_clauses(clauses, doc_name=doc_name)

    return ClauseAuditorEnv(clauses, embs, gt_pairs, budget_fraction=budget_fraction)


def run_dqn_ucb_eval(env: ClauseAuditorEnv, checkpoint_path: str, n_eval: int) -> dict:
    """Evaluate trained DQN+UCB agent on env without further training."""
    dqn = DQNAgent(state_dim=772, n_actions=2)
    ucb = UCBExplorer(dqn, n_pairs=env._n_pairs, c=0.3, warm_start_episodes=0)

    if os.path.exists(checkpoint_path):
        ucb.load(checkpoint_path)
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}, using untrained agent")

    precision_list, recall_list, f1_list, efficiency_list = [], [], [], []

    for ep in range(n_eval):
        ucb.reset()
        obs, info = env.reset(seed=SEED_OFFSET + ep)

        while True:
            i, j = env._pair_queue[env._queue_idx]
            n = len(env.clauses)
            flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
            action = ucb.select_action(obs, flat_idx)
            obs, reward, terminated, truncated, step_info = env.step(action)
            if terminated or truncated:
                break

        decisions = env.get_decisions()
        compared = {p for p, a in decisions.items() if a == COMPARE}
        gt = env.ground_truth_pairs

        tp = len(compared & gt)
        fp = len(compared - gt)
        fn = len(gt - compared)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        efficiency = len(compared) / max(env._n_pairs, 1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        efficiency_list.append(efficiency)

    return {
        "precision": float(np.mean(precision_list)),
        "precision_std": float(np.std(precision_list)),
        "recall": float(np.mean(recall_list)),
        "recall_std": float(np.std(recall_list)),
        "f1": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "efficiency": float(np.mean(efficiency_list)),
        "efficiency_std": float(np.std(efficiency_list)),
    }


def run_random_eval(env: ClauseAuditorEnv, n_eval: int) -> dict:
    agent = RandomAgent(compare_prob=0.15, seed=99)
    precision_list, recall_list, f1_list, efficiency_list = [], [], [], []
    for ep in range(n_eval):
        result = run_agent_episode(agent, env, seed=SEED_OFFSET + ep)
        precision_list.append(result["precision"])
        recall_list.append(result["recall"])
        f1_list.append(result["f1"])
        efficiency_list.append(result["efficiency"])
    return {
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list)),
        "efficiency": float(np.mean(efficiency_list)),
    }


def run_cosine_eval(env: ClauseAuditorEnv, n_eval: int) -> dict:
    agent = CosineSimilarityBaseline(budget_fraction=0.15)
    precision_list, recall_list, f1_list, efficiency_list = [], [], [], []
    for ep in range(n_eval):
        result = run_cosine_episode(agent, env, seed=SEED_OFFSET + ep)
        precision_list.append(result["precision"])
        recall_list.append(result["recall"])
        f1_list.append(result["f1"])
        efficiency_list.append(result["efficiency"])
    return {
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list)),
        "efficiency": float(np.mean(efficiency_list)),
    }


def get_per_type_breakdown(env: ClauseAuditorEnv, checkpoint_path: str, annotations_cfg: dict) -> dict:
    """Run a single eval episode and compute per-type breakdown."""
    with open(annotations_cfg["annotations_path"]) as f:
        ann = json.load(f)
    annotations_list = ann["contradictions"]

    dqn = DQNAgent(state_dim=772, n_actions=2)
    ucb = UCBExplorer(dqn, n_pairs=env._n_pairs, c=0.3, warm_start_episodes=0)
    if os.path.exists(checkpoint_path):
        ucb.load(checkpoint_path)

    ucb.reset()
    obs, info = env.reset(seed=SEED_OFFSET + 42)
    while True:
        i, j = env._pair_queue[env._queue_idx]
        n = len(env.clauses)
        flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
        action = ucb.select_action(obs, flat_idx)
        obs, reward, terminated, truncated, step_info = env.step(action)
        if terminated or truncated:
            break

    decisions = env.get_decisions()
    return compute_per_type_metrics(decisions, annotations_list)


def main():
    parser = argparse.ArgumentParser(description="AIRA Generalization Evaluation")
    parser.add_argument("--n-eval", type=int, default=N_EVAL, help="Episodes per document per agent")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Path to trained checkpoint")
    parser.add_argument("--use-random-embeddings", action="store_true")
    args = parser.parse_args()

    results = {"n_eval_episodes": args.n_eval, "checkpoint": args.checkpoint, "documents": {}}

    header = f"\n{'Document':<15} {'Agent':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Efficiency':>12}"
    print(header)
    print("-" * len(header))

    for doc_cfg in DOCUMENTS:
        name = doc_cfg["name"]
        split = doc_cfg["split"]
        print(f"\n[{name}] ({split})")

        env = load_env(doc_cfg, use_random_embeddings=args.use_random_embeddings)
        print(f"  Clauses: {len(env.clauses)}, Pairs: {env._n_pairs}, GT: {len(env.ground_truth_pairs)}")

        dqn_ucb = run_dqn_ucb_eval(env, args.checkpoint, args.n_eval)
        random_res = run_random_eval(env, args.n_eval)
        cosine_res = run_cosine_eval(env, args.n_eval)
        per_type = get_per_type_breakdown(env, args.checkpoint, doc_cfg)

        for agent_name, res in [("DQN+UCB", dqn_ucb), ("Random", random_res), ("Cosine", cosine_res)]:
            print(
                f"  {agent_name:<12} P={res['precision']:.3f}  R={res['recall']:.3f}  "
                f"F1={res['f1']:.3f}  Eff={res['efficiency']:.3f}"
            )

        results["documents"][name] = {
            "split": split,
            "n_clauses": len(env.clauses),
            "n_pairs": env._n_pairs,
            "n_contradictions": len(env.ground_truth_pairs),
            "dqn_ucb": dqn_ucb,
            "random": random_res,
            "cosine": cosine_res,
            "per_type_breakdown": per_type,
        }

    out_path = "experiments/generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Document':<15} {'Split':<12} {'DQN+UCB F1':>12} {'DQN+UCB Recall':>16} {'Efficiency':>12}")
    print("-" * 70)
    for name, doc_res in results["documents"].items():
        d = doc_res["dqn_ucb"]
        print(f"{name:<15} {doc_res['split']:<12} {d['f1']:>12.3f} {d['recall']:>16.3f} {d['efficiency']:>12.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
