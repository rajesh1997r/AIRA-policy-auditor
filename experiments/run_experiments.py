"""
Run AIRA experiments and save results to disk.

Usage:
    python experiments/run_experiments.py --experiment random
    python experiments/run_experiments.py --experiment dqn
    python experiments/run_experiments.py --experiment dqn_ucb
    python experiments/run_experiments.py --experiment all

Results are saved to:
    experiments/exp1_random/results.json
    experiments/exp2_dqn/results.json
    experiments/exp3_dqn_with_ucb/results.json
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from src.preprocessing.segmenter import segment_policy
from src.env.clause_auditor_env import ClauseAuditorEnv
from src.env.reward import SKIP, COMPARE
from src.agents.replay_buffer import ReplayBuffer, MIN_REPLAY_SIZE
from src.agents.dqn_agent import DQNAgent
from src.agents.ucb_explorer import UCBExplorer
from src.evaluation.baselines import RandomAgent, ExhaustiveAgent, run_agent_episode
from src.evaluation.metrics import compute_metrics, aggregate_metrics

# ─── Hyperparameters ────────────────────────────────────────────────────────
HP = {
    "episodes": 500,
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "gamma": 0.99,
    "lr": 1e-4,
    "buffer_capacity": 50_000,
    "budget_fraction": 0.15,
    "curriculum_start_budget": 0.30,   # start wide, decay to budget_fraction
    "curriculum_episodes": 200,        # over how many episodes
    "n_eval_episodes": 50,
    "ucb_c": 0.3,
    "ucb_warm_start": 10,
    "checkpoint_every": 100,
}


def load_env(
    policy_path: str,
    annotations_path: str,
    doc_prefix: str,
    budget_fraction: float = HP["budget_fraction"],
    use_random_embeddings: bool = False,
) -> ClauseAuditorEnv:
    """Build a ClauseAuditorEnv from a policy file and annotations."""
    with open(policy_path) as f:
        text = f.read()
    clauses = segment_policy(text, doc_prefix)

    import json as _json
    with open(annotations_path) as f:
        ann = _json.load(f)
    gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

    if use_random_embeddings:
        n = len(clauses)
        rng = np.random.default_rng(42)
        embs = rng.random((n, 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    else:
        from src.preprocessing.embedder import ClauseEmbedder
        embedder = ClauseEmbedder(cache_dir="data/processed")
        doc_name = os.path.splitext(os.path.basename(policy_path))[0]
        embs = embedder.embed_clauses(clauses, doc_name=doc_name)

    return ClauseAuditorEnv(clauses, embs, gt_pairs, budget_fraction=budget_fraction)


def curriculum_budget(episode: int) -> float:
    """Linearly decay budget from curriculum_start to budget_fraction."""
    start = HP["curriculum_start_budget"]
    end = HP["budget_fraction"]
    progress = min(episode / HP["curriculum_episodes"], 1.0)
    return start - progress * (start - end)


def run_random_experiment(env: ClauseAuditorEnv, out_dir: str):
    """Experiment 1: Random agent baseline."""
    os.makedirs(out_dir, exist_ok=True)
    agent = RandomAgent(compare_prob=HP["budget_fraction"], seed=0)

    reward_history, precision_history, recall_history = [], [], []
    pairs_compared_history = []

    print("Running Experiment 1: Random Agent...")
    for ep in tqdm(range(HP["n_eval_episodes"]), desc="Random"):
        result = run_agent_episode(agent, env, seed=ep)
        reward_history.append(result["total_reward"])
        precision_history.append(result["precision"])
        recall_history.append(result["recall"])
        pairs_compared_history.append(result["compare_count"])

    results = {
        "experiment": "random",
        "episodes": HP["n_eval_episodes"],
        "final_precision": float(np.mean(precision_history)),
        "final_precision_std": float(np.std(precision_history)),
        "final_recall": float(np.mean(recall_history)),
        "final_recall_std": float(np.std(recall_history)),
        "final_f1": float(
            2 * np.mean(precision_history) * np.mean(recall_history)
            / max(np.mean(precision_history) + np.mean(recall_history), 1e-9)
        ),
        "efficiency_score": float(np.mean([p / env._n_pairs for p in pairs_compared_history])),
        "faithfulness_score": 1.0,
        "reward_history": [float(r) for r in reward_history],
        "pairs_compared_history": [int(p) for p in pairs_compared_history],
        "precision_history": [float(p) for p in precision_history],
        "recall_history": [float(r) for r in recall_history],
    }

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Precision: {results['final_precision']:.3f} | Recall: {results['final_recall']:.3f}")
    print(f"  Saved to {out_path}")
    return results


def run_dqn_experiment(
    env: ClauseAuditorEnv, out_dir: str, use_ucb: bool = False
) -> dict:
    """Experiments 2 and 3: DQN (with or without UCB)."""
    os.makedirs(out_dir, exist_ok=True)
    exp_name = "dqn_ucb" if use_ucb else "dqn"
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Determine device
    if hasattr(torch := sys.modules.get("torch", None), "cuda") and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Lazy torch import
    import torch  # noqa: F811

    epsilon_decay_steps = HP["episodes"] * max(env._budget, 1)
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

    if use_ucb:
        agent = UCBExplorer(dqn, n_pairs=env._n_pairs, c=HP["ucb_c"], warm_start_episodes=HP["ucb_warm_start"])
    else:
        agent = dqn

    buffer = ReplayBuffer(capacity=HP["buffer_capacity"], device=device)

    reward_history, loss_history = [], []
    precision_history, recall_history, pairs_compared_history = [], [], []
    ucb_bonus_history = []

    print(f"\nRunning {'Experiment 3: DQN + UCB' if use_ucb else 'Experiment 2: DQN'}...")
    start_time = time.time()

    for episode in tqdm(range(HP["episodes"]), desc=exp_name.upper()):
        # Curriculum budget
        curr_budget = curriculum_budget(episode)
        env.curriculum_budget = curr_budget

        if use_ucb:
            agent.reset()

        obs, info = env.reset(seed=episode)
        ep_reward = 0.0
        ep_losses = []
        ep_log = []

        while True:
            if use_ucb:
                pair_idx = env._queue_idx  # current pair position in shuffled queue
                i, j = env._pair_queue[pair_idx]
                n = len(env.clauses)
                flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
                action = agent.select_action(obs, flat_idx)

                # Track average UCB bonus for plotting (UCB1: c * sqrt(ln(t)/N))
                n_visits = agent._visit_counts[flat_idx]
                if n_visits > 0 and agent._total_steps > 1:
                    bonus = HP["ucb_c"] * math.sqrt(
                        math.log(agent._total_steps) / n_visits
                    )
                    ucb_bonus_history.append(bonus)
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, float(done))
            ep_reward += reward
            ep_log.append((action, reward, step_info["is_contradiction"]))

            if buffer.ready():
                loss = agent.learn(buffer, batch_size=HP["batch_size"])
                ep_losses.append(loss)

            obs = next_obs
            if done:
                break

        # Evaluate this episode
        decisions = env.get_decisions()
        n_compared = sum(1 for a in decisions.values() if a == COMPARE)

        precision_history.append(
            sum(1 for k, a in decisions.items() if a == COMPARE and k in env.ground_truth_pairs)
            / max(n_compared, 1)
        )
        recall_history.append(
            sum(1 for k in env.ground_truth_pairs if decisions.get(k) == COMPARE)
            / max(len(env.ground_truth_pairs), 1)
        )
        pairs_compared_history.append(n_compared)
        reward_history.append(ep_reward)
        loss_history.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        # Checkpoint
        if (episode + 1) % HP["checkpoint_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep{episode+1}.pt")
            agent.save(ckpt_path)

    # Save final checkpoint
    agent.save(os.path.join(ckpt_dir, "final.pt"))
    elapsed = time.time() - start_time

    final_precision = float(np.mean(precision_history[-HP["n_eval_episodes"]:]))
    final_recall = float(np.mean(recall_history[-HP["n_eval_episodes"]:]))

    results = {
        "experiment": exp_name,
        "episodes": HP["episodes"],
        "training_time_seconds": elapsed,
        "final_precision": final_precision,
        "final_precision_std": float(np.std(precision_history[-HP["n_eval_episodes"]:])),
        "final_recall": final_recall,
        "final_recall_std": float(np.std(recall_history[-HP["n_eval_episodes"]:])),
        "final_f1": float(
            2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)
        ),
        "efficiency_score": float(
            np.mean(pairs_compared_history[-HP["n_eval_episodes"]:]) / env._n_pairs
        ),
        "faithfulness_score": 1.0,
        "reward_history": [float(r) for r in reward_history],
        "loss_history": [float(l) for l in loss_history],
        "pairs_compared_history": [int(p) for p in pairs_compared_history],
        "precision_history": [float(p) for p in precision_history],
        "recall_history": [float(r) for r in recall_history],
        "ucb_bonus_history": [float(b) for b in ucb_bonus_history],
        "hyperparameters": HP,
    }

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Precision: {final_precision:.3f} | Recall: {final_recall:.3f} | Time: {elapsed:.1f}s")
    print(f"  Saved to {out_path}")
    return results


def print_comparison_table(random_r, dqn_r, dqn_ucb_r):
    """Print final comparison table."""
    print("\n" + "=" * 65)
    print(f"{'Metric':<22} {'Random':>12} {'DQN':>12} {'DQN+UCB':>12}")
    print("-" * 65)

    def row(label, key, fmt=".3f"):
        vals = [random_r.get(key, 0), dqn_r.get(key, 0), dqn_ucb_r.get(key, 0)]
        print(f"{label:<22} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")

    row("Precision", "final_precision")
    row("Recall", "final_recall")
    row("F1", "final_f1")
    row("Efficiency (lower=better)", "efficiency_score")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run AIRA experiments")
    parser.add_argument(
        "--experiment",
        choices=["random", "dqn", "dqn_ucb", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--policy", default="data/raw/synthetic_policy.txt",
        help="Path to policy text file"
    )
    parser.add_argument(
        "--annotations", default="data/annotated/synthetic_contradictions.json",
        help="Path to annotations JSON"
    )
    parser.add_argument(
        "--doc-prefix", default="SYN",
        help="Clause ID prefix"
    )
    parser.add_argument(
        "--use-random-embeddings", action="store_true",
        help="Use random embeddings (no sentence-transformers required)"
    )
    args = parser.parse_args()

    print(f"Loading environment from: {args.policy}")
    env = load_env(
        args.policy,
        args.annotations,
        args.doc_prefix,
        use_random_embeddings=args.use_random_embeddings,
    )
    print(f"  Clauses: {len(env.clauses)}, Pairs: {env._n_pairs}, "
          f"True contradictions: {len(env.ground_truth_pairs)}, "
          f"Budget: {env._budget}")

    results = {}

    if args.experiment in ("random", "all"):
        results["random"] = run_random_experiment(env, "experiments/exp1_random")

    if args.experiment in ("dqn", "all"):
        results["dqn"] = run_dqn_experiment(env, "experiments/exp2_dqn", use_ucb=False)

    if args.experiment in ("dqn_ucb", "all"):
        results["dqn_ucb"] = run_dqn_experiment(env, "experiments/exp3_dqn_with_ucb", use_ucb=True)

    if args.experiment == "all" and len(results) == 3:
        print_comparison_table(results["random"], results["dqn"], results["dqn_ucb"])


if __name__ == "__main__":
    main()
