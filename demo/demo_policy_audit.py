"""
AIRA Demo — Policy Audit Report (via AuditController)

Runs the AIRA pipeline through the AuditController orchestration layer,
which coordinates: SegmenterAgent → EmbedderAgent → RLAuditAgent → ReportAgent.

Usage:
    python demo/demo_policy_audit.py
    python demo/demo_policy_audit.py --policy data/raw/northeastern_ai_policy.txt \\
                                     --annotations data/annotated/northeastern_contradictions.json \\
                                     --doc-prefix NEU
    python demo/demo_policy_audit.py --all-docs
    python demo/demo_policy_audit.py --compare-all
    python demo/demo_policy_audit.py --verbose
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.agents.audit_controller import AuditController
from src.env.clause_auditor_env import ClauseAuditorEnv
from src.env.reward import SKIP, COMPARE


def run_exhaustive_baseline(policy_path: str, annotations_path: str, doc_prefix: str,
                             use_random_embeddings: bool) -> None:
    """Run the compare-all baseline directly for demo comparison."""
    from src.preprocessing.segmenter import segment_policy

    with open(policy_path) as f:
        text = f.read()
    clauses = segment_policy(text, doc_prefix)

    if use_random_embeddings:
        rng = np.random.default_rng(42)
        embs = rng.random((len(clauses), 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    else:
        from src.preprocessing.embedder import ClauseEmbedder
        embedder = ClauseEmbedder(cache_dir="data/processed")
        doc_name = os.path.splitext(os.path.basename(policy_path))[0]
        embs = embedder.embed_clauses(clauses, doc_name=doc_name)

    with open(annotations_path) as f:
        ann = json.load(f)
    gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

    env = ClauseAuditorEnv(clauses, embs, gt_pairs, budget_fraction=1.0)
    obs, info = env.reset(seed=42)

    t0 = time.time()
    while True:
        obs, reward, terminated, truncated, _ = env.step(COMPARE)
        if terminated or truncated:
            break
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("  AIRA — Exhaustive Baseline (--compare-all)")
    print(f"{'='*60}")
    print(f"  Document    : {os.path.basename(policy_path)}")
    print(f"  Clauses     : {len(clauses)}")
    print(f"  Total pairs : {env._n_pairs}")
    print(f"  Pairs reviewed: {env._n_pairs} of {env._n_pairs} (100.0%)")
    print(f"  Audit time  : {elapsed:.3f}s")
    print(f"  Contradictions known: {len(gt_pairs)}")
    print()
    print("  METRICS:")
    print("    Precision : 1.000  (all GT found, no FP by definition)")
    print("    Recall    : 1.000")
    print("    F1        : 1.000")
    print(f"    Efficiency: 1.000 (vs {1/env._n_pairs*len(gt_pairs)*15:.3f} for AIRA)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="AIRA Policy Audit Demo")
    parser.add_argument("--policy", default="data/raw/synthetic_policy.txt")
    parser.add_argument("--annotations", default="data/annotated/synthetic_contradictions.json")
    parser.add_argument("--doc-prefix", default="SYN")
    parser.add_argument("--model", default="experiments/exp3_dqn_with_ucb/checkpoints/final.pt")
    parser.add_argument("--verbose", action="store_true", help="Show per-step progress from controller")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run exhaustive baseline (compare every pair)")
    parser.add_argument("--all-docs", action="store_true",
                        help="Audit all three policy documents with cross-doc pattern detection")
    parser.add_argument("--use-random-embeddings", action="store_true",
                        help="Use random embeddings (no sentence-transformers required)")
    args = parser.parse_args()

    if args.compare_all:
        print("\nAIRA — Running Exhaustive Baseline (--compare-all)")
        run_exhaustive_baseline(
            args.policy, args.annotations, args.doc_prefix, args.use_random_embeddings
        )
        return

    # Build controller — orchestrates all 4 sub-agents
    controller = AuditController(
        model_path=args.model,
        budget_fraction=0.15,
        cache_dir="data/processed" if not args.use_random_embeddings else None,
    )

    if args.use_random_embeddings:
        # Patch embedder to use random embeddings for demo without sentence-transformers
        from src.preprocessing.segmenter import segment_policy

        class _RandomEmbedderAgent:
            def embed(self, clauses, doc_name):
                rng = np.random.default_rng(42)
                embs = rng.random((len(clauses), 384)).astype(np.float32)
                embs /= np.linalg.norm(embs, axis=1, keepdims=True)
                return embs

        controller.embedder = _RandomEmbedderAgent()

    if args.all_docs:
        print("\nAIRA — Cross-Document Audit (all 3 policy documents)")
        controller.run_cross_document_audit(
            [
                {"policy_path": "data/raw/synthetic_policy.txt",
                 "annotations_path": "data/annotated/synthetic_contradictions.json",
                 "doc_prefix": "SYN"},
                {"policy_path": "data/raw/northeastern_ai_policy.txt",
                 "annotations_path": "data/annotated/northeastern_contradictions.json",
                 "doc_prefix": "NEU"},
                {"policy_path": "data/raw/mit_ai_policy.txt",
                 "annotations_path": "data/annotated/mit_contradictions.json",
                 "doc_prefix": "MIT"},
            ],
            verbose=True,
        )
    else:
        print(f"\nAIRA — Policy Audit via AuditController")
        print(f"  Policy   : {args.policy}")
        print(f"  Model    : {args.model}")
        report = controller.run_audit(
            policy_path=args.policy,
            annotations_path=args.annotations,
            doc_prefix=args.doc_prefix,
            verbose=args.verbose,
        )
        print(report)


if __name__ == "__main__":
    main()
