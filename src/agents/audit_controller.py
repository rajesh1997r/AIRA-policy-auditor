"""
AuditController — Multi-Component Orchestration Layer for AIRA

Frames the AIRA pipeline as an agentic system with four specialized sub-agents:

  ┌─────────────────────────────────────────────────────────────┐
  │                     AuditController                         │
  │                                                             │
  │  SegmenterAgent → EmbedderAgent → RLAuditAgent → ReportAgent│
  │                                       ↕                     │
  │                              ContradictionMemory            │
  └─────────────────────────────────────────────────────────────┘

Each sub-agent has a defined role, input/output contract, and error handling.
The controller manages handoffs, validates inter-agent outputs, and accumulates
cross-document contradiction patterns in ContradictionMemory.

Usage:
    from src.agents.audit_controller import AuditController, AuditReport

    controller = AuditController(model_path='experiments/exp3_dqn_with_ucb/checkpoints/final.pt')

    # Single document audit
    report = controller.run_audit(
        policy_path='data/raw/northeastern_ai_policy.txt',
        annotations_path='data/annotated/northeastern_contradictions.json',
        doc_prefix='NEU'
    )
    print(report)

    # Multi-document audit with cross-document pattern detection
    reports = controller.run_cross_document_audit([
        {'policy_path': 'data/raw/northeastern_ai_policy.txt',
         'annotations_path': 'data/annotated/northeastern_contradictions.json',
         'doc_prefix': 'NEU'},
        {'policy_path': 'data/raw/mit_ai_policy.txt',
         'annotations_path': 'data/annotated/mit_contradictions.json',
         'doc_prefix': 'MIT'},
    ])
    patterns = controller.memory.get_cross_document_patterns()
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

import numpy as np

from src.env.reward import COMPARE, SKIP


# ─── Data classes ──────────────────────────────────────────────────────────

@dataclass
class FlaggedPair:
    """A clause pair flagged by the RL agent as a potential contradiction."""
    clause_a_id: str
    clause_b_id: str
    clause_a_text: str
    clause_b_text: str
    confidence: float         # Q-value difference, normalized to [0, 1]
    is_true_contradiction: bool  # Only available when ground truth is known


@dataclass
class AuditReport:
    """Result of a single document audit."""
    doc_name: str
    doc_prefix: str
    n_clauses: int
    n_pairs: int
    n_contradictions_known: int      # from ground truth annotations
    flagged_pairs: List[FlaggedPair]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    efficiency: float                # pairs_reviewed / total_pairs
    elapsed_seconds: float
    model_path: str

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  AIRA Audit Report — {self.doc_name}",
            "=" * 60,
            f"  Clauses    : {self.n_clauses}",
            f"  Total pairs: {self.n_pairs}",
            f"  Efficiency : {self.efficiency:.1%} reviewed ({int(self.efficiency * self.n_pairs)} pairs)",
            f"  Audit time : {self.elapsed_seconds:.2f}s",
            "",
            f"  CONTRADICTIONS FOUND ({len([f for f in self.flagged_pairs if f.is_true_contradiction])}):",
        ]
        for i, fp in enumerate(
            sorted([f for f in self.flagged_pairs if f.is_true_contradiction],
                   key=lambda x: -x.confidence), 1
        ):
            lines.append(f"    [{i}] {fp.clause_a_id} vs {fp.clause_b_id} "
                         f"(conf={fp.confidence:.2f})")
        if any(not f.is_true_contradiction for f in self.flagged_pairs):
            lines.append(
                f"\n  FALSE POSITIVES: {self.false_positives}"
            )
        lines += [
            "",
            "  METRICS:",
            f"    Precision : {self.precision:.3f}",
            f"    Recall    : {self.recall:.3f}",
            f"    F1        : {self.f1:.3f}",
            f"    Efficiency: {self.efficiency:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─── Sub-agents ────────────────────────────────────────────────────────────

class SegmenterAgent:
    """
    Decomposes raw policy text into discrete clauses.

    Role: Text preprocessing and clause extraction.
    Input: Raw policy text (str), document prefix (str).
    Output: List of clause dicts (id, text, source_doc, section).
    """

    def segment(self, policy_path: str, doc_prefix: str) -> List[Dict]:
        from src.preprocessing.segmenter import segment_policy
        with open(policy_path) as f:
            text = f.read()
        clauses = segment_policy(text, doc_prefix)
        if not clauses:
            raise ValueError(
                f"SegmenterAgent: No clauses extracted from {policy_path}. "
                "Check that the policy file contains at least one paragraph > 15 tokens."
            )
        return clauses


class EmbedderAgent:
    """
    Converts clauses to dense semantic embeddings.

    Role: Embedding computation with caching.
    Input: List of clause dicts, document name.
    Output: np.ndarray of shape (N, 384), L2-normalized.
    """

    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        self._embedder = None  # Lazy init — model load is slow

    def _get_embedder(self):
        if self._embedder is None:
            from src.preprocessing.embedder import ClauseEmbedder
            self._embedder = ClauseEmbedder(cache_dir=self.cache_dir)
        return self._embedder

    def embed(self, clauses: List[Dict], doc_name: str) -> np.ndarray:
        embs = self._get_embedder().embed_clauses(clauses, doc_name=doc_name)
        assert embs.shape == (len(clauses), 384), (
            f"EmbedderAgent: Expected shape ({len(clauses)}, 384), got {embs.shape}"
        )
        return embs


class RLAuditAgent:
    """
    Core RL decision-making agent (DQN + UCB1).

    Role: Budget-constrained clause pair prioritization.
    Input: ClauseAuditorEnv initialized with clauses and embeddings.
    Output: List of FlaggedPair for pairs deemed COMPARE.
    """

    def __init__(self, model_path: str, budget_fraction: float = 0.15):
        self.model_path = model_path
        self.budget_fraction = budget_fraction
        self._dqn = None
        self._ucb = None

    def _load_agents(self, n_pairs: int):
        """Lazy-load agents once we know n_pairs (needed for UCB visit counts)."""
        from src.agents.dqn_agent import DQNAgent
        from src.agents.ucb_explorer import UCBExplorer

        dqn = DQNAgent(state_dim=772, n_actions=2)
        ucb = UCBExplorer(dqn, n_pairs=n_pairs, c=0.3, warm_start_episodes=0)

        if self.model_path and os.path.exists(self.model_path):
            ucb.load(self.model_path)
        else:
            # Still runs — just with untrained weights
            pass

        self._dqn = dqn
        self._ucb = ucb

    def run_episode(self, env, seed: int = 42) -> List[FlaggedPair]:
        from src.agents.ucb_explorer import UCBExplorer

        self._load_agents(env._n_pairs)
        self._ucb.reset()
        obs, info = env.reset(seed=seed)
        flagged = []

        while True:
            i, j = env._pair_queue[env._queue_idx]
            n = len(env.clauses)
            flat_idx = UCBExplorer.pair_to_idx(min(i, j), max(i, j), n)
            action = self._ucb.select_action(obs, flat_idx)

            q_vals = self._dqn.get_q_values(obs)
            confidence = float(
                np.clip((q_vals[COMPARE] - q_vals[SKIP]) / (abs(q_vals[COMPARE]) + 1e-6), 0.0, 1.0)
            )

            next_obs, reward, terminated, truncated, step_info = env.step(action)

            if action == COMPARE:
                clause_i = env._pair_queue[env._queue_idx - 1][0]
                clause_j = env._pair_queue[env._queue_idx - 1][1]
                flagged.append(FlaggedPair(
                    clause_a_id=step_info["pair"][0],
                    clause_b_id=step_info["pair"][1],
                    clause_a_text=env.clauses[clause_i]["text"],
                    clause_b_text=env.clauses[clause_j]["text"],
                    confidence=confidence,
                    is_true_contradiction=step_info["is_contradiction"],
                ))

            obs = next_obs
            if terminated or truncated:
                break

        return flagged


class ReportAgent:
    """
    Formats audit results into a structured AuditReport.

    Role: Metric computation and report assembly.
    Input: Flagged pairs, ground truth, env metadata.
    Output: AuditReport dataclass.
    """

    def format(
        self,
        doc_name: str,
        doc_prefix: str,
        env,
        flagged: List[FlaggedPair],
        gt_pairs: Set[frozenset],
        elapsed: float,
        model_path: str,
    ) -> AuditReport:
        flagged_keys = {frozenset({f.clause_a_id, f.clause_b_id}) for f in flagged}
        tp = len(flagged_keys & gt_pairs)
        fp = len(flagged_keys - gt_pairs)
        fn = len(gt_pairs - flagged_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        efficiency = len(flagged_keys) / max(env._n_pairs, 1)

        return AuditReport(
            doc_name=doc_name,
            doc_prefix=doc_prefix,
            n_clauses=len(env.clauses),
            n_pairs=env._n_pairs,
            n_contradictions_known=len(gt_pairs),
            flagged_pairs=flagged,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            efficiency=efficiency,
            elapsed_seconds=elapsed,
            model_path=model_path,
        )


# ─── Memory ────────────────────────────────────────────────────────────────

class ContradictionMemory:
    """
    Accumulates contradiction patterns across multiple document audits.

    Tracks which contradiction types appear repeatedly across documents,
    enabling cross-document pattern detection (e.g., PERMISSION_CONFLICT
    appears in both NEU and MIT policies).
    """

    def __init__(self):
        self._doc_flags: Dict[str, List[FlaggedPair]] = {}
        self._doc_types: Dict[str, Dict[str, List]] = {}  # doc -> type -> [pairs]

    def store(self, doc_prefix: str, flagged: List[FlaggedPair], annotations: list):
        """Store flagged pairs and map to contradiction types from annotations."""
        self._doc_flags[doc_prefix] = flagged

        flagged_keys = {frozenset({f.clause_a_id, f.clause_b_id}) for f in flagged
                        if f.is_true_contradiction}
        type_map: Dict[str, List] = {}
        for item in annotations:
            ctype = item.get("type", item.get("description", "UNKNOWN"))
            key = frozenset({item["clause_a"], item["clause_b"]})
            if key in flagged_keys:
                type_map.setdefault(ctype, []).append(key)

        self._doc_types[doc_prefix] = type_map

    def get_cross_document_patterns(self) -> Dict[str, List[str]]:
        """
        Find contradiction types that appear in multiple documents.

        Returns:
            dict mapping type_string → list of doc_prefixes where it was found.
        """
        type_to_docs: Dict[str, List[str]] = {}
        for doc, type_map in self._doc_types.items():
            for ctype in type_map:
                type_to_docs.setdefault(ctype, []).append(doc)

        return {k: v for k, v in type_to_docs.items() if len(v) > 1}

    def summary(self) -> str:
        lines = [f"\nContradictionMemory: {len(self._doc_flags)} documents audited"]
        for doc, type_map in self._doc_types.items():
            total = sum(len(v) for v in type_map.values())
            lines.append(f"  [{doc}]: {total} TP found — types: {list(type_map.keys())}")
        patterns = self.get_cross_document_patterns()
        if patterns:
            lines.append("\nCross-document patterns (repeated contradiction types):")
            for ctype, docs in patterns.items():
                lines.append(f"  {ctype}: seen in {docs}")
        return "\n".join(lines)


# ─── AuditController ───────────────────────────────────────────────────────

class AuditController:
    """
    Orchestrates the full AIRA policy audit pipeline.

    Components:
      SegmenterAgent  — policy text → clauses
      EmbedderAgent   — clauses → embeddings
      RLAuditAgent    — embeddings + MDP → flagged pairs
      ReportAgent     — flagged pairs → AuditReport
      ContradictionMemory — cross-document pattern accumulation

    Args:
        model_path: Path to trained DQN+UCB checkpoint.
        budget_fraction: Fraction of clause pairs to review per episode.
        cache_dir: Directory for embedding cache.
    """

    def __init__(
        self,
        model_path: str = "experiments/exp3_dqn_with_ucb/checkpoints/final.pt",
        budget_fraction: float = 0.15,
        cache_dir: str = "data/processed",
    ):
        self.model_path = model_path
        self.budget_fraction = budget_fraction

        self.segmenter = SegmenterAgent()
        self.embedder = EmbedderAgent(cache_dir=cache_dir)
        self.rl_agent = RLAuditAgent(model_path=model_path, budget_fraction=budget_fraction)
        self.reporter = ReportAgent()
        self.memory = ContradictionMemory()

    def run_audit(
        self,
        policy_path: str,
        annotations_path: str,
        doc_prefix: str,
        seed: int = 42,
        verbose: bool = False,
    ) -> AuditReport:
        """
        Run a complete audit on one policy document.

        Args:
            policy_path: Path to plain-text policy file.
            annotations_path: Path to JSON annotations (ground truth).
            doc_prefix: Clause ID prefix (e.g., 'NEU', 'MIT', 'SYN').
            seed: Random seed for episode reproducibility.
            verbose: Print progress to stdout.

        Returns:
            AuditReport with all metrics and flagged pairs.
        """
        from src.env.clause_auditor_env import ClauseAuditorEnv

        doc_name = os.path.splitext(os.path.basename(policy_path))[0]
        if verbose:
            print(f"\n[AuditController] Auditing: {doc_name}")

        # Step 1: Segmentation
        try:
            clauses = self.segmenter.segment(policy_path, doc_prefix)
        except ValueError as e:
            raise RuntimeError(f"Segmentation failed: {e}") from e
        if verbose:
            print(f"  SegmenterAgent: {len(clauses)} clauses extracted")

        # Step 2: Embedding
        embeddings = self.embedder.embed(clauses, doc_name=doc_name)
        if verbose:
            print(f"  EmbedderAgent: embeddings shape {embeddings.shape}")

        # Step 3: Load ground truth
        with open(annotations_path) as f:
            ann = json.load(f)
        gt_pairs = {frozenset({c["clause_a"], c["clause_b"]}) for c in ann["contradictions"]}

        # Step 4: Build env
        env = ClauseAuditorEnv(
            clauses=clauses,
            embeddings=embeddings,
            ground_truth_pairs=gt_pairs,
            budget_fraction=self.budget_fraction,
        )

        # Step 5: RL audit episode
        t0 = time.time()
        flagged = self.rl_agent.run_episode(env, seed=seed)
        elapsed = time.time() - t0
        if verbose:
            print(f"  RLAuditAgent: {len(flagged)} pairs flagged in {elapsed:.2f}s")

        # Step 6: Format report
        report = self.reporter.format(
            doc_name=doc_name,
            doc_prefix=doc_prefix,
            env=env,
            flagged=flagged,
            gt_pairs=gt_pairs,
            elapsed=elapsed,
            model_path=self.model_path,
        )

        # Step 7: Update memory
        self.memory.store(doc_prefix, flagged, ann["contradictions"])
        if verbose:
            print(f"  ReportAgent: F1={report.f1:.3f}  Recall={report.recall:.3f}")

        return report

    def run_cross_document_audit(
        self,
        policy_configs: List[Dict],
        seed: int = 42,
        verbose: bool = True,
    ) -> List[AuditReport]:
        """
        Audit multiple policy documents and detect cross-document patterns.

        Args:
            policy_configs: List of dicts, each with keys:
                policy_path, annotations_path, doc_prefix.
            seed: Seed for all episodes.
            verbose: Print per-document progress.

        Returns:
            List of AuditReport, one per document.
        """
        reports = []
        for cfg in policy_configs:
            report = self.run_audit(
                policy_path=cfg["policy_path"],
                annotations_path=cfg["annotations_path"],
                doc_prefix=cfg["doc_prefix"],
                seed=seed,
                verbose=verbose,
            )
            reports.append(report)
            if verbose:
                print(report)

        if verbose and len(reports) > 1:
            print(self.memory.summary())

        return reports


# ─── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIRA AuditController — single or multi-doc audit")
    parser.add_argument("--policy", default="data/raw/synthetic_policy.txt")
    parser.add_argument("--annotations", default="data/annotated/synthetic_contradictions.json")
    parser.add_argument("--doc-prefix", default="SYN")
    parser.add_argument("--model", default="experiments/exp3_dqn_with_ucb/checkpoints/final.pt")
    parser.add_argument("--all-docs", action="store_true", help="Audit all three policy documents")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    controller = AuditController(model_path=args.model)

    if args.all_docs:
        reports = controller.run_cross_document_audit(
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
        report = controller.run_audit(
            policy_path=args.policy,
            annotations_path=args.annotations,
            doc_prefix=args.doc_prefix,
            verbose=args.verbose,
        )
        print(report)
