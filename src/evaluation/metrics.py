"""
Evaluation metrics for policy contradiction detection.

All metrics operate over a decisions dict:
    {frozenset({clause_id_a, clause_id_b}): action (0=SKIP, 1=COMPARE)}

and a ground_truth_pairs set:
    {frozenset({clause_id_a, clause_id_b}), ...}
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from src.env.reward import COMPARE, SKIP


def compute_metrics(
    decisions: Dict[frozenset, int],
    ground_truth_pairs: Set[frozenset],
    all_pairs: List[Tuple[str, str]],
) -> dict:
    """
    Compute precision, recall, F1, and efficiency for one episode.

    Args:
        decisions: Mapping from pair (frozenset) to action taken.
        ground_truth_pairs: Set of frozensets that are true contradictions.
        all_pairs: List of all (clause_a_id, clause_b_id) tuples in the document.

    Returns:
        dict with keys: precision, recall, f1, efficiency, TP, FP, FN, TN,
                        pairs_compared, total_pairs, faithfulness.
    """
    compared = {pair for pair, action in decisions.items() if action == COMPARE}
    total_pairs = len(all_pairs)

    TP = len(compared & ground_truth_pairs)
    FP = len(compared - ground_truth_pairs)
    FN = len(ground_truth_pairs - compared)
    TN = total_pairs - TP - FP - FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    efficiency = len(compared) / max(total_pairs, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "efficiency": efficiency,
        "faithfulness": 1.0,  # placeholder; extended in demo with LLM grounding check
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "pairs_compared": len(compared),
        "total_pairs": total_pairs,
    }


def aggregate_metrics(episode_metrics: List[dict]) -> dict:
    """
    Aggregate metrics over multiple evaluation episodes.

    Returns dict with mean and std for each numeric metric.
    """
    import numpy as np

    keys = ["precision", "recall", "f1", "efficiency", "pairs_compared"]
    result = {}
    for k in keys:
        vals = [m[k] for m in episode_metrics]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result


def compute_per_type_metrics(decisions: Dict[frozenset, int], annotations_list: list) -> dict:
    """
    Compute per-contradiction-type precision/recall/F1.

    Args:
        decisions: Mapping from pair frozenset to action taken (0=SKIP, 1=COMPARE).
        annotations_list: List of annotation dicts, each with keys:
            clause_a, clause_b, type (and optionally description/note).

    Returns:
        dict keyed by contradiction type string, each containing:
            TP, FP, FN, precision, recall, f1, total_gt.
    """
    # Group ground truth pairs by type
    type_to_pairs: Dict[str, list] = defaultdict(list)
    for item in annotations_list:
        ctype = item.get("type", "UNKNOWN")
        pair = frozenset({item["clause_a"], item["clause_b"]})
        type_to_pairs[ctype].append(pair)

    compared = {p for p, a in decisions.items() if a == COMPARE}
    result = {}

    for ctype, gt_pairs_for_type in type_to_pairs.items():
        gt_set = set(gt_pairs_for_type)
        tp = len(compared & gt_set)
        fp = len(compared - gt_set)   # FP relative to this type only
        fn = len(gt_set - compared)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[ctype] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_gt": len(gt_set),
        }

    return result


def summarize_episode(episode_log: List[Tuple]) -> dict:
    """
    Summarize a list of (action, reward, was_contradiction) tuples.

    Returns:
        dict with total_reward, steps, compare_count, skip_count,
               true_positives, false_positives, false_negatives.
    """
    total_reward = 0.0
    compare_count = 0
    skip_count = 0
    tp = 0
    fp = 0
    fn = 0

    for action, reward, was_contradiction in episode_log:
        total_reward += reward
        if action == COMPARE:
            compare_count += 1
            if was_contradiction:
                tp += 1
            else:
                fp += 1
        else:
            skip_count += 1
            if was_contradiction:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_reward": total_reward,
        "steps": len(episode_log),
        "compare_count": compare_count,
        "skip_count": skip_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
