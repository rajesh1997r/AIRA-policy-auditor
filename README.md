# AIRA — Academic Integrity Risk Analyzer

An RL-powered system that audits academic policy documents for contradictions.
A DQN agent learns to identify which clause pairs are worth comparing, using UCB exploration to find contradictions efficiently — reviewing only **6.4% of all possible pairs** while achieving perfect recall.

---

## Problem Statement

University AI policies contain contradictory clauses that create inconsistent expectations for students. Manual review is subjective and slow. AIRA automates contradiction detection using reinforcement learning, learning to prioritize high-risk clause pairs within a fixed budget.

---

## Architecture

```
Policy Text (.txt)
      │
      ▼
 Segmenter ──────────────────────── Clause List [{id, text, section}]
      │
      ▼
 Embedder (all-MiniLM-L6-v2)        Embeddings (N × 384)
      │
      ▼
 ClauseAuditorEnv (Gymnasium)
 ┌────────────────────────────────────────────────────┐
 │ State: [emb_A | emb_B | cosine_sim | len_ratio |  │
 │          budget_used | step_frac]  = 772-dim       │
 │ Action: 0=SKIP, 1=COMPARE                          │
 │ Reward: +2.0 hit, -0.1 false pos, -0.5 miss, 0.0  │
 │ Budget: 15% of all pairs per episode               │
 └────────────────────────────────────────────────────┘
      │
      ▼
 DQN Agent (772→512→256→128→2)
 + UCB Explorer (c·√(ln t / N(i,j)) bonus on COMPARE)
      │
      ▼
 Audit Report: flagged contradictions + Precision/Recall/F1
```

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** S_t | 772-dim vector: [emb_A ‖ emb_B ‖ cosine_sim ‖ len_ratio ‖ budget_used_frac ‖ step_frac] |
| **Action** A_t | {0=SKIP, 1=COMPARE} |
| **Reward** R_t | +2.0 (true contradiction found), −0.1 (false COMPARE), −0.5 (missed contradiction), 0.0 (correct SKIP) |
| **Discount** γ | 0.99 |
| **Episode** | One document; terminates when budget (15% of pairs) exhausted |

**RL Methods:** Value-Based Learning (DQN) + Exploration Strategies (UCB1)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo with trained model
python demo/demo_policy_audit.py --use-random-embeddings \
  --model experiments/exp3_dqn_with_ucb/checkpoints/final.pt

# 3. Run all experiments
python experiments/run_experiments.py --experiment all --use-random-embeddings

# 4. Training notebook
jupyter notebook notebooks/02_training.ipynb

# 5. Analysis + figures
jupyter notebook notebooks/03_analysis.ipynb
```

---

## Results

| Metric | Random | Cosine Similarity | DQN | DQN+UCB (AIRA) |
|--------|--------|------------------|-----|----------------|
| Precision | 0.049 | 0.050 | 0.682 | **0.940** |
| Recall | 0.142 | 0.135 | 0.973 | **1.000** |
| F1 | 0.072 | 0.073 | 0.802 | **0.969** |
| Efficiency | 0.139 | 0.147 | 0.088 | **0.064** |

DQN+UCB finds every contradiction while reviewing only **6.4% of all pairs** — beating all baselines on every metric. The cosine similarity baseline (≈ random) proves that semantic similarity alone is insufficient: RL is required to learn which pairs are truly contradictory.

**Learning curve:** DQN+UCB converges to perfect recall by episode 400, with reward rising from −0.11 (Ep 1–100) to +15.84 (Ep 401–500).

---

## Datasets

| Policy | Clauses | Contradictions | Source |
|--------|---------|---------------|--------|
| `synthetic_policy.txt` | 17 | 8 | Synthetic (training/eval) |
| `northeastern_ai_policy.txt` | 14 | 6 | Representative NEU AI policy |
| `mit_ai_policy.txt` | 14 | 6 | Representative MIT AI policy |

All contradictions manually annotated with type labels (PERMISSION_CONFLICT, SCOPE_MISMATCH, CONSEQUENCE_CONFLICT, etc.).

---

## Repository Structure

```
TakeHomeFinalProject/
├── src/
│   ├── preprocessing/   segmenter.py, embedder.py
│   ├── env/             clause_auditor_env.py, reward.py
│   ├── agents/          dqn_agent.py, ucb_explorer.py, replay_buffer.py,
│   │                    audit_controller.py  ← orchestration layer
│   └── evaluation/      metrics.py (+ per-type breakdown), baselines.py
├── data/
│   ├── raw/             synthetic_policy.txt, northeastern_ai_policy.txt, mit_ai_policy.txt
│   └── annotated/       Ground truth contradiction pairs (JSON) for all 3 policies
├── experiments/
│   ├── run_experiments.py           Main training script
│   ├── eval_generalization.py       Zero-shot evaluation on NEU + MIT
│   ├── multi_seed_eval.py           5-seed statistical validation
│   ├── ucb_ablation.py              UCB c-value sensitivity sweep
│   ├── exp1_random/                 Random agent results
│   ├── exp2_dqn/                    DQN results + checkpoints
│   └── exp3_dqn_with_ucb/          DQN+UCB results + checkpoints
├── notebooks/
│   ├── 02_training.ipynb            Training loop with outputs
│   └── 03_analysis.ipynb            Figures + confusion matrix + per-type breakdown
├── results/figures/     learning_curve.png, comparison_bar.png,
│                        efficiency_tradeoff.png, ucb_bonus_decay.png,
│                        multi_seed_curve.png, ucb_ablation.png,
│                        confusion_matrix.png, per_type_breakdown.png
├── demo/
│   └── demo_policy_audit.py  CLI demo (uses AuditController)
├── MEMORY_BANK.md
└── requirements.txt
```

---

## Additional Experiments

Run after the main experiments for deeper analysis:

```bash
# Zero-shot generalization on NEU and MIT policies
python experiments/eval_generalization.py

# Multi-seed statistical validation (5 seeds × 500 episodes)
python experiments/multi_seed_eval.py

# UCB exploration constant ablation (c ∈ {0.0, 0.1, 0.3, 1.0, 1.41, 3.0})
python experiments/ucb_ablation.py

# Audit all three documents with cross-document pattern detection
python demo/demo_policy_audit.py --all-docs
```

---

## Ethical Considerations

**AIRA is a triage tool, not an autonomous judge.** Every pair flagged by the RL agent requires human review before any academic or disciplinary consequence is considered.

| Risk | Mitigation |
|------|-----------|
| False positives cause unwarranted scrutiny | Asymmetric reward (−0.1 FP vs −0.5 miss) keeps precision high; human review required |
| Synthetic training bias | Evaluated zero-shot on real NEU and MIT policies; generalization results reported |
| Embedding underperformance on non-English text | `all-MiniLM-L6-v2` is English-optimized; policy text in other languages requires a multilingual model |
| Distribution shift (new policy formats) | Confidence scores surface low-certainty decisions; periodic retraining is recommended |
| Disproportionate impact | Contradiction detection is document-level, not student-level — no individual profiling |

The 6.4% efficiency metric means 93.6% of clause pairs are never reviewed by the agent. This is intentional: the system is designed for **targeted flagging**, not exhaustive surveillance. An institution using AIRA should maintain human oversight for all flagged items and establish an appeal process before acting on automated findings.

---

## Course

INFO7375 — Reinforcement Learning for Agentic AI Systems
Northeastern University, Spring 2025
