# AIRA — Memory Bank

## Project Status: 🟢 NEARLY COMPLETE — PDF + Video remaining

---

## What Is AIRA?
Academic Integrity Risk Analyzer — an RL agent that audits policy documents for contradictions.
- **RL Method 1**: DQN (Value-Based Learning) — decides COMPARE vs SKIP for clause pairs
- **RL Method 2**: UCB1 (Exploration Strategies) — bonus on under-explored pairs

---

## Day 1 Checklist — Data + Environment
- [x] Project skeleton (dirs, requirements.txt, .gitignore)
- [x] `src/preprocessing/segmenter.py`
- [x] `src/preprocessing/embedder.py`
- [x] `data/raw/synthetic_policy.txt` (17 clauses, 8 contradictions)
- [x] `data/annotated/synthetic_contradictions.json`
- [x] `src/env/reward.py`
- [x] `src/env/clause_auditor_env.py`
- [x] `src/evaluation/metrics.py`
- [x] `src/evaluation/baselines.py`
- [x] Smoke test: random episode runs end-to-end

## Day 2 Checklist — Agent
- [x] `src/agents/replay_buffer.py`
- [x] `src/agents/dqn_agent.py`
- [x] `src/agents/ucb_explorer.py`
- [x] DQN sanity check: training loop verified (reward improves over episodes)

## Day 3 Checklist — Experiments
- [x] `experiments/run_experiments.py`
- [x] Run Experiment 1: Random baseline
- [x] Run Experiment 2: DQN (500 episodes)
- [x] Run Experiment 3: DQN+UCB (500 episodes, c=0.3)
- [x] `notebooks/02_training.ipynb` — executed, all cells have real output
- [x] `notebooks/03_analysis.ipynb` — executed, all 4 figures embedded inline
- [x] Generate 4 figures to `results/figures/`
- [x] `data/raw/northeastern_ai_policy.txt` (14 clauses, 6 contradictions)
- [x] `data/raw/mit_ai_policy.txt` (14 clauses, 6 contradictions)
- [x] `data/annotated/northeastern_contradictions.json`
- [x] `data/annotated/mit_contradictions.json`
- [x] `CosineSimilarityBaseline` added to `src/evaluation/baselines.py`

## Day 4 Checklist — Deliverables
- [x] `demo/demo_policy_audit.py` — runs with trained model, produces formatted report
- [x] `README.md` — updated with real results, baselines table, repo structure
- [ ] Technical report PDF
- [ ] Loom demo video (10 min)

---

## Key Numbers (final results — avg over last 50 episodes)

| Metric | Random | Cosine Sim | DQN | DQN+UCB |
|--------|--------|-----------|-----|---------|
| Precision | 0.049 | 0.050 | 0.682 | **0.940** |
| Recall | 0.142 | 0.135 | 0.973 | **1.000** |
| F1 | 0.072 | 0.073 | 0.802 | **0.969** |
| Efficiency | 0.139 | 0.147 | 0.088 | **0.064** |

Cosine Similarity ≈ Random → proves RL is necessary (not just semantic similarity).

**DQN+UCB learning curve milestones:**
- Ep 1-100:   reward=-0.11,  recall=0.356
- Ep 201-300: reward=+2.06,  recall=0.388
- Ep 301-400: reward=+12.87, recall=0.876
- Ep 401-500: reward=+15.84, recall=**1.000** ← converged

---

## Architecture
```
State: [emb_A(384) | emb_B(384) | cosine_sim | len_ratio | budget_used | step_frac] = 772-dim
Action: 0=SKIP, 1=COMPARE
Reward: +2.0 true hit, -0.1 false COMPARE, -0.5 missed, 0.0 correct skip
Network: 772 → 512(BN) → 256(BN) → 128 → 2
```

## MDP Formulation
- **State** S_t: feature vector of current clause pair
- **Action** A_t ∈ {0=SKIP, 1=COMPARE}
- **Reward** R_t: shaped to favor finding contradictions within budget
- **Discount** γ = 0.99
- **Episode**: one document, terminates when budget (15% of pairs) exhausted

## Hyperparameters (final)
```python
EPISODES = 500
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE_STEPS = 100
MIN_REPLAY_SIZE = 500
BUDGET_FRACTION = 0.15
UCB_C = 0.3          # tuned down from 1.41; prevents bonus overpowering Q-values
UCB_WARM_START = 10
```

## Bug Fixed (UCB visit counts)
**Problem**: `UCBExplorer.reset()` was wiping `_visit_counts` and `_total_steps` every episode,
making UCB exploration effectively memoryless across episodes.
**Fix**: `reset()` now only increments `_episode_count`. Counts accumulate across all 500 episodes,
giving true UCB1 convergence from exploration to exploitation.

## Datasets
| File | Clauses | Contradictions | Types |
|------|---------|---------------|-------|
| `data/raw/synthetic_policy.txt` | 17 | 8 | PERMISSION_CONFLICT, SCOPE_MISMATCH, DISCLOSURE_INCONSISTENCY |
| `data/raw/northeastern_ai_policy.txt` | 14 | 6 | PERMISSION_CONFLICT, DISCLOSURE_INCONSISTENCY, SCOPE_MISMATCH, ACCOMMODATION_CONFLICT, CONSEQUENCE_CONFLICT, COLLABORATION_CONFLICT |
| `data/raw/mit_ai_policy.txt` | 14 | 6 | PERMISSION_CONFLICT, ATTRIBUTION_CONFLICT, TOOL_CONFLICT, SCOPE_MISMATCH, RETENTION_CONFLICT, CONSEQUENCE_CONFLICT |

## Baselines (4 total)
- **Random**: 15% COMPARE probability — floor baseline
- **Cosine Similarity**: top-k pairs by embedding similarity — non-RL heuristic baseline
- **Exhaustive**: always COMPARE — upper bound on recall (100%)
- **DQN**: learned Q-values, no exploration bonus
- **DQN+UCB**: learned Q-values + UCB1 exploration — best overall

## Trained Checkpoints
- `experiments/exp2_dqn/checkpoints/final.pt` — standalone DQN
- `experiments/exp3_dqn_with_ucb/checkpoints/final.pt` — DQN+UCB (use this for demo)

## Figures (results/figures/)
- `learning_curve.png` — reward + recall over 500 episodes (DQN vs DQN+UCB)
- `comparison_bar.png` — Precision/Recall/F1 for Random, DQN, DQN+UCB
- `efficiency_tradeoff.png` — scatter: efficiency vs recall (ideal = top-left)
- `ucb_bonus_decay.png` — UCB bonus decreasing as visit counts accumulate

## Estimated Score
86/100 — gaps are ablation studies and PDF/video deliverables
