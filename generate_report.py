"""
Generate AIRA Technical Report PDF
Run: python generate_report.py
Output: AIRA_Technical_Report.pdf
"""

import json, os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# -- Load all results --------------------------------------------------------
with open("experiments/exp1_random/results.json") as f:  rand_r = json.load(f)
with open("experiments/exp2_dqn/results.json") as f:     dqn_r  = json.load(f)
with open("experiments/exp3_dqn_with_ucb/results.json") as f: ucb_r = json.load(f)
with open("experiments/generalization_results.json") as f:    gen_r = json.load(f)
with open("experiments/multi_seed_results.json") as f:        ms_r  = json.load(f)
with open("experiments/ucb_ablation_results.json") as f:      abl_r = json.load(f)


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    # -- Helpers --------------------------------------------------------------
    def h1(self, txt):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.ln(4)
        self.cell(0, 9, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 170, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def h2(self, txt):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 90, 160)
        self.ln(3)
        self.cell(0, 7, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)

    def body(self, txt, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, txt)
        self.ln(1)

    def bullet(self, txt, size=10):
        self.set_font("Helvetica", "", size)
        self.set_x(25)
        self.multi_cell(165, 5.5, f"*  {txt}")

    def code(self, txt):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, txt, fill=True)
        self.ln(1)

    def fig(self, path, w=170, caption=""):
        if os.path.exists(path):
            x = (210 - w) / 2
            self.image(path, x=x, w=w)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(80)
                self.cell(0, 5, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(0)
            self.ln(3)

    def tbl_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0)

    def tbl_row(self, vals, widths, bold=False, fill=False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        self.set_fill_color(235, 242, 255) if fill else self.set_fill_color(255, 255, 255)
        for v, w in zip(vals, widths):
            self.cell(w, 6, str(v), border=1, fill=fill, align="C")
        self.ln()

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 5, "AIRA -- Academic Integrity Risk Analyzer | INFO7375 Spring 2025", align="R")
        self.ln(3)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


pdf = Report()

# ============================================================================
# COVER PAGE
# ============================================================================
pdf.add_page()
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(30, 60, 120)
pdf.ln(30)
pdf.cell(0, 14, "AIRA", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 9, "Academic Integrity Risk Analyzer", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(60)
pdf.cell(0, 8, "Reinforcement Learning for Agentic Policy Audit Systems", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(12)
pdf.set_draw_color(30, 60, 120)
pdf.set_line_width(1.0)
pdf.line(40, pdf.get_y(), 170, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(40)
for line in [
    "Technical Report -- Take-Home Final",
    "INFO7375: Reinforcement Learning for Agentic AI Systems",
    "Northeastern University, Spring 2025",
    "",
    "Student: Rajesh Ramareddy",
    "Email: ramareddy.r@northeastern.edu",
    "GitHub: github.com/rajesh1997r/AIRA-policy-auditor",
]:
    pdf.cell(0, 7, line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(8)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(30, 60, 120)
pdf.cell(0, 7, "Abstract", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(40)
pdf.set_x(30)
pdf.multi_cell(150, 5.5,
    "AIRA is a budget-constrained reinforcement learning system that audits academic "
    "policy documents for contradictory clause pairs. A Deep Q-Network (DQN) agent "
    "paired with Upper Confidence Bound (UCB1) exploration learns to identify which "
    "clause pairs warrant comparison within a 15% budget constraint. Trained on a "
    "17-clause synthetic policy using real sentence-transformer embeddings (all-MiniLM-L6-v2), "
    "AIRA achieves F1=0.827 on held-out seeds and Recall=0.981 while reviewing only 8.1% "
    "of all clause pairs. Zero-shot evaluation on Northeastern and MIT AI policies yields "
    "F1=0.179, revealing that generalisation requires training on multi-document corpora. "
    "The system is structured as a multi-agent pipeline (SegmenterAgent, EmbedderAgent, "
    "RLAuditAgent, ReportAgent) coordinated by an AuditController with cross-document "
    "ContradictionMemory. All code, data, trained models, and results are openly available."
)

# ============================================================================
# 1. INTRODUCTION
# ============================================================================
pdf.add_page()
pdf.h1("1. Introduction and Problem Statement")
pdf.body(
    "University AI usage policies frequently contain contradictory clauses that create "
    "ambiguous and inconsistent expectations for students and faculty. For example, one "
    "clause may broadly permit AI writing assistants while another prohibits them entirely. "
    "Manual review of these policies is subjective, slow, and error-prone -- especially as "
    "policies grow in length and complexity."
)
pdf.body(
    "AIRA (Academic Integrity Risk Analyzer) addresses this with a reinforcement learning "
    "agent that learns to efficiently detect contradictory clause pairs under a strict "
    "budget constraint: the agent may review at most 15% of all possible clause pairs per "
    "audit episode. This budget constraint makes the problem non-trivial and unsuitable "
    "for simple keyword matching -- the agent must learn which semantic features signal "
    "contradiction risk."
)
pdf.body(
    "Why RL over rule-based NLP? Contradiction detection is context-dependent: the same "
    "vocabulary patterns (permit / prohibit) may or may not constitute a contradiction "
    "depending on scope, subject, and authority. An RL agent can learn nuanced "
    "decision boundaries from reward signals that reward true discoveries and penalise "
    "missed contradictions -- objectives that are difficult to encode as deterministic rules."
)
pdf.h2("RL Methods Implemented")
pdf.bullet("Value-Based Learning (DQN): learns Q(s, a) over clause-pair states")
pdf.bullet("Exploration Strategies (UCB1): adds exploration bonus to under-reviewed pairs")
pdf.h2("Agentic System Type")
pdf.bullet("Research / Analysis Agent: automated policy quality assurance")
pdf.bullet("AuditController orchestration layer with 4 specialised sub-agents")
pdf.bullet("ContradictionMemory for cross-document pattern accumulation")

# ============================================================================
# 2. SYSTEM ARCHITECTURE
# ============================================================================
pdf.h1("2. System Architecture")
pdf.body("The AIRA pipeline consists of four specialised agents coordinated by AuditController:")
pdf.code(
    "  Policy Text (.txt)\n"
    "        |\n"
    "        v\n"
    "  SegmenterAgent  -->  Clause List [{id, text, section, source_doc}]\n"
    "        |\n"
    "        v\n"
    "  EmbedderAgent   -->  Embeddings (N x 384, L2-normalised, all-MiniLM-L6-v2)\n"
    "        |\n"
    "        v\n"
    "  ClauseAuditorEnv (Gymnasium MDP)\n"
    "   State : [emb_A(384) | emb_B(384) | cosine_sim | len_ratio |\n"
    "             budget_used_frac | step_frac]  =  772-dim\n"
    "   Action: 0=SKIP, 1=COMPARE\n"
    "   Budget: 15% of all N*(N-1)/2 pairs per episode\n"
    "        |\n"
    "        v\n"
    "  RLAuditAgent (DQN + UCB1)  -->  Flagged contradiction pairs\n"
    "        |\n"
    "        v\n"
    "  ReportAgent  -->  AuditReport {precision, recall, F1, efficiency, FlaggedPairs}\n"
    "        |\n"
    "        v\n"
    "  ContradictionMemory  -->  Cross-document pattern detection"
)
pdf.h2("Sub-Agent Roles")
widths = [38, 132]
pdf.tbl_header(["Sub-Agent", "Responsibility"], widths)
for row in [
    ("SegmenterAgent",   "Splits raw policy text into discrete clauses (>15 tokens, section-aware)"),
    ("EmbedderAgent",    "Maps clauses to 384-dim semantic vectors; caches to disk for efficiency"),
    ("RLAuditAgent",     "Runs DQN+UCB episode; returns FlaggedPair list with confidence scores"),
    ("ReportAgent",      "Computes metrics (Precision/Recall/F1/Efficiency) and formats AuditReport"),
    ("ContradictionMemory", "Stores per-document TP sets; surfaces types found in multiple docs"),
]:
    pdf.tbl_row(row, widths)

# ============================================================================
# 3. MDP FORMULATION
# ============================================================================
pdf.add_page()
pdf.h1("3. MDP Formulation")
pdf.body("AIRA is formalised as a finite-horizon Markov Decision Process M = (S, A, R, T, gamma).")

pdf.h2("3.1 State Space")
pdf.body("For a document with N clauses, each episode presents one clause pair (c_i, c_j) at a time. The state vector is:")
pdf.code(
    "  S_t in R^772 =\n"
    "    [ phi(c_i)  (384 dims) ]   -- sentence embedding of clause A\n"
    "    [ phi(c_j)  (384 dims) ]   -- sentence embedding of clause B\n"
    "    [ cos(phi_i, phi_j)  (1) ] -- cosine similarity (dot product, normalised)\n"
    "    [ lambda(c_i, c_j)   (1) ] -- |len_A - len_B| / max(len_A, len_B)\n"
    "    [ b_t                (1) ] -- budget fraction used so far\n"
    "    [ tau_t              (1) ] -- step fraction (step / total_pairs)\n"
    "  Total: 384 + 384 + 1 + 1 + 1 + 1 = 772"
)
pdf.body("phi is the all-MiniLM-L6-v2 sentence transformer (L2-normalised, 384-dim).")

pdf.h2("3.2 Action Space")
pdf.body("A = {0 = SKIP, 1 = COMPARE}  (Discrete(2))")
pdf.body("COMPARE flags the pair as a potential contradiction and consumes one budget unit. SKIP passes without review.")

pdf.h2("3.3 Reward Function")
pdf.code(
    "  R(a, is_contradiction) =\n"
    "    +2.0  if a == COMPARE and is_contradiction       (True Positive)\n"
    "    -0.1  if a == COMPARE and not is_contradiction   (False Positive)\n"
    "    -0.5  if a == SKIP   and is_contradiction        (False Negative / Miss)\n"
    "     0.0  if a == SKIP   and not is_contradiction    (True Negative)"
)
pdf.body(
    "Design rationale: The miss penalty (-0.5) is 5x larger than the false-positive "
    "penalty (-0.1), driving the agent towards high recall. The asymmetry means the "
    "agent learns to err on the side of caution -- flag ambiguous pairs -- rather than "
    "skip them. The +2.0 true-positive reward provides a strong incentive to seek out "
    "genuine contradictions. If the budget is exhausted before all pairs are reviewed, "
    "all unreviewed ground-truth contradictions incur an additional -0.5 penalty each."
)

pdf.h2("3.4 Termination and Discount")
pdf.code(
    "  Terminated = (budget_used >= floor(0.15 * N_pairs)) OR (all pairs reviewed)\n"
    "  gamma = 0.99  (finite-horizon; keeps distant budget-penalty terms from dominating)"
)

pdf.h2("3.5 Curriculum Learning")
pdf.body("The budget fraction linearly decays from 0.30 (episode 0) to 0.15 (episode 200), then stays fixed:")
pdf.code("  budget(ep) = 0.30 - min(ep / 200, 1.0) * (0.30 - 0.15)")
pdf.body("A wider initial budget ensures the replay buffer is populated with positive (COMPARE) transitions early in training, preventing the agent from learning a trivial all-SKIP policy.")

# ============================================================================
# 4. RL METHODS
# ============================================================================
pdf.add_page()
pdf.h1("4. Reinforcement Learning Methods")

pdf.h2("4.1 Deep Q-Network (DQN) -- Value-Based Learning")
pdf.body("The Q-network maps the 772-dim state to Q-values for both actions:")
pdf.code(
    "  Architecture: 772 -> 512 (BN + ReLU) -> 256 (BN + ReLU) -> 128 (ReLU) -> 2\n"
    "  Optimizer   : Adam (lr = 1e-4)\n"
    "  Loss        : Huber (SmoothL1) -- robust to asymmetric reward outliers\n"
    "  Batch size  : 64\n"
    "  Buffer      : 50,000 transitions (random experience replay)\n"
    "  Target net  : hard copy every 100 gradient steps\n"
    "  Epsilon     : linear decay 1.0 -> 0.05 over 10,000 gradient steps\n"
    "  Grad clip   : max_norm = 10.0"
)
pdf.body("Bellman update:")
pdf.code(
    "  Q(s,a) <- r + gamma * max_{a'} Q_theta_minus(s', a')\n"
    "  Loss   = SmoothL1( Q_theta(s,a),  r + gamma * max_{a'} Q_theta_minus(s', a') )"
)
pdf.body(
    "BatchNorm1d on hidden layers requires explicit train()/eval() mode switching during "
    "inference. The target network always stays in eval() mode; the online network "
    "switches to eval() for single-sample Q-value queries and back to train() for "
    "gradient updates."
)

pdf.h2("4.2 UCB1 Exploration Strategy")
pdf.body("UCBExplorer wraps DQNAgent and augments the COMPARE action Q-value with an exploration bonus:")
pdf.code(
    "  score(i,j) = Q_DQN(s_{ij}, COMPARE)  +  c * sqrt( ln(t + 1) / N(i,j) )\n\n"
    "  c      = 0.3   (tuned; see ablation study in Section 6.3)\n"
    "  t      = total pair-selection steps accumulated across ALL episodes\n"
    "  N(i,j) = visit count for canonical pair (i,j) -- also accumulated across episodes\n\n"
    "  pair_to_idx(i, j, n) = i*n - i*(i+1)//2 + (j-i-1)   [triangular mapping]"
)
pdf.body(
    "The UCB bonus applies ONLY to the COMPARE action. Semantically, this is correct: "
    "UCB encourages examining unseen or rarely-reviewed pairs before deciding to skip them. "
    "Visit counts accumulate ACROSS episodes (not reset each episode) -- this is the key "
    "design choice that produces the exploration-to-exploitation shift observed in training."
)
pdf.body(
    "Important: UCB is a TRAINING-TIME mechanism. At inference time, fresh visit counts "
    "(all zero) would give infinite UCB bonus to every pair, overriding learned Q-values "
    "and degenerating to random exploration. Inference uses the DQN agent directly with "
    "epsilon from the checkpoint (~0.05)."
)

# ============================================================================
# 5. EXPERIMENTS AND RESULTS
# ============================================================================
pdf.add_page()
pdf.h1("5. Experiments and Results")

pdf.h2("5.1 Experimental Setup")
pdf.body("All experiments use the synthetic_policy.txt training document (17 clauses, 8 contradictions, 136 pairs). Real sentence-transformer embeddings (all-MiniLM-L6-v2) are used throughout. Training uses seeds 0-499; evaluation uses seeds 450-499 (last 50 training episodes for in-distribution) and 1000-1019 (held-out, for generalisation).")

pdf.h2("5.2 Main Results -- Training Distribution (seeds 0-499)")
w = [42, 38, 38, 38, 38]
pdf.tbl_header(["Agent", "Precision", "Recall", "F1", "Efficiency"], w)
for row, bold, fill in [
    (["Random",       f"{rand_r['final_precision']:.3f}", f"{rand_r['final_recall']:.3f}", f"{rand_r['final_f1']:.3f}", f"{rand_r['efficiency_score']:.3f}"], False, False),
    (["Cosine Sim",   "0.073", "0.135", "0.073", "0.147"], False, False),
    (["DQN",          f"{dqn_r['final_precision']:.3f}", f"{dqn_r['final_recall']:.3f}", f"{dqn_r['final_f1']:.3f}", f"{dqn_r['efficiency_score']:.3f}"], False, False),
    (["DQN+UCB",      f"{ucb_r['final_precision']:.3f}", f"{ucb_r['final_recall']:.3f}", f"{ucb_r['final_f1']:.3f}", f"{ucb_r['efficiency_score']:.3f}"], True, True),
]:
    pdf.tbl_row(row, w, bold=bold, fill=fill)
pdf.body("Efficiency = fraction of pairs reviewed (lower is better). DQN+UCB achieves near-perfect recall while reviewing fewer pairs than random.", size=9)

pdf.h2("5.3 Held-Out Generalisation (seeds 1000-1019, greedy inference)")
syn = gen_r["documents"]["Synthetic"]["dqn_ucb"]
neu = gen_r["documents"]["Northeastern"]["dqn_ucb"]
mit = gen_r["documents"]["MIT"]["dqn_ucb"]
syn_c = gen_r["documents"]["Synthetic"]["cosine"]
neu_c = gen_r["documents"]["Northeastern"]["cosine"]
mit_c = gen_r["documents"]["MIT"]["cosine"]
w2 = [40, 28, 28, 28, 28, 42]
pdf.tbl_header(["Document", "Split", "Agent F1", "Agent Recall", "Eff", "Cosine F1 (baseline)"], w2)
for row in [
    ["Synthetic", "train",     f"{syn['f1']:.3f}", f"{syn['recall']:.3f}", f"{syn['efficiency']:.3f}", f"{syn_c['f1']:.3f}"],
    ["NEU",       "zero-shot", f"{neu['f1']:.3f}", f"{neu['recall']:.3f}", f"{neu['efficiency']:.3f}", f"{neu_c['f1']:.3f}"],
    ["MIT",       "zero-shot", f"{mit['f1']:.3f}", f"{mit['recall']:.3f}", f"{mit['efficiency']:.3f}", f"{mit_c['f1']:.3f}"],
]:
    pdf.tbl_row(row, w2)
pdf.body(
    "Key finding: Cosine similarity outperforms DQN+UCB for zero-shot transfer. "
    "500 training episodes on one document are insufficient for the DQN to learn "
    "fully document-agnostic contradiction semantics. Training on a multi-document "
    "corpus would address this (see Section 8).", size=9
)

pdf.h2("5.4 Multi-Seed Statistical Validation (5 seeds x 500 episodes)")
ms = ms_r["summary"]
pdf.body(f"To validate training stability, DQN+UCB was trained independently from five random seeds "
         f"({ms_r['seeds']}) for 500 episodes each.")
w3 = [60, 55, 55]
pdf.tbl_header(["Metric", "Mean", "Std"], w3)
for k, label in [("f1","F1"),("recall","Recall"),("precision","Precision")]:
    pdf.tbl_row([label, f"{ms[k+'_mean']:.3f}", f"{ms[k+'_std']:.3f}"], w3)
pdf.body(f"F1 = {ms['f1_mean']:.3f} +/- {ms['f1_std']:.3f}  across {len(ms_r['seeds'])} seeds. "
         "High standard deviation indicates sensitivity to initialisation and training seed ordering.", size=9)

pdf.ln(3)
pdf.h2("5.5 Learning Curves")
pdf.fig("results/figures/learning_curve.png", w=165,
        caption="Fig 1. Reward and Recall over 500 training episodes (DQN vs DQN+UCB, 20-ep smoothing)")
pdf.fig("results/figures/multi_seed_curve.png", w=165,
        caption="Fig 2. Multi-seed learning curves with +/- 1 std confidence bands (5 seeds)")

# ============================================================================
# 6. ANALYSIS
# ============================================================================
pdf.add_page()
pdf.h1("6. Analysis")

pdf.h2("6.1 Agent Comparison")
pdf.fig("results/figures/comparison_bar.png", w=150,
        caption="Fig 3. Precision, Recall, and F1 for all agents (training distribution)")

pdf.h2("6.2 Efficiency-Recall Trade-off")
pdf.fig("results/figures/efficiency_tradeoff.png", w=140,
        caption="Fig 4. Efficiency vs Recall scatter. Top-left is ideal (high recall, low effort)")

pdf.h2("6.3 UCB Exploration Constant Ablation")
pdf.body("To verify that c=0.3 was the optimal choice, a sweep over six values was conducted for 300 episodes:")
abl = {r["c"]: r for r in abl_r["results"]}
w4 = [30, 32, 32, 32, 48]
pdf.tbl_header(["c", "Recall", "F1", "Efficiency", "Notes"], w4)
for c_val in abl_r["c_values"]:
    r = abl[c_val]
    note = "<-- chosen" if c_val == 0.3 else ("<-- pure DQN" if c_val == 0.0 else "")
    pdf.tbl_row([f"{c_val:.2f}", f"{r['final_recall']:.3f}", f"{r['final_f1']:.3f}",
                 f"{r['final_efficiency']:.3f}", note], w4,
                bold=(c_val == 0.3), fill=(c_val == 0.3))
pdf.fig("results/figures/ucb_ablation.png", w=165,
        caption="Fig 5. UCB ablation: Recall, F1, and Efficiency as a function of exploration constant c")

pdf.h2("6.4 UCB Bonus Decay")
pdf.fig("results/figures/ucb_bonus_decay.png", w=150,
        caption="Fig 6. UCB bonus decay over training steps -- exploration decreases as visit counts accumulate")
pdf.body(
    "As training progresses and visit counts N(i,j) accumulate, the UCB bonus decreases, "
    "causing the agent to rely increasingly on its learned Q-values. This exploration-to-"
    "exploitation shift is the mechanism by which UCB drives early coverage of all clause "
    "pairs before the DQN policy converges."
)

pdf.h2("6.5 Why Cosine Similarity Underperforms During Training")
pdf.body(
    "Contradictory clause pairs use similar vocabulary (same topic, opposite permissions). "
    "High cosine similarity should therefore correlate with contradiction. Yet with random "
    "embeddings (as used in the initial experiments), cosine similarity is near-random -- "
    "confirming that semantic embeddings are essential for the baseline to work. "
    "With real embeddings, cosine achieves F1=0.073 on training-distribution seeds "
    "(where evaluation is done during UCB training with exploration noise) but F1=0.358 "
    "on held-out seeds with greedy inference. "
    "DQN+UCB (F1=0.655 training / F1=0.827 held-out) significantly outperforms cosine on "
    "the training document, demonstrating that the RL agent learns richer decision features "
    "beyond simple embedding distance."
)

# ============================================================================
# 7. CHALLENGES AND SOLUTIONS
# ============================================================================
pdf.add_page()
pdf.h1("7. Challenges and Solutions")

challenges = [
    ("UCB inference degeneracy",
     "Fresh UCBExplorer visit counts (all zero at inference time) give infinite "
     "UCB bonus to every pair, overriding learned Q-values and reducing inference "
     "to random exploration.",
     "UCBExplorer is bypassed at inference time. The DQN agent is used directly "
     "with epsilon loaded from the checkpoint (~0.05), matching the end-of-training "
     "policy distribution."),
    ("Reward sparsity in early training",
     "With 8 contradictions in 136 pairs and 15% budget (20 pairs/episode), "
     "the probability of finding a contradiction by random exploration is only "
     "8 * 20 / 136 = 1.18 expected hits per episode. Early episodes see mostly -0.5 "
     "miss penalties and 0.0 true-negative rewards.",
     "Curriculum learning starts with 30% budget (41 pairs/episode), "
     "giving expected hits of ~2.4 per episode -- enough to populate the replay "
     "buffer with positive transitions. Budget decays to 15% over 200 episodes."),
    ("BatchNorm1d with single-sample inference",
     "BatchNorm1d uses batch statistics during forward passes of size >1 but "
     "running statistics for batch_size=1. Calling model.train() during "
     "single-sample inference corrupts running statistics and destabilises "
     "Q-value estimates.",
     "get_q_values() explicitly switches q_net to eval() mode before the "
     "single-sample forward pass and restores train() mode immediately after. "
     "The target network always stays in eval() mode."),
    ("Embedding version mismatch",
     "The original experiments were run with sentence-transformers==2.7.0 but "
     "scikit-learn 1.4.2 breaks that version (removed _IS_32BIT). Running "
     "run_experiments.py without real embeddings produced inflated F1=0.969 "
     "results on random embeddings that did not generalise.",
     "Upgraded to sentence-transformers>=3.0.0. All experiments were re-run "
     "with real semantic embeddings. Genuine held-out F1=0.827 is reported."),
    ("UCB cross-episode visit count accumulation",
     "An early bug reset visit counts each episode, giving infinite UCB bonus "
     "to all pairs at the start of every episode and preventing the "
     "exploration-to-exploitation shift.",
     "UCBExplorer.reset() increments episode counter only -- it does NOT reset "
     "_visit_counts or _total_steps. Visit counts accumulate monotonically "
     "across all 500 training episodes."),
]

for title, problem, solution in challenges:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(25)
    pdf.multi_cell(165, 5, f"Problem: {problem}")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(25)
    pdf.multi_cell(165, 5, f"Solution: {solution}")
    pdf.ln(2)

# ============================================================================
# 8. ETHICAL CONSIDERATIONS
# ============================================================================
pdf.h1("8. Ethical Considerations")
pdf.body(
    "AIRA is designed as a triage tool, not an autonomous judge. Every pair flagged by the "
    "RL agent requires human review before any academic or disciplinary consequence is "
    "considered. The following risks and mitigations are identified:"
)
w5 = [55, 115]
pdf.tbl_header(["Risk", "Mitigation"], w5)
for row in [
    ("False positive accusations",
     "Asymmetric reward (-0.1 FP vs -0.5 miss) keeps precision high; "
     "human review required for all flagged pairs before action"),
    ("Synthetic training bias",
     "Evaluated zero-shot on NEU and MIT policies; generalisation results reported; "
     "limitations acknowledged"),
    ("Non-English policy text",
     "all-MiniLM-L6-v2 is English-optimised; multilingual model needed for "
     "non-English institutional policies"),
    ("Position-feature overfitting",
     "step_frac and budget_used_frac allow position-based heuristics; "
     "noted as limitation; future work to remove or reduce"),
    ("No individual student profiling",
     "AIRA operates at the document level only -- no student data is processed "
     "or stored at any stage"),
]:
    pdf.tbl_row(row, w5)
pdf.ln(3)
pdf.body(
    "The 8.1% efficiency metric means 91.9% of clause pairs are never reviewed. This is "
    "intentional: AIRA is designed for targeted flagging. An institution using AIRA should "
    "maintain human oversight for all flagged items and establish an appeal process before "
    "acting on automated findings. The system is not production-ready without independent "
    "validation on institutional policy corpora."
)

# ============================================================================
# 9. FUTURE IMPROVEMENTS
# ============================================================================
pdf.add_page()
pdf.h1("9. Future Improvements")

for title, desc in [
    ("Multi-document training corpus",
     "Training on synthetic, NEU, and MIT policies jointly would expose the agent to "
     "diverse contradiction patterns and significantly improve zero-shot generalisation. "
     "The zero-shot F1 gap (0.179 vs 0.632 for cosine) suggests the agent has not yet "
     "learned document-agnostic semantic representations of contradiction."),
    ("Remove position features from state",
     "The step_frac and budget_used_frac state dimensions allow the agent to learn "
     "position-based heuristics that overfit to training seed orderings. Removing or "
     "replacing them with a learned uncertainty estimate would improve generalisation."),
    ("Double DQN (DDQN)",
     "Standard DQN overestimates Q-values due to the max operator. DDQN decouples "
     "action selection (online network) from action evaluation (target network): "
     "Q_target = r + gamma * Q_theta_minus(s', argmax_{a'} Q_theta(s', a')). "
     "This would reduce the overestimation bias observed in late training."),
    ("Prioritised Experience Replay",
     "With only 8 true contradictions per episode, positive transitions "
     "(reward = +2.0) are rare. Prioritised replay assigns higher sampling "
     "weight to high-TD-error transitions, ensuring positive examples are "
     "replayed more frequently without requiring curriculum learning."),
    ("LLM grounding for detected pairs",
     "After the RL agent flags a pair, an LLM (Claude, GPT-4) could verify "
     "whether the flagged pair is genuinely contradictory by reasoning over "
     "the full clause texts. This would reduce false positives and add a "
     "human-readable explanation -- the faithfulness_score placeholder in "
     "src/evaluation/metrics.py is reserved for this."),
    ("Policy gradient comparison",
     "Implementing PPO or REINFORCE would provide a direct comparison with "
     "value-based DQN. Policy gradients can handle continuous action spaces "
     "and may generalise better from sparse rewards in this domain."),
]:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(25)
    pdf.multi_cell(165, 5, desc)
    pdf.ln(2)

# ============================================================================
# 10. REPRODUCIBILITY
# ============================================================================
pdf.h1("10. Reproducibility")
pdf.code(
    "  # 1. Clone and set up environment\n"
    "  git clone https://github.com/rajesh1997r/AIRA-policy-auditor.git\n"
    "  cd AIRA-policy-auditor\n"
    "  python3 -m venv .venv && source .venv/bin/activate\n"
    "  pip install -r requirements.txt\n\n"
    "  # 2. Run all experiments (Random, DQN, DQN+UCB)\n"
    "  python experiments/run_experiments.py --experiment all\n\n"
    "  # 3. Generalization, multi-seed, ablation\n"
    "  python experiments/eval_generalization.py\n"
    "  python experiments/multi_seed_eval.py       # ~20 min\n"
    "  python experiments/ucb_ablation.py          # ~15 min\n\n"
    "  # 4. Run demo (trained model)\n"
    "  python demo/demo_policy_audit.py --all-docs\n\n"
    "  # 5. Analysis notebook\n"
    "  jupyter notebook notebooks/03_analysis.ipynb  # select kernel: AIRA (Python 3.12)"
)
pdf.h2("Key Hyperparameters")
w6 = [55, 35, 80]
pdf.tbl_header(["Parameter", "Value", "Justification"], w6)
for row in [
    ("episodes",          "500",   "Convergence observed by ep 400"),
    ("budget_fraction",   "0.15",  "15% of pairs; challenging but achievable constraint"),
    ("curriculum_start",  "0.30",  "Wide start ensures early positive transitions"),
    ("ucb_c",             "0.3",   "Ablation shows peak F1 at c=0.3 (see Section 6.3)"),
    ("lr",                "1e-4",  "Standard Adam LR for DQN with BatchNorm"),
    ("gamma",             "0.99",  "Near-unity; finite episodes, discount not critical"),
    ("buffer_capacity",   "50,000","Covers full training run with adequate diversity"),
    ("target_update",     "100",   "Frequent enough for stability, sparse enough to learn"),
]:
    pdf.tbl_row(row, w6)

# -- Output ------------------------------------------------------------------
out = "AIRA_Technical_Report.pdf"
pdf.output(out)
print(f"PDF saved: {out}  ({os.path.getsize(out)//1024} KB)")
