"""ODE's Phase 0 values in one place, each with where it comes from (docs/DECISIONS.md, 2026-11-26).

Sources were read on 2026-11-26 (arXiv abstract + HTML full text). "Ours" marks a value that
is our own engineering choice, not taken from a paper; say so when presenting it.
"""

# ---- learning (tokencut/ode/bandit.py) ----
# Ridge prior 1 (A starts as the identity): Li et al. 2010, Algorithms 1-2 ("A_a <- I_d");
# ACRouter / CodeRouterBench (Zhou et al. 2026, arXiv 2606.22902) run LinUCB with lambda = 1.
RIDGE = 1.0
# Exploration weight 1: ACRouter LinUCB "alpha = lambda = 1"; OrcaRouter (Bao et al. 2026,
# arXiv 2605.30736) "alpha = 1.0". Li et al. 2010: alpha needs tuning (CTR is an inverted U in
# alpha) -> tune later on replay data only, never on the held-out tasks.
ALPHA = 1.0
# Switch margin: OrcaRouter keeps the arm with the better training mean when the top two score
# within 0.02 (pick-flip rate 16.7% -> 2.4%, about -0.86 Arena points). Adapted here to keep the
# current model when another arm wins by <= 0.02 (fewer model switches, i.e. fewer cache rebuilds).
SWITCH_MARGIN = 0.02
# Random tries with the probability logged: Bietti, Agarwal, Langford 2021 ("A Contextual
# Bandit Bake-off", JMLR) - IPS evaluation needs a minimum exploration probability > 0;
# Li et al. 2011 replay evaluation needs randomized logging. 5% = ours, inside the epsilon range
# of the bake-off's grid (to be checked in its appendix before quoting the exact grid).
EXPLORE = 0.05

# ---- reward (tokencut/ode/reward.py) ----
# r = solved - LAMBDA * min(1, cost / C_ref). Shape and weight from OrcaRouter Eq. 3-4:
# reward = 1.0 x quality - 0.4 x cost, with cost "clipped to [0,1]" against a per-deployment
# budget scale (latency weight 0 unless measured, as here). RouterBench (Hu et al. 2024) uses
# the same trade-off as lambda x quality - cost ("willingness to pay") and reports several
# lambdas -> we also report a lambda sweep. ACRouter: score - 0.1 x cost in raw USD.
LAMBDA = 0.4
# C_ref = the budget scale: 95th percentile of Opus per-turn cost on the practice set (ours).
C_REF_PERCENTILE = 95

# ---- when ODE decides (tokencut/ode/features.py, gateway/ode.py) ----
# Per task, never per step: TwinRouterBench (Yang et al. 2026, arXiv 2605.18859) found every
# router lost money on its SWE-bench slice because "one under-routed step fails the whole
# trajectory"; SWE-Router (Son et al. 2026, arXiv 2607.00053) routes per task.
# Threshold 0 (ours, from our pilot): any typed message the router does not call `no_brief`
# starts a task. At 0.87 (the task-brief threshold) only 15 of 48 pilot turns qualified.
# Router confidence still reaches the bandit through the probabilities in x.
DECISION_THRESHOLD = 0.0

# ---- context (tokencut/ode/features.py) ----
# 13 numbers, 20 shared: small on purpose for ~hundreds of rows. Li et al. 2010 compressed users
# and articles to 6 features each (36 shared); ParetoBandit (Taberner-Miller 2026,
# arXiv 2604.00136) uses 25 PCA components + bias (d = 26).

# ---- safety (ours, engineering) ----
WINDOW_MARGIN = 0.20  # keep 20% of a model's context window free
