# OPE reliability under three logging regimes -- ch13 field-deployments capstone.
# Question: when can offline off-policy evaluation and selection pick the right
# deployment policy? Three logging regimes on one hidden pricing MDP: a randomized
# A/B log (uniform, full action support), a narrow-support incumbent log
# (epsilon-greedy around the exact myopic rule, propensities logged), and a
# historical-mixture log (each episode served by one of 7 past pricing policies, so
# the reference-discount axis is swept). The hypothesis under test: direct-method
# reliability is governed by STATE-OCCUPANCY coverage, which policy-diverse log
# history supplies and any single policy's log does not. Both outcomes are
# reportable; the diagnostics gates (FQE calibration, truth cross-check, clone
# fidelity, support) decide whether the numbers are trustworthy at all.
# The DGP, policy heads, and OPE/OPS wiring live in promo_env.py and pipeline.py;
# this script is the experiment driver + outputs.
import os
import sys
import json
import time
import hashlib
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
for _n in ("d3rlpy", "scope_rl", "SCOPE-RL"):
    logging.getLogger(_n).setLevel(logging.ERROR)
# d3rlpy logs via structlog (its info/debug lines bypass stdlib levels); filter them out.
try:
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL)
    )
except Exception:
    pass
# d3rlpy's internal FQE fit renders a tqdm bar and passes disable=False explicitly, so a
# default-only patch is overridden. Force disable=True on the base tqdm __init__ regardless
# of the caller's argument (must run before d3rlpy is imported below).
try:
    import tqdm.std as _tqdm_std

    _tqdm_orig_init = _tqdm_std.tqdm.__init__

    def _tqdm_quiet_init(self, *a, **k):
        k["disable"] = True
        return _tqdm_orig_init(self, *a, **k)

    _tqdm_std.tqdm.__init__ = _tqdm_quiet_init
except Exception:
    pass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE, BENCH_STYLE

apply_style()

# pipeline.py / promo_env.py live in this dir (added to sys.path by the two lines above
# only reaching repo root; add this dir too).
sys.path.insert(0, os.path.dirname(__file__))
from promo_env import (
    PromoConfig,
    PromoEnv,
    DISCOUNTS,
    uniform_batch,
    constant_batch,
    myopic_batch,
    softened_batch,
)
from pipeline import (
    generate_log,
    train_candidates,
    true_values,
    build_ope_inputs,
    evaluate_ope,
    ops_metrics,
)
import ope_diagnostics as diag
from ope_diagnostics import (
    FQE_EPOCH_LEN,
    fqe_expected_epochs,
    effective_sample_size,
    coverage_stats,
)

# --- configuration -----------------------------------------------------------
SCRIPT_NAME = "field_ope_reliability"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
OUT_DIR = os.path.dirname(__file__)

N_SEEDS = 10
REGIMES = ["ab", "incumbent", "mixture"]
REGIME_SEEDS = {r: list(range(N_SEEDS)) for r in REGIMES}
REGIME_LABEL = {
    "ab": "A/B (uniform)",
    "incumbent": "Incumbent (narrow)",
    "mixture": "Historical mixture",
}
ESTIMATORS = ["dm", "pdis", "dr"]  # scope-rl estimator_name keys
ESTIMATOR_LABEL = {"dm": "DM", "pdis": "PDIS", "dr": "DR"}
CAND_ORDER = ["no_promo", "disc5", "disc10", "myopic", "uniform", "cql"]
CAND_LABEL = {
    "no_promo": "No promo",
    "disc5": r"5\% discount",
    "disc10": r"10\% discount",
    "myopic": "Myopic incumbent",
    "uniform": "Uniform",
    "cql": "Conservative Q-learning (CQL; offline RL)",
}
# candidates that coincide with a mixture component (their aligned stratum gives IS
# a real effective sample size under the mixture log)
ALIGNED_CANDS = ["no_promo", "disc5", "disc10", "myopic"]
# calibration candidate per regime: the candidate whose occupancy the log covers by
# construction, so any DM error there is pure fit error. Under 'ab' the uniform
# candidate IS the behavior policy (fatal gate); under 'incumbent' the myopic
# candidate is the behavior policy up to clone fidelity (fatal gate); under
# 'mixture' the uniform candidate is merely on-support, so the check warns only.
CAL_CAND = {"ab": "uniform", "incumbent": "myopic", "mixture": "uniform"}
CAL_FATAL = {"ab": True, "incumbent": True, "mixture": False}

# Training budgets (one place; hashed into every component so a change re-runs cleanly).
# fqe_steps MUST be a multiple of FQE_EPOCH_LEN (10000): d3rlpy runs n_steps//n_steps_per_epoch
# epochs with n_steps_per_epoch defaulting to 10000, so anything below trains the FQE for
# ZERO gradient steps. This is the bug that voided the first run; the guard below refuses it.
RUN_CFG = {
    "n_traj": 500,
    "bc_steps": 8000,
    "bc_myopic": "1500ep_5x",  # myopic clone: 1500 episodes, 5x bc_steps (fidelity probe)
    "menu_seed": "fixed_v2",  # candidates seeded independently of log-generation RNG use
    "cql_steps": 2000,
    # pilot 2026-07-19: 20k calibrates but DM on the off-occupancy candidate still
    # moves 0.42 to 50k; 50k->100k movement <= 0.053 (refit noise) with errors
    # unchanged, so 50k is the converged budget and the residual A/B bias is
    # coverage, not undertraining.
    "fqe_steps": 50000,
    "fqe_hidden": (256, 256),
    "n_on_policy": 500,
    "true_n_ep": 8000,
    "cand_epsilon": 0.3,  # soften candidates so IS has finite overlap
    "incumbent_epsilon": 0.3,  # aligned with cand_epsilon: myopic candidate == behavior
    "mix_epsilon": 0.3,  # per-component softening in the mixture log
    "cql_on": "ab_log",  # CQL trains on the fixed A/B log of the seed, every regime
    "truth_convention": "gamma0_v2",  # scope-rl on-policy values corrected to gamma^0
    "gamma": PromoConfig().gamma,
    "K": PromoConfig().K,
    "horizon": PromoConfig().horizon,
}

if RUN_CFG["fqe_steps"] % FQE_EPOCH_LEN != 0 or RUN_CFG["fqe_steps"] < FQE_EPOCH_LEN:
    raise ValueError(
        f"fqe_steps={RUN_CFG['fqe_steps']} would train FQE for "
        f"{fqe_expected_epochs(RUN_CFG['fqe_steps'])} epochs; must be a multiple of "
        f"{FQE_EPOCH_LEN} and >= {FQE_EPOCH_LEN}."
    )


def _component_config(regime, seed):
    return {**RUN_CFG, "regime": regime, "seed": int(seed)}


# --- candidate-identity probe -------------------------------------------------
def _probe_states(cfg):
    """A fixed batch of states (contexts x reference-discount grid) for hashing each
    candidate's greedy actions, so candidate identity can be compared across regimes
    of the same seed (the menu must be identical for an apples-to-apples table)."""
    rng = np.random.default_rng(4242)
    n = 512
    C = np.clip(rng.normal(size=(n, cfg.K)), -cfg.ctx_clip, cfg.ctx_clip)
    r = np.linspace(0.0, DISCOUNTS[-1], n)
    t = np.full(n, 0.5)
    return np.concatenate([C, r[:, None], t[:, None]], axis=1).astype(np.float32)


def _candidate_hashes(cfg, cands):
    probe = _probe_states(cfg)
    out = {}
    for nm, head in cands.items():
        greedy = np.asarray(head.base_policy.predict(probe)).astype(np.int64)
        out[nm] = hashlib.md5(greedy.tobytes()).hexdigest()[:12]
    return out


# --- one (regime, seed) cell -------------------------------------------------
def run_one(regime, seed):
    """Full OPE->OPS pass for one logging regime and seed. Returns plain-python data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    cfg = PromoConfig()
    t0 = time.time()
    # the A/B log of this seed is always built: CQL trains on it in every regime so
    # the candidate menu is identical across regimes (cql_on = 'ab_log').
    ab_logged, _ = generate_log(cfg, "ab", n_traj=RUN_CFG["n_traj"], seed=seed)
    if regime == "ab":
        logged, head = ab_logged, None
    else:
        logged, head = generate_log(
            cfg,
            regime,
            n_traj=RUN_CFG["n_traj"],
            seed=seed,
            incumbent_epsilon=RUN_CFG["incumbent_epsilon"],
            mix_epsilon=RUN_CFG["mix_epsilon"],
        )
    # reseed RIGHT BEFORE candidate training: regimes consume different amounts of
    # global RNG while generating logs, and without this the myopic-clone/CQL nets
    # get different inits across regimes of the same seed (caught by CAND_DRIFT)
    torch.manual_seed(10_000 + seed)
    np.random.seed(10_000 + seed)
    cands, fidelity = train_candidates(
        cfg,
        ab_logged,
        seed,
        bc_steps=RUN_CFG["bc_steps"],
        cql_steps=RUN_CFG["cql_steps"],
        cand_epsilon=RUN_CFG["cand_epsilon"],
    )
    tv = true_values(cfg, cands, n_ep=RUN_CFG["true_n_ep"], seed=5000 + seed)
    input_dict = build_ope_inputs(
        cfg,
        logged,
        cands,
        seed,
        fqe_steps=RUN_CFG["fqe_steps"],
        n_on_policy=RUN_CFG["n_on_policy"],
        fqe_hidden=RUN_CFG["fqe_hidden"],
    )
    ope = evaluate_ope(logged, input_dict)
    ops = ops_metrics(ope, input_dict)  # {est: {regret, rank_corr, rankings}}
    epv = ope.estimate_policy_value(input_dict)  # {policy: {dm,pdis,dr,on_policy}}
    # coerce to plain floats for a small, portable cache payload
    epv_plain = {pol: {k: float(v) for k, v in d.items()} for pol, d in epv.items()}
    tv_plain = {nm: [float(m), float(se)] for nm, (m, se) in tv.items()}
    diagnostics = _cell_diagnostics(logged, input_dict, list(cands.keys()))
    diagnostics["fidelity"] = {nm: float(v) for nm, v in fidelity.items()}
    diagnostics["cand_hash"] = _candidate_hashes(cfg, cands)
    diagnostics["wallclock_s"] = float(time.time() - t0)
    if regime == "mixture" and head is not None:
        counts = np.bincount(
            np.asarray(head.component_trace), minlength=head.n_components
        )
        diagnostics["mixture_component_counts"] = counts.tolist()
        diagnostics["pscore_min"] = float(np.min(np.asarray(logged["pscore"])))
        diagnostics["mix_epsilon"] = RUN_CFG["mix_epsilon"]
        diagnostics["n_components"] = int(head.n_components)
    return {"ops": ops, "epv": epv_plain, "true_mc": tv_plain, "diag": diagnostics}


def _cell_diagnostics(logged, input_dict, cand_names):
    """Per-cell raw diagnostics (cached; alerts are computed from these at output time).
    Records action coverage of the log and, per candidate, trajectory-IS effective sample
    size, surviving-trajectory count, and max weight -- the tell for the curse of horizon."""
    n_actions = int(logged["n_actions"])
    n_traj = int(logged["n_trajectories"])
    horizon = int(logged["step_per_trajectory"])
    la = np.asarray(logged["action"]).astype(int)
    pscore = np.asarray(logged["pscore"], dtype=float)
    idx = np.arange(la.shape[0])

    cov = coverage_stats(la, n_actions)
    # sanity: the flat log must be trajectory-major fixed-length so the reshape aligns
    done = np.asarray(logged["terminal"], dtype=float).reshape(n_traj, horizon)
    cov["traj_major_ok"] = bool(np.all(done[:, -1] == 1.0))

    per_cand = {}
    for nm in cand_names:
        ead = np.asarray(input_dict[nm]["evaluation_policy_action_dist"])  # (size, N)
        pi_e = ead[idx, la]  # pi_e(logged action | state)
        ratio = pi_e / pscore
        traj_w = ratio.reshape(n_traj, horizon).prod(
            axis=1
        )  # cumulative per trajectory
        per_cand[nm] = {
            "ess": effective_sample_size(traj_w),
            "surviving": int((traj_w > 0).sum()),
            "max_weight": float(traj_w.max()),
            "n_traj": n_traj,
        }
    return {
        "coverage": cov,
        "per_cand": per_cand,
        "fqe_expected_epochs": fqe_expected_epochs(RUN_CFG["fqe_steps"]),
    }


# --- reference-state coverage (the mechanism panel) ---------------------------
def compute_reference_coverage():
    """Roll the three LOGGING policies and the SOFTENED candidate policies through the
    DGP and record the reference-discount r each visits. The coverage gap for a
    (log, candidate) pair is the fraction of the softened candidate's visited states
    below the log's 1st percentile of r. The candidates rolled here are the same
    epsilon-greedy policies OPS scores (a prior version rolled the un-softened
    no-promo, which drives r fully to 0 and overstated the gap). Env-only, no
    training; cached once."""
    cfg = PromoConfig()
    env = PromoEnv(cfg, seed=0)
    B, K, T = 4000, cfg.K, cfg.horizon
    eps_c = RUN_CFG["cand_epsilon"]
    eps_i = RUN_CFG["incumbent_epsilon"]
    eps_m = RUN_CFG["mix_epsilon"]
    n_comp = len(DISCOUNTS) + 1

    def visited_r(policy, seed):
        rng = np.random.default_rng(seed)
        C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
        r = np.full(B, cfg.r_init)
        rs = []
        for t in range(T):
            obs = np.concatenate(
                [C, r[:, None], np.full((B, 1), t / T)], axis=1
            ).astype(np.float32)
            a = np.asarray(policy(env, obs, rng)).astype(int)
            rs.append(r.copy())
            r = (t * r + DISCOUNTS[a]) / (t + 1)
            C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
        return np.concatenate(rs)

    def make_mixture_policy():
        # per-EPISODE component assignment, vectorized over rows: components are
        # drawn at the episode start (t/T == 0) and kept for the whole episode.
        state = {"ks": None}

        def pol(env_, obs, rng):
            if state["ks"] is None or obs[0, -1] == 0.0:
                state["ks"] = rng.integers(n_comp, size=obs.shape[0])
            ks = state["ks"]
            greedy = np.empty(obs.shape[0], dtype=int)
            for k in range(len(DISCOUNTS)):
                greedy[ks == k] = k
            my_mask = ks == len(DISCOUNTS)
            if my_mask.any():
                greedy[my_mask] = np.asarray(myopic_batch(env_, obs[my_mask])).astype(
                    int
                )
            explore = rng.random(obs.shape[0]) < eps_m
            rand_a = rng.integers(len(DISCOUNTS), size=obs.shape[0])
            return np.where(explore, rand_a, greedy)

        return pol

    logs = {
        "ab": visited_r(uniform_batch, 0),
        "incumbent": visited_r(softened_batch(myopic_batch, eps_i), 1),
        "mixture": visited_r(make_mixture_policy(), 2),
    }
    cands = {
        "no_promo": visited_r(softened_batch(constant_batch(0), eps_c), 3),
        "myopic": visited_r(softened_batch(myopic_batch, eps_c), 4),
    }
    pct1 = {lg: float(np.quantile(r, 0.01)) for lg, r in logs.items()}
    gaps = {
        lg: {nm: float(np.mean(cr < pct1[lg])) for nm, cr in cands.items()}
        for lg in logs
    }
    return {
        "log_r": {lg: r for lg, r in logs.items()},
        "cand_r": {nm: r for nm, r in cands.items()},
        "log_1pct": pct1,
        "gaps": gaps,
    }


# --- driver ------------------------------------------------------------------
def compute_data(force=None):
    force = force or set()
    results = {}
    for regime in REGIMES:
        for seed in REGIME_SEEDS[regime]:
            comp = f"{regime}_s{seed}"
            forced = comp in force or regime in force or "shared" in force
            results[comp] = compute_or_load(
                CACHE_DIR,
                SCRIPT_NAME,
                comp,
                _component_config(regime, seed),
                run_one,
                regime,
                seed,
                force=forced,
            )
    refcov = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "refcov",
        {
            "tag": "reference_coverage_v2",
            "cand_epsilon": RUN_CFG["cand_epsilon"],
            "incumbent_epsilon": RUN_CFG["incumbent_epsilon"],
            "mix_epsilon": RUN_CFG["mix_epsilon"],
        },
        compute_reference_coverage,
        force=("refcov" in force or "shared" in force),
    )
    return {"results": results, "refcov": refcov}


# --- aggregation -------------------------------------------------------------
def _mean_se(xs):
    a = np.asarray(xs, dtype=float)
    a = a[
        np.isfinite(a)
    ]  # a degenerate ranking yields a NaN rank-corr; drop, don't poison
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return m, se


def aggregate(data):
    """Per (estimator, regime): mean+-SE of regret@1 and rank-correlation over seeds."""
    res = data["results"]
    agg = {est: {} for est in ESTIMATORS}
    for est in ESTIMATORS:
        for regime in REGIMES:
            regrets, corrs = [], []
            for seed in REGIME_SEEDS[regime]:
                cell = res[f"{regime}_s{seed}"]["ops"][est]
                regrets.append(cell["regret"])
                corrs.append(cell["rank_corr"])
            agg[est][regime] = {
                "regret": _mean_se(regrets),
                "rank_corr": _mean_se(corrs),
            }
    return agg


def paired_dm_contrast(data):
    """Within-seed paired difference of DM rank correlation, mixture minus A/B --
    the primary claim. Paired over the seeds the two regimes share."""
    res = data["results"]
    shared = [s for s in REGIME_SEEDS["mixture"] if s in REGIME_SEEDS["ab"]]
    diffs = [
        res[f"mixture_s{s}"]["ops"]["dm"]["rank_corr"]
        - res[f"ab_s{s}"]["ops"]["dm"]["rank_corr"]
        for s in shared
    ]
    return _mean_se(diffs)


# --- outputs -----------------------------------------------------------------
def _fmt(m, se):
    return f"{m:.3f} ({se:.3f})"


def write_table(agg, path):
    # rank estimators by mixture-regime rank-corr (the discriminating regime), best first
    order = sorted(
        ESTIMATORS, key=lambda e: agg[e]["mixture"]["rank_corr"][0], reverse=True
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Off-policy selection reliability of three estimators (direct method, DM; "
        r"per-decision importance sampling, PDIS; doubly robust, DR) under three logging regimes, on the targeted-"
        r"promotions MDP. Regret@1 is the value gap between the selected and the best "
        r"deployment policy (reward units, lower is better); rank correlation is Spearman "
        r"between estimated and true policy value (higher is better). Cells are mean "
        r"(standard error) over " + str(N_SEEDS) + r" seeds.}",
        r"\label{tab:field_ope_reliability}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{A/B (uniform)} & \multicolumn{2}{c}{Incumbent (narrow)} & "
        r"\multicolumn{2}{c}{Historical mixture} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Estimator & Regret@1 & Rank corr. & Regret@1 & Rank corr. "
        r"& Regret@1 & Rank corr. \\",
        r"\midrule",
    ]
    for est in order:
        cells = []
        for regime in REGIMES:
            a = agg[est][regime]
            cells += [_fmt(*a["regret"]), _fmt(*a["rank_corr"])]
        lines.append(f"{ESTIMATOR_LABEL[est]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Table written: {path}")


def _cand_stats(data):
    """Per candidate: true field value (on-policy value, pooled over seeds and regimes),
    and, per regime, the fitted-Q direct-method error and the per-trajectory importance-
    sampling effective sample size. These are the mechanism behind the reliability table."""
    res = data["results"]
    stats = {}
    for nm in CAND_ORDER:
        onp_all = [
            res[f"{rg}_s{s}"]["epv"][nm]["on_policy"]
            for rg in REGIMES
            for s in REGIME_SEEDS[rg]
        ]
        row = {"true": _mean_se(onp_all)}
        for rg in REGIMES:
            dmerr = [
                res[f"{rg}_s{s}"]["epv"][nm]["dm"]
                - res[f"{rg}_s{s}"]["epv"][nm]["on_policy"]
                for s in REGIME_SEEDS[rg]
            ]
            ess = [
                res[f"{rg}_s{s}"]["diag"]["per_cand"][nm]["ess"]
                for s in REGIME_SEEDS[rg]
            ]
            row[rg] = {"dm_err": _mean_se(dmerr), "ess": _mean_se(ess)}
        stats[nm] = row
    return stats


def write_candidates_table(data, path):
    """Per-candidate mechanism table: DM error and IS effective sample size per regime."""
    stats = _cand_stats(data)
    order = sorted(CAND_ORDER, key=lambda nm: stats[nm]["true"][0], reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-candidate mechanism behind Table~\ref{tab:field_ope_reliability}. "
        r"True value is the on-policy field value of the softened candidate; DM error is "
        r"the fitted-Q direct-method estimate minus the true value (negative means "
        r"under-valued); importance-sampling (IS) effective sample size is $(\sum w)^2/\sum w^2$ over the log's "
        + str(RUN_CFG["n_traj"])
        + r" trajectories. Rows are rank-ordered by true value. Cell entries are means "
        r"over " + str(N_SEEDS) + r" seeds.}",
        r"\label{tab:field_ope_candidates}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{DM error} & \multicolumn{3}{c}{IS eff.\ sample size} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Candidate & True value & A/B & Incumb. & Mixture & A/B & Incumb. & Mixture \\",
        r"\midrule",
    ]
    for nm in order:
        s = stats[nm]
        cells = [f"{s['true'][0]:.3f}"]
        for rg in REGIMES:
            cells.append(f"{s[rg]['dm_err'][0]:+.3f}")
        for rg in REGIMES:
            cells.append(f"{s[rg]['ess'][0]:.1f}")
        lines.append(f"{CAND_LABEL[nm]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Candidates table written: {path}")


def write_macros(data, agg, path):
    """Emit the few prose numbers as LaTeX macros so nothing is hand-typed."""
    res = data["results"]
    gaps = data["refcov"]["gaps"]
    diff_m, diff_se = paired_dm_contrast(data)
    dmerr_best_ab = _mean_se(
        [
            res[f"ab_s{s}"]["epv"]["no_promo"]["dm"]
            - res[f"ab_s{s}"]["epv"]["no_promo"]["on_policy"]
            for s in REGIME_SEEDS["ab"]
        ]
    )[0]
    dmerr_best_mix = _mean_se(
        [
            res[f"mixture_s{s}"]["epv"]["no_promo"]["dm"]
            - res[f"mixture_s{s}"]["epv"]["no_promo"]["on_policy"]
            for s in REGIME_SEEDS["mixture"]
        ]
    )[0]
    ess_best_ab = _mean_se(
        [
            res[f"ab_s{s}"]["diag"]["per_cand"]["no_promo"]["ess"]
            for s in REGIME_SEEDS["ab"]
        ]
    )[0]
    ess_aligned_mix = _mean_se(
        [
            res[f"mixture_s{s}"]["diag"]["per_cand"][nm]["ess"]
            for s in REGIME_SEEDS["mixture"]
            for nm in ALIGNED_CANDS
        ]
    )[0]
    macros = {
        "fieldopecovgapab": f"{gaps['ab']['no_promo'] * 100:.0f}",
        "fieldopecovgapinc": f"{gaps['incumbent']['no_promo'] * 100:.0f}",
        "fieldopecovgapmix": f"{gaps['mixture']['no_promo'] * 100:.0f}",
        "fieldopedmab": f"{agg['dm']['ab']['rank_corr'][0]:.2f}",
        "fieldopedminc": f"{agg['dm']['incumbent']['rank_corr'][0]:.2f}",
        "fieldopedmmix": f"{agg['dm']['mixture']['rank_corr'][0]:.2f}",
        "fieldopedmdiffabmix": f"{diff_m:.2f}",
        "fieldopedmdiffabmixse": f"{diff_se:.2f}",
        "fieldopedrab": f"{agg['dr']['ab']['rank_corr'][0]:.2f}",
        "fieldopedrinc": f"{agg['dr']['incumbent']['rank_corr'][0]:.2f}",
        "fieldopedrmix": f"{agg['dr']['mixture']['rank_corr'][0]:.2f}",
        "fieldoperegretab": f"{agg['dm']['ab']['regret'][0]:.2f}",
        "fieldoperegretmix": f"{agg['dm']['mixture']['regret'][0]:.2f}",
        "fieldopeseeds": str(N_SEEDS),
        "fieldopehorizon": str(RUN_CFG["horizon"]),
        "fieldopefqesteps": str(RUN_CFG["fqe_steps"]),
        "fieldopedmerrbestab": f"{dmerr_best_ab:.2f}",
        "fieldopedmerrbestmix": f"{dmerr_best_mix:.2f}",
        "fieldopeessbest": f"{ess_best_ab:.1f}",
        "fieldopeessalignedmix": f"{ess_aligned_mix:.0f}",
    }
    with open(path, "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  Macros written: {path}")


def write_figure(data, agg, path):
    """Two panels. Left: reference-discount occupancy of the three logs against the
    softened true-best candidate -- the coverage mechanism. Right: DM estimate vs true
    value under the A/B and mixture logs, pooled over seeds -- the consequence."""
    import matplotlib.pyplot as plt

    res = data["results"]
    rc = data["refcov"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # left: reference-discount coverage
    bins = np.linspace(0.0, DISCOUNTS[-1], 40)
    log_colors = {
        "ab": COLORS["gray"],
        "incumbent": COLORS["orange"],
        "mixture": COLORS["purple"],
    }
    for lg in REGIMES:
        axL.hist(
            rc["log_r"][lg],
            bins=bins,
            density=True,
            alpha=0.45,
            color=log_colors[lg],
            label=f"{REGIME_LABEL[lg]} log",
        )
    axL.hist(
        rc["cand_r"]["no_promo"],
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["green"],
        label="no-promo cand. (true best)",
    )
    axL.axvline(rc["log_1pct"]["ab"], **BENCH_STYLE, label="A/B log 1st pct")
    axL.set_xlabel("reference discount $r$ visited")
    axL.set_ylabel("density")
    axL.set_title(
        f"State coverage: {rc['gaps']['ab']['no_promo'] * 100:.0f}% of best-candidate\n"
        f"states below the A/B log's 1st percentile"
    )
    axL.legend(fontsize=6, loc="upper right")

    # right: DM estimate vs true field value, A/B vs mixture, pooled over seeds
    cand_color = {
        "no_promo": COLORS["green"],
        "disc5": COLORS["blue"],
        "disc10": COLORS["cyan"],
        "myopic": COLORS["orange"],
        "uniform": COLORS["gray"],
        "cql": COLORS["red"],
    }
    marker = {"ab": "o", "mixture": "^"}
    for rg in ["ab", "mixture"]:
        for pol, col in cand_color.items():
            xs = [res[f"{rg}_s{s}"]["epv"][pol]["on_policy"] for s in REGIME_SEEDS[rg]]
            ys = [res[f"{rg}_s{s}"]["epv"][pol]["dm"] for s in REGIME_SEEDS[rg]]
            axR.scatter(
                xs,
                ys,
                s=22,
                alpha=0.7,
                color=col,
                marker=marker[rg],
                label=pol if rg == "ab" else None,
                zorder=3,
            )
    allv = [
        res[f"{rg}_s{s}"]["epv"][p][k]
        for rg in ["ab", "mixture"]
        for s in REGIME_SEEDS[rg]
        for p in cand_color
        for k in ("on_policy", "dm")
    ]
    lo, hi = min(allv), max(allv)
    axR.plot([lo, hi], [lo, hi], **BENCH_STYLE, label="perfect (45$^\\circ$)")
    axR.set_xlabel("true field value")
    axR.set_ylabel("DM (fitted-Q) estimate")
    axR.set_title(
        f"DM rank corr.: {agg['dm']['ab']['rank_corr'][0]:.2f} (A/B, circles) vs "
        f"{agg['dm']['mixture']['rank_corr'][0]:.2f} (mixture, triangles)"
    )
    axR.legend(fontsize=6, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure written: {path}")


def print_summary(data, agg):
    print("=" * 72)
    print("OPE RELIABILITY UNDER THREE LOGGING REGIMES -- ch13 capstone")
    print("=" * 72)
    print("\nConfig:")
    for k, v in RUN_CFG.items():
        print(f"  {k:18s} {v}")
    print(f"  seeds              {REGIME_SEEDS['ab']}")

    print("\nOff-policy SELECTION reliability (mean [SE] over seeds):")
    hdr = f"{'estimator':10s} {'regime':20s} {'regret@1':>18s} {'rank_corr':>18s}"
    print(hdr)
    print("-" * len(hdr))
    for est in ESTIMATORS:
        for regime in REGIMES:
            rg = agg[est][regime]["regret"]
            rc = agg[est][regime]["rank_corr"]
            print(
                f"{ESTIMATOR_LABEL[est]:10s} {REGIME_LABEL[regime]:20s} "
                f"{_fmt(*rg):>18s} {_fmt(*rc):>18s}"
            )
    diff_m, diff_se = paired_dm_contrast(data)
    print(
        f"\nPaired DM rank-corr contrast (mixture - A/B, within seed): "
        f"{diff_m:+.3f} ({diff_se:.3f})"
    )

    # true (MC) field value of each candidate, one representative seed, as a sanity anchor
    print("\nTrue field value of each candidate (independent MC, seed 0, mean [SE]):")
    tv = data["results"]["ab_s0"]["true_mc"]
    for nm, (m, se) in sorted(tv.items(), key=lambda kv: -kv[1][0]):
        print(f"  {nm:10s} {m:.3f} [{se:.3f}]")

    print("\nReference-coverage gaps (softened no-promo candidate vs each log):")
    for lg in REGIMES:
        g = data["refcov"]["gaps"][lg]["no_promo"]
        print(f"  {REGIME_LABEL[lg]:20s} {g * 100:5.1f}% below the log's 1st pctile")

    print("\nOutput files:")
    print(f"  {os.path.join(OUT_DIR, SCRIPT_NAME + '_table.tex')}")
    print(f"  {os.path.join(OUT_DIR, SCRIPT_NAME + '_candidates.tex')}")
    print(f"  {os.path.join(OUT_DIR, SCRIPT_NAME + '_mechanism.png')}")


# --- diagnostics logging + alerts (log every sample/step, loud flags) --------
def emit_diagnostics(data):
    """Run every step-health check across all cells, print DETAILED per-cell logs and
    AGGREGATED summaries, dump a machine-readable JSON, and return the AlertBank so the
    caller can stamp the run INVALID on a FATAL alert. Alerts recomputed from cached raw
    metrics, so --plots-only re-checks without recomputation."""
    res = data["results"]
    bank = diag.AlertBank()

    print("\n" + "=" * 72)
    print("STEP DIAGNOSTICS -- detailed, per (regime, seed)")
    print("=" * 72)
    for regime in REGIMES:
        for seed in REGIME_SEEDS[regime]:
            cell = res[f"{regime}_s{seed}"]
            where = f"{regime}/s{seed}"
            d = cell["diag"]
            cov = d["coverage"]
            mc = {nm: v[0] for nm, v in cell["true_mc"].items()}
            onp = {nm: cell["epv"][nm]["on_policy"] for nm in cell["epv"]}
            dm = {nm: cell["epv"][nm]["dm"] for nm in cell["epv"]}
            ess = {nm: d["per_cand"][nm]["ess"] for nm in d["per_cand"]}
            surv = {nm: d["per_cand"][nm]["surviving"] for nm in d["per_cand"]}
            rc = {e: cell["ops"][e]["rank_corr"] for e in ESTIMATORS}

            # run the checks (register alerts)
            diag.check_support(cov, where, bank)
            if not cov.get("traj_major_ok", True):
                bank.add(
                    "SUPPORT_HOLE", where, "log not trajectory-major; IS reshape unsafe"
                )
            diag.check_fqe_health(dm, mc, where, bank)
            diag.check_truth(mc, onp, where, bank)
            diag.check_dm_calibration(
                dm, onp, CAL_CAND[regime], where, bank, fatal=CAL_FATAL[regime]
            )
            diag.check_clone_fidelity(d.get("fidelity", {}), where, bank)
            if regime == "mixture" and "mixture_component_counts" in d:
                diag.check_mixture_log(
                    d["mixture_component_counts"],
                    d["n_components"],
                    d["pscore_min"],
                    d["mix_epsilon"],
                    len(DISCOUNTS),
                    where,
                    bank,
                )
            diag.check_is_degeneracy(ess, surv, where, bank)
            diag.check_rank_inversion(rc, where, bank)

            # detailed log line
            print(
                f"\n[{where}] fqe_epochs={d['fqe_expected_epochs']} "
                f"log_entropy={cov['entropy']:.3f} action_counts={cov['counts']} "
                f"support_holes={cov['support_holes']} "
                f"wallclock={d.get('wallclock_s', float('nan')):.0f}s"
            )
            if d.get("fidelity"):
                fid = " ".join(f"{k}={v:.3f}" for k, v in d["fidelity"].items())
                print(f"  clone fidelity: {fid}")
            if regime == "mixture" and "mixture_component_counts" in d:
                print(
                    f"  mixture components drawn: {d['mixture_component_counts']} "
                    f"pscore_min={d['pscore_min']:.4f}"
                )
            print(
                f"  {'candidate':10s} {'true':>7s} {'onpol':>7s} {'DM':>7s} "
                f"{'PDIS':>8s} {'DR':>8s} {'IS_ESS':>7s} {'surv':>5s}"
            )
            for nm in CAND_ORDER:
                e = cell["epv"][nm]
                print(
                    f"  {nm:10s} {mc[nm]:7.3f} {onp[nm]:7.3f} {e['dm']:7.3f} "
                    f"{e['pdis']:8.3f} {e['dr']:8.3f} {ess[nm]:7.2f} {surv[nm]:5d}"
                )
            print(
                "  rank_corr: "
                + " ".join(f"{ESTIMATOR_LABEL[e]}={rc[e]:.3f}" for e in ESTIMATORS)
            )

    # aggregated diagnostics
    print("\n" + "=" * 72)
    print("STEP DIAGNOSTICS -- aggregated over seeds (mean [SE])")
    print("=" * 72)
    for regime in REGIMES:
        print(f"\n{REGIME_LABEL[regime]} log:")
        print(f"  {'candidate':10s} {'true':>14s} {'IS_ESS':>14s} {'DM_err':>16s}")
        for nm in CAND_ORDER:
            trues = [
                res[f"{regime}_s{s}"]["epv"][nm]["on_policy"]
                for s in REGIME_SEEDS[regime]
            ]
            esss = [
                res[f"{regime}_s{s}"]["diag"]["per_cand"][nm]["ess"]
                for s in REGIME_SEEDS[regime]
            ]
            # DM error against the on-policy truth (the value the estimators are scored on)
            dmerr = [
                res[f"{regime}_s{s}"]["epv"][nm]["dm"]
                - res[f"{regime}_s{s}"]["epv"][nm]["on_policy"]
                for s in REGIME_SEEDS[regime]
            ]
            print(
                f"  {nm:10s} {_fmt(*_mean_se(trues)):>14s} "
                f"{_fmt(*_mean_se(esss)):>14s} {_fmt(*_mean_se(dmerr)):>16s}"
            )

    # seed-pooled calibration-bias gate per regime (refit noise averages out; a
    # nonzero pooled mean is systematic miscalibration)
    for regime in REGIMES:
        cal = CAL_CAND[regime]
        cal_errs = [
            res[f"{regime}_s{s}"]["epv"][cal]["dm"]
            - res[f"{regime}_s{s}"]["epv"][cal]["on_policy"]
            for s in REGIME_SEEDS[regime]
        ]
        m = diag.check_dm_calibration_bias(
            cal_errs, regime, cal, bank, fatal=CAL_FATAL[regime]
        )
        print(
            f"\n{REGIME_LABEL[regime]}: pooled DM calibration bias on {cal}: "
            f"{m['mean_err']:+.3f} over {m['n']} seeds (gate {diag.THRESH['dm_cal_bias']:.2f})"
        )

    # candidate-identity cross-check: the menu must be identical across regimes of a seed
    for seed in REGIME_SEEDS["ab"]:
        hashes = {
            rg: res[f"{rg}_s{seed}"]["diag"].get("cand_hash")
            for rg in REGIMES
            if seed in REGIME_SEEDS[rg]
        }
        vals = [h for h in hashes.values() if h]
        if len(vals) > 1 and any(v != vals[0] for v in vals[1:]):
            bank.add(
                "CAND_DRIFT",
                f"s{seed}",
                f"candidate greedy-action hashes differ across regimes: {hashes}",
            )

    # machine-readable dump
    dump_path = os.path.join(CACHE_DIR, f"{SCRIPT_NAME}_diagnostics.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(dump_path, "w") as f:
        json.dump(
            {"alerts": bank.fired(), "thresholds": diag.THRESH},
            f,
            indent=2,
            default=float,
        )

    print("\n" + bank.banner())
    print(f"\nDiagnostics dump: {dump_path}")
    return bank


def generate_outputs(data):
    agg = aggregate(data)
    write_table(agg, os.path.join(OUT_DIR, f"{SCRIPT_NAME}_table.tex"))
    write_candidates_table(data, os.path.join(OUT_DIR, f"{SCRIPT_NAME}_candidates.tex"))
    write_macros(data, agg, os.path.join(OUT_DIR, f"{SCRIPT_NAME}_macros.tex"))
    write_figure(data, agg, os.path.join(OUT_DIR, f"{SCRIPT_NAME}_mechanism.png"))
    print_summary(data, agg)
    bank = emit_diagnostics(data)
    return bank


# --- pilot: FQE convergence sweep --------------------------------------------
def run_pilot():
    """Convergence pilot for the FQE budget (never writes final outputs). For seeds
    {0,1} x regimes {ab, mixture} x fqe_steps {20000, 50000}: run the full cell but
    with the reduced candidate menu {uniform, no_promo, myopic}, and report the DM
    calibration error and cross-budget DM movement per candidate. Selection rule:
    smallest budget with |cal err| < dm_cal_abs on every pilot cell AND max DM
    movement 20k->50k < 0.05."""
    cfg = PromoConfig()
    budgets = [20000, 50000]
    keep = ["uniform", "no_promo", "myopic"]
    rows = {}
    for regime in ["ab", "mixture"]:
        for seed in [0, 1]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.set_num_threads(4)
            ab_logged, _ = generate_log(cfg, "ab", n_traj=RUN_CFG["n_traj"], seed=seed)
            if regime == "ab":
                logged = ab_logged
            else:
                logged, _ = generate_log(
                    cfg,
                    regime,
                    n_traj=RUN_CFG["n_traj"],
                    seed=seed,
                    mix_epsilon=RUN_CFG["mix_epsilon"],
                )
            cands, _fid = train_candidates(
                cfg,
                ab_logged,
                seed,
                bc_steps=RUN_CFG["bc_steps"],
                cql_steps=RUN_CFG["cql_steps"],
                cand_epsilon=RUN_CFG["cand_epsilon"],
            )
            cands = {k: cands[k] for k in keep}
            for fs in budgets:
                comp = f"pilot_{regime}_s{seed}_f{fs}"

                def _cell(fs=fs, logged=logged, cands=cands, seed=seed):
                    t0 = time.time()
                    inp = build_ope_inputs(
                        cfg,
                        logged,
                        cands,
                        seed,
                        fqe_steps=fs,
                        n_on_policy=RUN_CFG["n_on_policy"],
                        fqe_hidden=RUN_CFG["fqe_hidden"],
                    )
                    epv = evaluate_ope(logged, inp).estimate_policy_value(inp)
                    return {
                        "epv": {
                            p: {k: float(v) for k, v in d.items()}
                            for p, d in epv.items()
                        },
                        "wallclock_s": float(time.time() - t0),
                    }

                rows[comp] = compute_or_load(
                    CACHE_DIR,
                    SCRIPT_NAME,
                    comp,
                    {
                        **RUN_CFG,
                        "pilot": True,
                        "regime": regime,
                        "seed": seed,
                        "fqe_steps": fs,
                    },
                    _cell,
                )

    print("=" * 72)
    print("FQE CONVERGENCE PILOT")
    print("=" * 72)
    print(
        f"{'cell':16s} {'budget':>7s} {'cand':10s} {'DM':>8s} {'true':>8s} "
        f"{'err':>8s} {'secs':>6s}"
    )
    worst_cal = {fs: 0.0 for fs in budgets}
    worst_move = 0.0
    for regime in ["ab", "mixture"]:
        for seed in [0, 1]:
            cal = CAL_CAND[regime]
            for fs in budgets:
                cell = rows[f"pilot_{regime}_s{seed}_f{fs}"]
                for nm in keep:
                    e = cell["epv"][nm]
                    err = e["dm"] - e["on_policy"]
                    if nm == cal:
                        worst_cal[fs] = max(worst_cal[fs], abs(err))
                    print(
                        f"{regime + '/s' + str(seed):16s} {fs:7d} {nm:10s} "
                        f"{e['dm']:8.3f} {e['on_policy']:8.3f} {err:+8.3f} "
                        f"{cell['wallclock_s']:6.0f}"
                    )
            for nm in keep:
                lo = rows[f"pilot_{regime}_s{seed}_f{budgets[0]}"]["epv"][nm]["dm"]
                hi = rows[f"pilot_{regime}_s{seed}_f{budgets[1]}"]["epv"][nm]["dm"]
                worst_move = max(worst_move, abs(hi - lo))
    print(f"\nworst |calibration err| at 20k: {worst_cal[20000]:.3f}")
    print(f"worst |calibration err| at 50k: {worst_cal[50000]:.3f}")
    print(f"worst |DM movement| 20k -> 50k: {worst_move:.3f}")
    tol = diag.THRESH["dm_cal_abs"]
    if worst_cal[20000] < tol and worst_move < 0.05:
        print("RECOMMENDATION: fqe_steps=20000 (calibrated, stable)")
    elif worst_cal[50000] < tol:
        print("RECOMMENDATION: fqe_steps=50000")
    else:
        print(
            "RECOMMENDATION: neither budget calibrates; escalate to 100000 "
            "or report DM at best-achieved calibration (branch-B prose)."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_component_args(parser)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run the FQE convergence pilot sweep and exit (no final outputs)",
    )
    args = parser.parse_args()
    if args.pilot:
        run_pilot()
        return
    force = parse_force_set(args)
    if args.plots_only:
        force = set()
    data = compute_data(force=force)
    if not args.data_only:
        bank = generate_outputs(data)
        if bank.has_fatal:
            print(
                "\nEXIT: run INVALID (FATAL alert). See the ALERTS FIRED banner above."
            )
            sys.exit(2)


if __name__ == "__main__":
    main()
