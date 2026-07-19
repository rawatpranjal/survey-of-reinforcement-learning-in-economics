# OPE reliability under two logging regimes -- ch13 field-deployments capstone.
# Question: does offline off-policy evaluation and selection pick the right deployment
# policy? We show the answer depends on the LOGGING REGIME. Under an A/B log (uniform,
# full support) OPS ranks the deployment menu well; under an observational log (an
# epsilon-greedy imitation of the myopic incumbent, support-collapsing) OPS selection
# degrades. The DGP, behavior policies, candidate menu, and OPE/OPS wiring live in
# promo_env.py and pipeline.py; this script is the experiment driver + outputs.
import os
import sys
import json
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
SEEDS = list(range(N_SEEDS))
REGIMES = ["ab", "observational"]
REGIME_LABEL = {"ab": "A/B", "observational": "Observational"}
ESTIMATORS = ["dm", "pdis", "dr"]  # scope-rl estimator_name keys
ESTIMATOR_LABEL = {"dm": "DM", "pdis": "PDIS", "dr": "DR"}
CAND_ORDER = ["no_promo", "disc5", "disc10", "myopic", "uniform", "cql"]
CAND_LABEL = {
    "no_promo": "No promo",
    "disc5": r"5\% discount",
    "disc10": r"10\% discount",
    "myopic": "Myopic incumbent",
    "uniform": "Uniform",
    "cql": "CQL (offline RL)",
}

# Training budgets (one place; hashed into every component so a change re-runs cleanly).
# fqe_steps MUST be a multiple of FQE_EPOCH_LEN (10000): d3rlpy runs n_steps//n_steps_per_epoch
# epochs with n_steps_per_epoch defaulting to 10000, so anything below trains the FQE for
# ZERO gradient steps. This is the bug that voided the first run; the guard below refuses it.
RUN_CFG = {
    "n_traj": 500,
    "bc_steps": 1200,
    "cql_steps": 2000,
    "fqe_steps": 20000,
    "n_on_policy": 500,
    "true_n_ep": 8000,
    "cand_epsilon": 0.3,  # soften candidates so IS has finite overlap (0.0 = deterministic)
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


# --- one (regime, seed) cell -------------------------------------------------
def run_one(regime, seed):
    """Full OPE->OPS pass for one logging regime and seed. Returns plain-python data."""
    cfg = PromoConfig()
    logged = generate_log(cfg, regime, n_traj=RUN_CFG["n_traj"], seed=seed)
    cands = train_candidates(
        cfg,
        logged,
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
    )
    ope = evaluate_ope(logged, input_dict)
    ops = ops_metrics(ope, input_dict)  # {est: {regret, rank_corr, rankings}}
    epv = ope.estimate_policy_value(input_dict)  # {policy: {dm,pdis,dr,on_policy}}
    # coerce to plain floats for a small, portable cache payload
    epv_plain = {pol: {k: float(v) for k, v in d.items()} for pol, d in epv.items()}
    tv_plain = {nm: [float(m), float(se)] for nm, (m, se) in tv.items()}
    diagnostics = _cell_diagnostics(logged, input_dict, list(cands.keys()))
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


# --- reference-state coverage (the mechanism behind the DM inversion) ---------
def compute_reference_coverage():
    """Roll the A/B (uniform) logging policy and the idealized candidate policies through
    the DGP and record the reference-discount r they visit. The best policy (no_promo)
    drives r to ~0, a region the uniform log almost never visits, so FQE must extrapolate
    and under-values it. Env-only, no training; cached once."""
    cfg = PromoConfig()
    env = PromoEnv(cfg, seed=0)
    B, K, T = 4000, cfg.K, cfg.horizon

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

    log_r = visited_r(uniform_batch, 0)
    lo = float(np.quantile(log_r, 0.01))
    no_promo_r = visited_r(constant_batch(0), 1)
    gap = float(np.mean(no_promo_r < lo))  # fraction of best-policy states off the log
    return {
        "log_r": log_r,
        "no_promo_r": no_promo_r,
        "myopic_r": visited_r(myopic_batch, 2),
        "log_1pct": lo,
        "coverage_gap_frac": gap,
    }


# --- driver ------------------------------------------------------------------
def compute_data(force=None):
    force = force or set()
    results = {}
    for regime in REGIMES:
        for seed in SEEDS:
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
        {"tag": "reference_coverage_v1"},
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
            for seed in SEEDS:
                cell = res[f"{regime}_s{seed}"]["ops"][est]
                regrets.append(cell["regret"])
                corrs.append(cell["rank_corr"])
            agg[est][regime] = {
                "regret": _mean_se(regrets),
                "rank_corr": _mean_se(corrs),
            }
    return agg


# --- outputs -----------------------------------------------------------------
def _fmt(m, se):
    return f"{m:.3f} ({se:.3f})"


def write_table(agg, path):
    # rank estimators by observational rank-corr (the discriminating regime), best first
    order = sorted(
        ESTIMATORS, key=lambda e: agg[e]["observational"]["rank_corr"][0], reverse=True
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Off-policy selection reliability of three estimators (DM, per-decision "
        r"importance sampling, doubly robust) under two logging regimes, on the targeted-"
        r"promotions MDP. Regret@1 is the value gap between the selected and the best "
        r"deployment policy (reward units, lower is better); rank correlation is Spearman "
        r"between estimated and true policy value (higher is better). Cells are mean "
        r"(standard error) over " + str(N_SEEDS) + r" seeds.}",
        r"\label{tab:field_ope_reliability}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{A/B log (full support)} & "
        r"\multicolumn{2}{c}{Observational log} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Estimator & Regret@1 & Rank corr. & Regret@1 & Rank corr. \\",
        r"\midrule",
    ]
    for est in order:
        ab = agg[est]["ab"]
        ob = agg[est]["observational"]
        lines.append(
            f"{ESTIMATOR_LABEL[est]} & {_fmt(*ab['regret'])} & {_fmt(*ab['rank_corr'])} "
            f"& {_fmt(*ob['regret'])} & {_fmt(*ob['rank_corr'])} \\\\"
        )
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
            res[f"{rg}_s{s}"]["epv"][nm]["on_policy"] for rg in REGIMES for s in SEEDS
        ]
        row = {"true": _mean_se(onp_all)}
        for rg in REGIMES:
            dmerr = [
                res[f"{rg}_s{s}"]["epv"][nm]["dm"]
                - res[f"{rg}_s{s}"]["epv"][nm]["on_policy"]
                for s in SEEDS
            ]
            ess = [res[f"{rg}_s{s}"]["diag"]["per_cand"][nm]["ess"] for s in SEEDS]
            row[rg] = {"dm_err": _mean_se(dmerr), "ess": _mean_se(ess)}
        stats[nm] = row
    return stats


def write_candidates_table(data, path):
    """Per-candidate mechanism table: why the ranking inverts. The true-best policy carries
    the largest negative DM error (FQE under-values it under state-occupancy shift) and an
    effective sample size near one (IS cannot weight it over the horizon), while the log's
    own uniform policy is the only one importance sampling can weight."""
    stats = _cand_stats(data)
    order = sorted(CAND_ORDER, key=lambda nm: stats[nm]["true"][0], reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-candidate mechanism behind the reliability of "
        r"Table~\ref{tab:field_ope_reliability}. True value is the Monte-Carlo on-policy "
        r"field value; DM error is the fitted-Q direct-method estimate minus the true value "
        r"(negative means under-valued); IS effective sample size is $(\sum w)^2/\sum w^2$ "
        r"over the log's 500 trajectories. Rows are rank-ordered by true value; the true-best "
        r"policy has the largest negative DM error and an effective sample size near one, "
        r"while the uniform policy that generated the A/B log is the only one importance "
        r"sampling can weight. Mean (standard error) over "
        + str(N_SEEDS)
        + r" seeds.}",
        r"\label{tab:field_ope_candidates}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{DM error} & "
        r"\multicolumn{2}{c}{IS eff.\ sample size} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Candidate & True value & A/B & Obs. & A/B & Obs. \\",
        r"\midrule",
    ]
    for nm in order:
        s = stats[nm]
        lines.append(
            f"{CAND_LABEL[nm]} & {_fmt(*s['true'])} & {_fmt(*s['ab']['dm_err'])} "
            f"& {_fmt(*s['observational']['dm_err'])} & {_fmt(*s['ab']['ess'])} "
            f"& {_fmt(*s['observational']['ess'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Candidates table written: {path}")


def write_macros(data, agg, path):
    """Emit the few prose numbers as LaTeX macros so nothing is hand-typed."""
    res = data["results"]
    gap = data["refcov"]["coverage_gap_frac"] * 100
    dm_ab = agg["dm"]["ab"]["rank_corr"][0]
    dr_ab = agg["dr"]["ab"]["rank_corr"][0]
    dr_ob = agg["dr"]["observational"]["rank_corr"][0]
    # mechanism numbers for the prose: the true-best policy's DM error and IS ESS under A/B
    dmerr_best_ab = _mean_se(
        [
            res[f"ab_s{s}"]["epv"]["no_promo"]["dm"]
            - res[f"ab_s{s}"]["epv"]["no_promo"]["on_policy"]
            for s in SEEDS
        ]
    )[0]
    ess_best_ab = _mean_se(
        [res[f"ab_s{s}"]["diag"]["per_cand"]["no_promo"]["ess"] for s in SEEDS]
    )[0]
    macros = {
        "fieldopecoveragegap": f"{gap:.0f}",
        "fieldopedmab": f"{dm_ab:.2f}",
        "fieldopedrab": f"{dr_ab:.2f}",
        "fieldopedrobs": f"{dr_ob:.2f}",
        "fieldopeseeds": str(N_SEEDS),
        "fieldopehorizon": str(RUN_CFG["horizon"]),
        "fieldopedmerrbestab": f"{dmerr_best_ab:.2f}",
        "fieldopeessbest": f"{ess_best_ab:.1f}",
    }
    with open(path, "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  Macros written: {path}")


def write_figure(data, agg, path):
    """Two panels: (left) the reference-state coverage gap that makes the problem hard --
    the best policy (no_promo) visits low-r states the A/B log almost never sees; (right)
    the consequence -- the fitted-Q direct method under-values exactly the highest-value
    policies under the A/B log, so its estimated ranking inverts the truth."""
    import matplotlib.pyplot as plt

    res = data["results"]
    rc = data["refcov"]
    cand_color = {
        "no_promo": COLORS["green"],
        "disc5": COLORS["blue"],
        "disc10": COLORS["cyan"],
        "myopic": COLORS["orange"],
        "uniform": COLORS["gray"],
        "cql": COLORS["red"],
    }
    fig, (axL, axR) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # left: reference-discount coverage
    bins = np.linspace(0.0, DISCOUNTS[-1], 40)
    axL.hist(
        rc["log_r"],
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["gray"],
        label="A/B log (uniform behavior)",
    )
    axL.hist(
        rc["no_promo_r"],
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["green"],
        label="no-promo (true best)",
    )
    axL.hist(
        rc["myopic_r"],
        bins=bins,
        density=True,
        alpha=0.45,
        color=COLORS["orange"],
        label="myopic",
    )
    axL.axvline(rc["log_1pct"], **BENCH_STYLE, label="A/B log 1st pct")
    axL.set_xlabel("reference discount $r$ visited")
    axL.set_ylabel("density")
    axL.set_title(
        f"State coverage: {rc['coverage_gap_frac'] * 100:.0f}% of best-policy\n"
        f"states below the A/B log's 1st percentile"
    )
    axL.legend(fontsize=7, loc="upper right")

    # right: DM estimate vs true field value under A/B (full support), pooled over seeds.
    # x is the on-policy value (the same epsilon-greedy candidate OPS ranks against), not
    # the greedy MC, so the axis is exactly the truth the estimators are scored on.
    for pol, col in cand_color.items():
        xs = [res[f"ab_s{s}"]["epv"][pol]["on_policy"] for s in SEEDS]
        ys = [res[f"ab_s{s}"]["epv"][pol]["dm"] for s in SEEDS]
        axR.scatter(xs, ys, s=26, alpha=0.75, color=col, label=pol, zorder=3)
    allv = [
        res[f"ab_s{s}"]["epv"][p]["on_policy"] for s in SEEDS for p in cand_color
    ] + [res[f"ab_s{s}"]["epv"][p]["dm"] for s in SEEDS for p in cand_color]
    lo, hi = min(allv), max(allv)
    axR.plot([lo, hi], [lo, hi], **BENCH_STYLE, label="perfect (45$^\\circ$)")
    rcm = agg["dm"]["ab"]["rank_corr"][0]
    axR.set_xlabel("true field value")
    axR.set_ylabel("DM (fitted-Q) estimate, A/B log")
    axR.set_title(f"DM inverts under full support:\nrank corr. {rcm:.2f}")
    axR.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure written: {path}")


def print_summary(data, agg):
    print("=" * 72)
    print("OPE RELIABILITY UNDER TWO LOGGING REGIMES -- ch13 capstone")
    print("=" * 72)
    print("\nConfig:")
    for k, v in RUN_CFG.items():
        print(f"  {k:14s} {v}")
    print(f"  seeds          {SEEDS}")

    print("\nOff-policy SELECTION reliability (mean [SE] over seeds):")
    hdr = f"{'estimator':10s} {'regime':14s} {'regret@1':>18s} {'rank_corr':>18s}"
    print(hdr)
    print("-" * len(hdr))
    for est in ESTIMATORS:
        for regime in REGIMES:
            rg = agg[est][regime]["regret"]
            rc = agg[est][regime]["rank_corr"]
            print(
                f"{ESTIMATOR_LABEL[est]:10s} {REGIME_LABEL[regime]:14s} "
                f"{_fmt(*rg):>18s} {_fmt(*rc):>18s}"
            )

    # true (MC) field value of each candidate, one representative seed, as a sanity anchor
    print("\nTrue field value of each candidate (independent MC, seed 0, mean [SE]):")
    tv = data["results"]["ab_s0"]["true_mc"]
    for nm, (m, se) in sorted(tv.items(), key=lambda kv: -kv[1][0]):
        print(f"  {nm:10s} {m:.3f} [{se:.3f}]")

    print("\nOutput files:")
    print(f"  {os.path.join(OUT_DIR, SCRIPT_NAME + '_table.tex')}")
    print(f"  {os.path.join(OUT_DIR, SCRIPT_NAME + '_mechanism.png')}")


# --- diagnostics logging + alerts (user directive: log every sample/step, loud flags) ----
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
        for seed in SEEDS:
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
            diag.check_is_degeneracy(ess, surv, where, bank)
            diag.check_rank_inversion(rc, where, bank)

            # detailed log line
            print(
                f"\n[{where}] fqe_epochs={d['fqe_expected_epochs']} "
                f"log_entropy={cov['entropy']:.3f} action_counts={cov['counts']} "
                f"support_holes={cov['support_holes']}"
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
            trues = [res[f"{regime}_s{s}"]["epv"][nm]["on_policy"] for s in SEEDS]
            esss = [res[f"{regime}_s{s}"]["diag"]["per_cand"][nm]["ess"] for s in SEEDS]
            # DM error against the on-policy truth (the value the estimators are scored on)
            dmerr = [
                res[f"{regime}_s{s}"]["epv"][nm]["dm"]
                - res[f"{regime}_s{s}"]["epv"][nm]["on_policy"]
                for s in SEEDS
            ]
            print(
                f"  {nm:10s} {_fmt(*_mean_se(trues)):>14s} "
                f"{_fmt(*_mean_se(esss)):>14s} {_fmt(*_mean_se(dmerr)):>16s}"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_component_args(parser)
    args = parser.parse_args()
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
