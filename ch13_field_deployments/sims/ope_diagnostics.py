# Diagnostics + alerts for the OPE-reliability study. Every pipeline step emits metrics
# and a health verdict so a silent failure (e.g. the FQE-trained-0-epochs bug) cannot pass
# unnoticed. Pure functions + a small AlertBank; no scope_rl/d3rlpy imports here so the
# primitives are unit-testable in isolation.
#
# Alert severities:
#   FATAL  -> the run's numbers are not trustworthy; the driver stamps the run INVALID.
#   INFO   -> an expected scientific finding (IS degeneracy, rank inversion), surfaced loud
#             but not fatal.
import numpy as np

# d3rlpy trains FQE for (n_steps // n_steps_per_epoch) epochs; the default epoch length is
# 10000, so any fqe_steps below it trains for ZERO gradient steps. This constant is the
# tripwire for that bug.
FQE_EPOCH_LEN = 10000

# Thresholds (one place; imported by the driver and the tests).
THRESH = {
    "fqe_scale_frac": 0.25,  # DM(best) must be >= this * true(best), else FQE_UNTRAINED
    # tightened from 0.15 once the scope-rl discount off-by-one (on-policy value = gamma *
    # standard-convention value) is corrected in the pipeline; the loose tolerance was
    # exactly wide enough to hide that bug.
    "truth_rel_tol": 0.05,  # |MC_true - scoperl_onpolicy| / true > this -> TRUTH_MISMATCH
    "ess_min": 5.0,  # trajectory-IS effective sample size below this -> IS_DEGENERATE
    "rank_inv": -0.5,  # rank correlation below this -> RANK_INVERSION (informational)
    # calibration gates on the behavior-policy candidate. Measured FQE refit noise at
    # the frozen budget: cal-error sd 0.04 (ab, exact behavior candidate) to 0.07
    # (incumbent/mixture, clone or on-support candidate), all means within 1.5 SE of
    # zero over 10 seeds. Per-cell FATAL therefore sits at ~3 sd (catastrophic fits
    # only) and the bias gate on the SEED-POOLED mean is what catches a systematic
    # miscalibration (the greedy-TD-target bug produced +0.26 pooled, 5x this gate).
    "dm_cal_warn": 0.10,  # per-cell |DM - truth| above this -> INFO DM_CAL_WARN
    "dm_cal_fatal": 0.20,  # per-cell |DM - truth| above this -> FATAL FQE_MISCALIBRATED
    "dm_cal_bias": 0.05,  # |mean cal error over seeds| above this -> FATAL FQE_MISCALIBRATED
    # BC clone argmax match rate below this -> CLONE_INFIDELITY. 0.90, not higher: the
    # residual disagreements of a converged myopic clone sit on argmax near-ties (the
    # rule's action boundary), and no bias enters the results table because the true
    # value is computed for the clone head itself; the gate protects the row LABEL.
    "clone_fidelity_min": 0.90,
}


# ---------------------------------------------------------------------------
# Alert bank
# ---------------------------------------------------------------------------
class AlertBank:
    """Collects alerts across a run. FATAL alerts make the run INVALID."""

    FATAL_KINDS = {
        "FQE_UNTRAINED",
        "TRUTH_MISMATCH",
        "SUPPORT_HOLE",
        "FQE_MISCALIBRATED",
        "CLONE_INFIDELITY",
    }

    def __init__(self):
        self.alerts = []  # list of dicts: {kind, severity, where, detail}

    def add(self, kind, where, detail):
        severity = "FATAL" if kind in self.FATAL_KINDS else "INFO"
        self.alerts.append(
            {"kind": kind, "severity": severity, "where": where, "detail": detail}
        )

    def fired(self, kind=None):
        if kind is None:
            return list(self.alerts)
        return [a for a in self.alerts if a["kind"] == kind]

    @property
    def has_fatal(self):
        return any(a["severity"] == "FATAL" for a in self.alerts)

    def banner(self):
        """Top-of-output ALERTS FIRED banner."""
        if not self.alerts:
            return "ALERTS: none fired. All step health checks passed."
        lines = ["=" * 72, "ALERTS FIRED", "=" * 72]
        # fatal first, then info; grouped by kind with counts
        for sev in ("FATAL", "INFO"):
            kinds = sorted({a["kind"] for a in self.alerts if a["severity"] == sev})
            for kind in kinds:
                hits = [a for a in self.alerts if a["kind"] == kind]
                lines.append(f"[{sev}] {kind}  x{len(hits)}")
                for a in hits[:12]:
                    lines.append(f"    {a['where']}: {a['detail']}")
                if len(hits) > 12:
                    lines.append(f"    ... (+{len(hits) - 12} more)")
        if self.has_fatal:
            lines.append("-" * 72)
            lines.append(
                "RUN STAMPED INVALID: a FATAL alert fired. Numbers not trustworthy."
            )
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric primitives (pure)
# ---------------------------------------------------------------------------
def effective_sample_size(weights):
    """ESS = (sum w)^2 / sum(w^2). Returns 0.0 if all weights are zero."""
    w = np.asarray(weights, dtype=float)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / s2)


def eval_action_prob(logged_actions, greedy_actions, epsilon, n_actions):
    """Pi_e(logged_action | state) for an epsilon-greedy head:
    (1-eps) + eps/N if the logged action is the greedy action, else eps/N."""
    la = np.asarray(logged_actions)
    ga = np.asarray(greedy_actions)
    match = la == ga
    return np.where(match, (1.0 - epsilon) + epsilon / n_actions, epsilon / n_actions)


def trajectory_is_weights(pi_e, pi_b, n_traj, horizon):
    """Per-trajectory cumulative importance weight prod_t pi_e(a_t)/pi_b(a_t).
    pi_e, pi_b are flat arrays of length n_traj*horizon, trajectory-major."""
    ratio = (np.asarray(pi_e, dtype=float) / np.asarray(pi_b, dtype=float)).reshape(
        n_traj, horizon
    )
    return np.prod(ratio, axis=1)


def coverage_stats(logged_actions, n_actions):
    """Action-coverage of a logged dataset: per-action counts, entropy, support holes."""
    la = np.asarray(logged_actions).astype(int)
    counts = np.bincount(la, minlength=n_actions)
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
    holes = [int(a) for a in range(n_actions) if counts[a] == 0]
    return {
        "counts": counts.tolist(),
        "entropy": entropy,
        "support_holes": holes,
        "min_frac": float(p.min()),
    }


# ---------------------------------------------------------------------------
# Step health checks (compute metrics + register alerts)
# ---------------------------------------------------------------------------
def check_fqe_health(dm_by_cand, true_by_cand, where, bank):
    """FATAL FQE_UNTRAINED if the DM estimate scale is far below the true-value scale
    (the tell for a value function trained for ~0 gradient steps)."""
    best = max(true_by_cand, key=true_by_cand.get)
    true_best = true_by_cand[best]
    dm_best_mag = max(abs(v) for v in dm_by_cand.values())
    ok = dm_best_mag >= THRESH["fqe_scale_frac"] * abs(true_best)
    metric = {"dm_max_abs": dm_best_mag, "true_best": true_best, "ok": ok}
    if not ok:
        bank.add(
            "FQE_UNTRAINED",
            where,
            f"DM scale {dm_best_mag:.3f} << true best {true_best:.3f} "
            f"(< {THRESH['fqe_scale_frac']:.2f}x); FQE likely trained ~0 steps",
        )
    return metric


def check_truth(mc_by_cand, onpolicy_by_cand, where, bank):
    """FATAL TRUTH_MISMATCH if independent MC and Scope-RL on-policy truth disagree."""
    worst = 0.0
    worst_cand = None
    for nm in mc_by_cand:
        if nm not in onpolicy_by_cand:
            continue
        mc, op = mc_by_cand[nm], onpolicy_by_cand[nm]
        denom = max(abs(mc), 1e-8)
        rel = abs(mc - op) / denom
        if rel > worst:
            worst, worst_cand = rel, nm
    ok = worst <= THRESH["truth_rel_tol"]
    if not ok:
        bank.add(
            "TRUTH_MISMATCH",
            where,
            f"MC vs Scope-RL on-policy disagree by {worst:.1%} at {worst_cand}",
        )
    return {"worst_rel_gap": worst, "worst_cand": worst_cand, "ok": ok}


def check_is_degeneracy(ess_by_cand, surviving_by_cand, where, bank):
    """INFO IS_DEGENERATE per candidate whose trajectory-IS ESS is below threshold
    (expected for deterministic targets over a long horizon: the curse of horizon)."""
    flagged = []
    for nm, ess in ess_by_cand.items():
        if ess < THRESH["ess_min"]:
            flagged.append(nm)
            bank.add(
                "IS_DEGENERATE",
                where,
                f"{nm}: ESS {ess:.2f} < {THRESH['ess_min']:.0f}, "
                f"surviving trajectories {surviving_by_cand.get(nm, '?')}",
            )
    return {"flagged": flagged}


def check_rank_inversion(rank_corr_by_est, where, bank):
    """INFO RANK_INVERSION when an estimator's rank correlation is strongly negative."""
    flagged = []
    for est, rc in rank_corr_by_est.items():
        if rc is not None and rc < THRESH["rank_inv"]:
            flagged.append(est)
            bank.add("RANK_INVERSION", where, f"{est}: rank corr {rc:.3f}")
    return {"flagged": flagged}


def check_dm_calibration(dm_by_cand, truth_by_cand, cal_cand, where, bank, fatal=True):
    """Per-cell calibration check for the fitted-Q direct method: on the calibration
    candidate (the behavior policy itself, or the closest on-support candidate),
    coverage is perfect, so any DM error is pure fit error. A single cell's error is
    dominated by refit noise, so the per-cell FATAL sits at dm_cal_fatal (~3 sd of
    measured noise) and errors in the warn band surface as INFO DM_CAL_WARN. The
    systematic-bias gate lives in check_dm_calibration_bias, on the seed-pooled mean.
    The old fqe_scale_frac check only caught Q at random init; these catch Q trained
    but still biased where it has no coverage excuse."""
    dm, truth = dm_by_cand[cal_cand], truth_by_cand[cal_cand]
    err = dm - truth
    ok = abs(err) <= THRESH["dm_cal_fatal"]
    if not ok:
        bank.add(
            "FQE_MISCALIBRATED" if fatal else "DM_CAL_WARN",
            where,
            f"{cal_cand}: DM {dm:.3f} vs truth {truth:.3f} "
            f"(err {err:+.3f}, per-cell fatal {THRESH['dm_cal_fatal']:.2f})",
        )
    elif abs(err) > THRESH["dm_cal_warn"]:
        bank.add(
            "DM_CAL_WARN",
            where,
            f"{cal_cand}: DM {dm:.3f} vs truth {truth:.3f} "
            f"(err {err:+.3f}, warn band {THRESH['dm_cal_warn']:.2f})",
        )
    return {"cal_cand": cal_cand, "err": float(err), "ok": ok}


def check_dm_calibration_bias(cal_errs, regime, cal_cand, bank, fatal=True):
    """Seed-pooled calibration-bias gate: FATAL FQE_MISCALIBRATED if the MEAN DM
    error on the calibration candidate across seeds exceeds dm_cal_bias. Refit noise
    averages out over seeds, so a pooled mean this large is systematic (the
    greedy-TD-target bug produced a pooled +0.26 here, five times this gate)."""
    errs = np.asarray(cal_errs, dtype=float)
    mean = float(errs.mean())
    ok = abs(mean) <= THRESH["dm_cal_bias"]
    if not ok:
        bank.add(
            "FQE_MISCALIBRATED" if fatal else "DM_CAL_WARN",
            f"{regime}/pooled",
            f"{cal_cand}: mean cal error {mean:+.3f} over {errs.size} seeds "
            f"(bias gate {THRESH['dm_cal_bias']:.2f})",
        )
    return {"mean_err": mean, "n": int(errs.size), "ok": ok}


def check_clone_fidelity(fidelity_by_cand, where, bank):
    """FATAL CLONE_INFIDELITY if a BC candidate clone's argmax match rate against
    its reference rule falls below threshold (an unfaithful clone silently changes
    which policy the menu contains; the old observational log's imitation collapsed
    to the max discount this way)."""
    flagged = []
    for nm, rate in fidelity_by_cand.items():
        if rate < THRESH["clone_fidelity_min"]:
            flagged.append(nm)
            bank.add(
                "CLONE_INFIDELITY",
                where,
                f"{nm}: clone argmax match rate {rate:.3f} "
                f"< {THRESH['clone_fidelity_min']:.2f}",
            )
    return {"flagged": flagged}


def check_mixture_log(
    component_counts, n_components, pscore_min, epsilon, n_actions, where, bank
):
    """FATAL SUPPORT_HOLE if some mixture component was never drawn across the log's
    episodes, or a logged propensity fell below the analytic floor eps/N (both would
    invalidate the stratified-IS argument for the mixture regime)."""
    counts = np.asarray(component_counts)
    ok = True
    if len(counts) != n_components or (counts == 0).any():
        ok = False
        bank.add(
            "SUPPORT_HOLE",
            where,
            f"mixture components drawn {counts.tolist()} (some never drawn)",
        )
    floor = epsilon / n_actions - 1e-9
    if pscore_min < floor:
        ok = False
        bank.add(
            "SUPPORT_HOLE",
            where,
            f"logged pscore {pscore_min:.6f} below analytic floor {floor:.6f}",
        )
    return {
        "component_counts": counts.tolist(),
        "pscore_min": float(pscore_min),
        "ok": ok,
    }


def check_support(cov, where, bank):
    """FATAL SUPPORT_HOLE if the logged dataset never played some action (breaks IS)."""
    if cov["support_holes"]:
        bank.add(
            "SUPPORT_HOLE",
            where,
            f"actions never logged: {cov['support_holes']} (min frac {cov['min_frac']:.4f})",
        )
    return cov


def fqe_expected_epochs(fqe_steps, epoch_len=FQE_EPOCH_LEN):
    """The number of epochs d3rlpy will actually run for a given n_steps. 0 => untrained."""
    return int(fqe_steps) // int(epoch_len)
