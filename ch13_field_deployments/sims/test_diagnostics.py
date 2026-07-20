# Unit tests for the OPE diagnostics + alert layer. These prove the alerts have teeth:
# the FQE-untrained bug that voided the first run MUST trip FQE_UNTRAINED, and the
# curse-of-horizon IS collapse MUST trip IS_DEGENERATE. Pure (no scope_rl), fast.
import numpy as np
import ope_diagnostics as d


# --- primitives --------------------------------------------------------------
def test_ess_bounds():
    assert d.effective_sample_size([1, 1, 1, 1]) == 4.0  # equal weights -> N
    assert d.effective_sample_size([1, 0, 0, 0]) == 1.0  # one survivor -> 1
    assert d.effective_sample_size([0, 0, 0]) == 0.0  # all dead -> 0
    # one huge weight dominates -> ESS near 1
    assert d.effective_sample_size([1e6, 1, 1, 1]) < 1.01


def test_coverage_detects_hole():
    cov = d.coverage_stats([0, 0, 1, 1, 2, 2], n_actions=4)  # action 3 never played
    assert cov["support_holes"] == [3]
    assert cov["min_frac"] == 0.0
    full = d.coverage_stats([0, 1, 2, 3], n_actions=4)
    assert full["support_holes"] == [] and full["entropy"] > 0


def test_eval_action_prob_epsilon_greedy():
    la = np.array([0, 1, 2])
    greedy = np.array([0, 0, 0])  # greedy action is 0 everywhere
    p = d.eval_action_prob(la, greedy, epsilon=0.3, n_actions=6)
    assert abs(p[0] - (0.7 + 0.3 / 6)) < 1e-9  # matched greedy
    assert abs(p[1] - (0.3 / 6)) < 1e-9  # non-greedy
    # deterministic (eps=0): non-greedy gets probability 0 -> IS weight will be 0
    p0 = d.eval_action_prob(la, greedy, epsilon=0.0, n_actions=6)
    assert p0[0] == 1.0 and p0[1] == 0.0


def test_trajectory_is_weights_degenerate_for_deterministic_target():
    # 3 trajectories x horizon 4, uniform behavior over 6 actions (pi_b = 1/6).
    n_traj, H, N = 3, 4, 6
    pscore = np.full(n_traj * H, 1.0 / N)
    logged = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    )  # traj0 all-0, traj1 has a 1, traj2 all-2
    greedy = np.zeros(n_traj * H, dtype=int)  # deterministic target: always action 0
    pi_e = d.eval_action_prob(logged, greedy, epsilon=0.0, n_actions=N)
    w = d.trajectory_is_weights(pi_e, pscore, n_traj, H)
    # only traj0 (all greedy) survives; traj1 (a 1) and traj2 (2s) get weight 0
    assert w[0] > 0 and w[1] == 0 and w[2] == 0
    assert d.effective_sample_size(w) == 1.0  # exactly one survivor


# --- alert bank --------------------------------------------------------------
def test_alertbank_fatal_routing_and_banner():
    b = d.AlertBank()
    assert not b.has_fatal and "none fired" in b.banner()
    b.add("IS_DEGENERATE", "ab/s0", "ess low")
    assert not b.has_fatal  # informational only
    b.add("FQE_UNTRAINED", "ab/s0", "q~0")
    assert b.has_fatal and "INVALID" in b.banner()
    assert len(b.fired("FQE_UNTRAINED")) == 1


# --- the meta-tests: the alerts catch the real failure modes -----------------
def test_fqe_untrained_tripwire():
    # d3rlpy trains n_steps // 10000 epochs; the shipped 1500 was zero epochs.
    assert d.fqe_expected_epochs(1500) == 0
    assert d.fqe_expected_epochs(8000) == 0
    assert d.fqe_expected_epochs(20000) == 2


def test_fqe_health_fires_when_untrained():
    # untrained FQE: DM estimates ~0 while true values are ~3-4 -> FATAL FQE_UNTRAINED.
    b = d.AlertBank()
    dm = {"a": 0.02, "b": -0.01, "c": 0.03}
    true = {"a": 4.2, "b": 3.5, "c": 3.9}
    d.check_fqe_health(dm, true, "ab/s0", b)
    assert b.has_fatal and len(b.fired("FQE_UNTRAINED")) == 1
    # trained FQE at the right scale -> no alert.
    b2 = d.AlertBank()
    d.check_fqe_health({"a": 4.0, "b": 3.6, "c": 3.8}, true, "ab/s0", b2)
    assert not b2.has_fatal


def test_is_degeneracy_alert():
    b = d.AlertBank()
    d.check_is_degeneracy(
        {"det": 0.0, "soft": 2.0, "match": 500.0},
        {"det": 0, "soft": 3, "match": 500},
        "ab/s0",
        b,
    )
    kinds = {a["kind"] for a in b.fired()}
    assert "IS_DEGENERATE" in kinds
    flagged = {
        a["where"] for a in b.fired("IS_DEGENERATE")
    }  # all under threshold flagged
    assert (
        len(b.fired("IS_DEGENERATE")) == 2
    )  # det and soft below ess_min=5; match above


def test_truth_mismatch_alert():
    b = d.AlertBank()
    # the scope-rl discount off-by-one produced a ~5% gap; the tightened 5% tol
    # must catch anything that large (the old 15% tol silently hid it)
    d.check_truth({"a": 4.0}, {"a": 4.0 * 0.95 * 0.99}, "ab/s0", b)
    assert b.has_fatal and len(b.fired("TRUTH_MISMATCH")) == 1
    b2 = d.AlertBank()
    d.check_truth({"a": 4.0}, {"a": 4.0 * 1.03}, "ab/s0", b2)  # 3% within tol
    assert not b2.has_fatal


def test_rank_inversion_alert():
    b = d.AlertBank()
    d.check_rank_inversion({"dm": -0.94, "pdis": 0.1, "dr": None}, "ab/s0", b)
    assert len(b.fired("RANK_INVERSION")) == 1  # only dm below -0.5; None ignored
    assert not b.has_fatal  # informational


def test_dm_calibration_alert():
    # per-cell: error 0.5 (past the 0.20 catastrophic gate) -> FATAL
    b = d.AlertBank()
    m = d.check_dm_calibration(
        {"uniform": 3.0, "x": 9.9}, {"uniform": 3.5, "x": 0.0}, "uniform", "ab/s0", b
    )
    assert b.has_fatal and len(b.fired("FQE_MISCALIBRATED")) == 1
    assert abs(m["err"] + 0.5) < 1e-9
    # error 0.15 sits in the warn band (refit noise scale) -> INFO only
    b2 = d.AlertBank()
    d.check_dm_calibration(
        {"uniform": 3.35, "x": 9.9}, {"uniform": 3.5, "x": 0.0}, "uniform", "ab/s1", b2
    )
    assert not b2.has_fatal and len(b2.fired("DM_CAL_WARN")) == 1
    # error 0.05 -> silent
    b3 = d.AlertBank()
    d.check_dm_calibration(
        {"uniform": 3.45, "x": 9.9}, {"uniform": 3.5, "x": 0.0}, "uniform", "ab/s2", b3
    )
    assert not b3.has_fatal and not b3.fired()
    # non-fatal mode (mixture regime): catastrophic error surfaces as INFO
    b4 = d.AlertBank()
    d.check_dm_calibration(
        {"uniform": 3.0}, {"uniform": 3.5}, "uniform", "mix/s0", b4, fatal=False
    )
    assert not b4.has_fatal and len(b4.fired("DM_CAL_WARN")) == 1


def test_dm_calibration_bias_alert():
    # mean-zero noise at the observed sd -> silent
    b = d.AlertBank()
    d.check_dm_calibration_bias(
        [-0.05, -0.13, 0.09, 0.02, 0.11, 0.04, 0.03, 0.03, 0.04, -0.10],
        "incumbent",
        "myopic",
        b,
    )
    assert not b.has_fatal
    # the greedy-TD-target bug's signature (pooled +0.26) -> FATAL
    b2 = d.AlertBank()
    d.check_dm_calibration_bias([0.26] * 10, "ab", "uniform", b2)
    assert b2.has_fatal and len(b2.fired("FQE_MISCALIBRATED")) == 1


def test_clone_fidelity_alert():
    b = d.AlertBank()
    d.check_clone_fidelity({"myopic": 0.71, "no_promo": 1.0}, "ab/s0", b)
    assert b.has_fatal and len(b.fired("CLONE_INFIDELITY")) == 1
    b2 = d.AlertBank()
    d.check_clone_fidelity({"myopic": 0.99, "no_promo": 1.0}, "ab/s0", b2)
    assert not b2.has_fatal


def test_mixture_log_alert():
    # all 7 components drawn, pscores at/above the floor -> silent
    b = d.AlertBank()
    d.check_mixture_log([40, 42, 39, 45, 41, 44, 49], 7, 0.05, 0.3, 6, "mix/s0", b)
    assert not b.has_fatal
    # a component never drawn -> FATAL
    b2 = d.AlertBank()
    d.check_mixture_log([40, 0, 39, 45, 41, 44, 49], 7, 0.05, 0.3, 6, "mix/s0", b2)
    assert b2.has_fatal
    # a pscore below the analytic floor eps/N -> FATAL
    b3 = d.AlertBank()
    d.check_mixture_log([40, 42, 39, 45, 41, 44, 49], 7, 0.01, 0.3, 6, "mix/s0", b3)
    assert b3.has_fatal
