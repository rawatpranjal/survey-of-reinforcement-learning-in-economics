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
    d.check_truth({"a": 4.0}, {"a": 4.0 * 1.30}, "ab/s0", b)  # 30% gap > 15% tol
    assert b.has_fatal and len(b.fired("TRUTH_MISMATCH")) == 1
    b2 = d.AlertBank()
    d.check_truth({"a": 4.0}, {"a": 4.0 * 1.05}, "ab/s0", b2)  # 5% within tol
    assert not b2.has_fatal


def test_rank_inversion_alert():
    b = d.AlertBank()
    d.check_rank_inversion({"dm": -0.94, "pdis": 0.1, "dr": None}, "ab/s0", b)
    assert len(b.fired("RANK_INVERSION")) == 1  # only dm below -0.5; None ignored
    assert not b.has_fatal  # informational
