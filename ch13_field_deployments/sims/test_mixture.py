# Teeth tests for the historical-mixture logging head. The load-bearing claim is the
# PROPENSITY DECISION: the logged pscore is the realized component's epsilon-greedy
# propensity (stratified sampling), not the per-step mixture marginal. The micro-oracle
# below shows realized-component pscores give unbiased trajectory IS while marginal
# pscores are measurably biased on the same log.
import numpy as np
import pytest

from promo_env import PromoConfig, N_ACTIONS
from pipeline import behavior_head


CFG = PromoConfig(K=8)


def _obs(t_frac, n=1, seed=0):
    rng = np.random.default_rng(seed)
    C = np.clip(rng.normal(size=(n, CFG.K)), -CFG.ctx_clip, CFG.ctx_clip)
    r = np.full(n, CFG.r_init)
    t = np.full(n, t_frac)
    return np.concatenate([C, r[:, None], t[:, None]], axis=1).astype(np.float32)


def test_component_redraw_only_at_episode_start():
    head = behavior_head(CFG, "mixture", seed=0, mix_epsilon=0.3)
    T = CFG.horizon
    for ep in range(5):
        for t in range(T):
            head.sample_action_and_output_pscore(_obs(t / T, seed=ep * T + t))
    assert len(head.component_trace) == 5, "exactly one component draw per episode"


def test_boundary_mismatch_raises():
    head = behavior_head(CFG, "mixture", seed=0, mix_epsilon=0.3)
    head.sample_action_and_output_pscore(_obs(0.0))  # episode start, ok
    with pytest.raises(RuntimeError):
        # a start-of-episode observation mid-episode contradicts the step counter
        head.sample_action_and_output_pscore(_obs(0.0))


def test_first_call_must_be_episode_start():
    head = behavior_head(CFG, "mixture", seed=0, mix_epsilon=0.3)
    with pytest.raises(RuntimeError):
        head.sample_action_and_output_pscore(_obs(0.5))


def test_marginal_distribution_normalized():
    head = behavior_head(CFG, "mixture", seed=0, mix_epsilon=0.3)
    x = np.concatenate([_obs(0.2, seed=k) for k in range(50)])
    probs = head.calc_action_choice_probability(x)
    assert probs.shape == (50, N_ACTIONS)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.min() >= 0.3 / N_ACTIONS - 1e-12


def test_stratified_is_unbiased_marginal_biased():
    """Micro-oracle on a 2-step, 2-action toy. Behavior: per-EPISODE component draw
    between two epsilon-greedy policies (toward action 0 / toward action 1). Target:
    the soft policy toward action 1. Reward: 1 if a==1 else 0, undiscounted, so the
    true target value is 2*pi_t(1). Trajectory IS with the REALIZED component's
    pscore must recover it; the same estimator fed the per-step MARGINAL pscore is
    biased because within-episode actions are correlated through the component."""
    rng = np.random.default_rng(0)
    eps = 0.4
    n_act, T, n_ep = 2, 2, 200_000
    p_hi = 1 - eps + eps / n_act  # prob of the component's preferred action
    p_lo = eps / n_act

    ks = rng.integers(
        2, size=n_ep
    )  # component per episode (0: prefers 0, 1: prefers 1)
    prefer = np.repeat(ks, T).reshape(n_ep, T)
    u = rng.random((n_ep, T))
    a = np.where(u < p_hi, prefer, 1 - prefer)  # epsilon-greedy around preference
    reward = (a == 1).astype(float)

    # target policy: prefers action 1 with the same epsilon
    pi_t = np.where(a == 1, p_hi, p_lo)
    # realized-component pscore (what the serving system logs)
    ps_real = np.where(a == prefer, p_hi, p_lo)
    # per-step mixture marginal: both actions equally likely a priori
    ps_marg = np.full_like(pi_t, 0.5)

    w_real = (pi_t / ps_real).reshape(n_ep, T).prod(axis=1)
    w_marg = (pi_t / ps_marg).reshape(n_ep, T).prod(axis=1)
    ret = reward.reshape(n_ep, T).sum(axis=1)

    true_value = 2 * p_hi  # E[sum r] under the target policy
    est_real = float(np.mean(w_real * ret))
    est_marg = float(np.mean(w_marg * ret))
    se_real = float(np.std(w_real * ret, ddof=1) / np.sqrt(n_ep))

    assert abs(est_real - true_value) < 4 * se_real, (
        f"realized-component IS biased: {est_real:.4f} vs true {true_value:.4f}"
    )
    assert abs(est_marg - true_value) > 10 * se_real, (
        f"marginal-pscore IS should be measurably biased, got {est_marg:.4f} "
        f"vs true {true_value:.4f}"
    )
