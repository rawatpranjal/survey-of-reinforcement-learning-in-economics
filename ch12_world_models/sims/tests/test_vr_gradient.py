"""MBVRPolicy (MB-LG-VR) guardrails.

1. No-leak: like its two siblings, the VR learner must never read true regime
   parameters or call the oracle solver.
2. Antithetic mechanics: _noise_rollout consumes the supplied noise exactly
   (horizon-1 hand check) and a mirrored pair produces actions symmetric
   around the policy mean.
3. Baseline no-lookahead: the first update call sees a zero baseline; the
   quadratic fit only appears after enough PAST points accumulate.
4. Variance: with the policy held fixed, the VR gradient estimate has strictly
   lower variance than the plain REINFORCE estimate at the same rollout budget.
"""

import copy

import numpy as np
import pytest

import cobweb_paradigms as cp

RP = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1)


def _fitted_policy(cls, seed=0, **kwargs):
    pol = cls(gamma=0.95, explore_std=0.15, warmup=5, **kwargs)
    pol.reset(RP, seed=seed)
    rng = np.random.default_rng(11)
    q_prev = 1.0
    for _ in range(12):
        q = float(rng.uniform(0.5, 3.0))
        p = RP["a"] - RP["b"] * q + float(rng.normal(0, RP["sigma"]))
        r = p * q - 0.5 * RP["c"] * q**2 - 0.5 * RP["phi"] * (q - q_prev) ** 2
        pol.buffer.append((q_prev, q, p, r))
        q_prev = q
    pol._fit_ensemble()
    pol.K0, pol.Kq = 0.4, 0.25
    return pol


def test_vr_never_calls_oracle_or_expected_reward(monkeypatch):
    calls = []

    def trap(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("true-param helper called from VR path — leak")

    monkeypatch.setattr(cp, "solve_oracle_lq", trap)
    monkeypatch.setattr(cp, "expected_reward", trap)

    pol = cp.MBVRPolicy(gamma=0.95, explore_std=0.15, warmup=5)
    pol.reset(RP, seed=0)
    env = cp.CobwebEnv(
        a=RP["a"],
        b=RP["b"],
        c=RP["c"],
        phi=RP["phi"],
        sigma=RP["sigma"],
        gamma=0.95,
        T=30,
        seed=0,
    )
    state = env.reset()
    for t in range(30):
        action = pol.act(state, t)
        next_state, reward, done, _ = env.step(action)
        pol.observe(state, action, reward, next_state)
        state = next_state

    assert not calls
    assert len(pol.ensemble) == pol.ensemble_size
    assert (pol.K0, pol.Kq) != (0.0, 0.0)


def test_noise_rollout_horizon_one_hand_check():
    pol = _fitted_policy(cp.MBVRPolicy)
    pol.rollout_horizon = 1
    member = {"a_hat": 3.9, "b_hat": 0.52, "sigma": 0.1}
    q0, ea, ep = 1.3, 0.07, -0.04
    g0, gq, ret = pol._noise_rollout(q0, member, np.array([ea]), np.array([ep]))
    mean = pol.K0 + pol.Kq * q0
    a = float(np.clip(mean + ea, pol.q_min, pol.q_max))
    p = member["a_hat"] - member["b_hat"] * a + ep
    r = p * a - 0.5 * pol.c_hat * a**2 - 0.5 * pol.phi_hat * (a - q0) ** 2
    score = ea / pol.explore_std**2
    assert ret == pytest.approx(r)
    assert g0 == pytest.approx(score)
    assert gq == pytest.approx(score * q0)


def test_antithetic_pair_is_mirrored_around_mean():
    pol = _fitted_policy(cp.MBVRPolicy)
    member = pol.ensemble[0]
    H = pol.rollout_horizon
    eps_a = np.full(H, 0.09)
    eps_p = np.zeros(H)
    g0_plus, _, _ = pol._noise_rollout(1.3, member, eps_a, eps_p)
    g0_minus, _, _ = pol._noise_rollout(1.3, member, -eps_a, -eps_p)
    # Score at each step is eps/std^2 regardless of the state path, so the
    # mirrored twin's summed score is the exact negative.
    assert g0_plus == pytest.approx(-g0_minus)


def test_baseline_no_lookahead():
    pol = _fitted_policy(cp.MBVRPolicy)
    assert pol.baseline_pts == []
    assert pol._baseline_value(1.0) == 0.0  # first batch faces zero baseline
    pol._update_policy()
    assert len(pol.baseline_pts) == 2 * (pol.n_rollouts // 2)
    pol._update_policy()  # 20 points >= 10 -> quadratic fit exists
    assert pol.baseline_w is not None


def test_vr_gradient_variance_below_reinforce():
    template_vr = _fitted_policy(cp.MBVRPolicy)
    template_rf = _fitted_policy(cp.MBPOPolicy)
    # Same fitted ensemble and buffer in both templates (same seed recipe).
    draws_vr, draws_rf = [], []
    for i in range(300):
        pv = copy.deepcopy(template_vr)
        pv.rng = np.random.default_rng(1000 + i)
        K0_before = pv.K0
        pv._update_policy()
        draws_vr.append((pv.K0 - K0_before) / pv.policy_lr)

        pr = copy.deepcopy(template_rf)
        pr.rng = np.random.default_rng(1000 + i)
        pr.baseline = 0.0  # same zero-baseline starting point as VR's first call
        K0_before = pr.K0
        pr._update_policy()
        draws_rf.append((pr.K0 - K0_before) / pr.policy_lr)

    var_vr = float(np.var(draws_vr))
    var_rf = float(np.var(draws_rf))
    assert var_vr < var_rf, f"VR variance {var_vr:.4f} not below REINFORCE {var_rf:.4f}"
