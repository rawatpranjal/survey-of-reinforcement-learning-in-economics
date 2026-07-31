"""MBPathwisePolicy guardrails.

1. No-leak: the pathwise learner must never read the true regime parameters
   or call the oracle solver. Enforced by monkey-patching solve_oracle_lq and
   expected_reward to raise on any call from its code path (same pattern as
   test_cobweb_ga_no_param_leak.py).
2. Gradient identity: the forward-sensitivity recursion in _pathwise_rollout
   must equal a central finite difference of its own total_return in (K0, Kq).
"""

import numpy as np
import pytest

import cobweb_paradigms as cp

RP = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1)


def _make_policy(seed=0):
    pol = cp.MBPathwisePolicy(gamma=0.95, explore_std=0.15, warmup=5)
    pol.reset(RP, seed=seed)
    return pol


def test_pathwise_never_calls_oracle_or_expected_reward(monkeypatch):
    calls = []

    def trap(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("oracle/true-param helper called from pathwise path — leak")

    monkeypatch.setattr(cp, "solve_oracle_lq", trap)
    monkeypatch.setattr(cp, "expected_reward", trap)

    pol = _make_policy()
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
    for t in range(30):  # past warmup: ensemble fits + policy updates run
        action = pol.act(state, t)
        next_state, reward, done, _ = env.step(action)
        pol.observe(state, action, reward, next_state)
        state = next_state

    assert not calls, f"pathwise path called a true-param helper {len(calls)} times"
    # The updates actually ran (path exercised, not vacuously clean).
    assert len(pol.ensemble) == pol.ensemble_size
    assert (pol.K0, pol.Kq) != (0.0, 0.0)


def test_pathwise_reset_does_not_store_true_params():
    weird = dict(a=999.0, b=888.0, c=777.0, phi=666.0, sigma=0.1)
    pol = cp.MBPathwisePolicy(gamma=0.95, explore_std=0.15, warmup=5)
    pol.reset(weird, seed=0)
    stored = [v for v in pol.__dict__.values() if isinstance(v, float)]
    for sentinel in (999.0, 888.0, 777.0, 666.0):
        assert sentinel not in stored, f"true param {sentinel} stored on the policy"


def test_pathwise_gradient_matches_finite_difference():
    pol = _make_policy(seed=3)
    rng = np.random.default_rng(7)
    # Synthetic replay buffer from the true model (test-side only; the policy
    # sees the tuples, not the parameters).
    q_prev = 1.0
    for _ in range(12):
        q = float(rng.uniform(0.5, 3.0))
        p = RP["a"] - RP["b"] * q + float(rng.normal(0, RP["sigma"]))
        r = p * q - 0.5 * RP["c"] * q**2 - 0.5 * RP["phi"] * (q - q_prev) ** 2
        pol.buffer.append((q_prev, q, p, r))
        q_prev = q
    pol._fit_ensemble()
    assert len(pol.ensemble) == pol.ensemble_size

    pol.K0, pol.Kq = 0.4, 0.25
    q0 = 1.3
    eps = 1e-6
    for member in pol.ensemble[:3]:
        g0, gq, _ = pol._pathwise_rollout(q0, member)

        pol.K0 += eps
        _, _, j_up = pol._pathwise_rollout(q0, member)
        pol.K0 -= 2 * eps
        _, _, j_dn = pol._pathwise_rollout(q0, member)
        pol.K0 += eps
        fd0 = (j_up - j_dn) / (2 * eps)

        pol.Kq += eps
        _, _, j_up = pol._pathwise_rollout(q0, member)
        pol.Kq -= 2 * eps
        _, _, j_dn = pol._pathwise_rollout(q0, member)
        pol.Kq += eps
        fdq = (j_up - j_dn) / (2 * eps)

        assert g0 == pytest.approx(fd0, rel=1e-5, abs=1e-7), (
            f"dJ/dK0 analytic {g0} vs FD {fd0}"
        )
        assert gq == pytest.approx(fdq, rel=1e-5, abs=1e-7), (
            f"dJ/dKq analytic {gq} vs FD {fdq}"
        )
