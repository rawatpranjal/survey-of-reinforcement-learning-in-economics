"""The MBPOPolicy class must implement actual branched rollouts under a
learned dynamics ensemble. The previous code in this slot was a closed-form
Riccati planner on point estimates — that gets renamed to ParametricLQLearner.
"""

import numpy as np
import pytest

import cobweb_paradigms as cp


def test_parametric_lq_learner_exists():
    """The closed-form LQ planner survives under its honest name."""
    assert hasattr(cp, 'ParametricLQLearner'), (
        "Renamed class ParametricLQLearner should exist in cobweb_paradigms."
    )
    inst = cp.ParametricLQLearner(gamma=0.95, explore_std=0.1, warmup=5)
    assert inst.name == 'Model-Based LQ', (
        f"name should be 'Model-Based LQ'; got {inst.name!r}"
    )


def test_real_mbpo_has_learnable_policy_and_ensemble():
    assert hasattr(cp, 'MBPOPolicy'), "MBPOPolicy must exist."
    mbpo = cp.MBPOPolicy(
        gamma=0.95, explore_std=0.1, warmup=5,
        ensemble_size=5, rollout_horizon=5, n_rollouts=10,
    )
    rp = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1)
    mbpo.reset(rp, seed=0)

    assert hasattr(mbpo, 'K0') and hasattr(mbpo, 'Kq'), (
        "Real MBPO must use a parameterized linear policy "
        "q = K0 + Kq * q_prev."
    )
    assert hasattr(mbpo, 'ensemble'), (
        "Real MBPO must maintain an ensemble of dynamics models."
    )
    # Step the agent a few times so the buffer fills and the ensemble is fit.
    env = cp.CobwebEnv(
        a=rp['a'], b=rp['b'], c=rp['c'], phi=rp['phi'],
        sigma=rp['sigma'], gamma=0.95, T=20, seed=0,
    )
    state = env.reset()
    for t in range(15):
        action = mbpo.act(state, t)
        next_state, reward, done, _ = env.step(action)
        mbpo.observe(state, action, reward, next_state)
        state = next_state
    assert len(mbpo.ensemble) >= 2, (
        f"Ensemble must have >= 2 members after warmup; got "
        f"{len(mbpo.ensemble)}"
    )


def test_real_mbpo_name():
    """Display name is 'MB-LG-REINFORCE' (simplified MBPO variant). Class
    name MBPOPolicy retained for backwards-compatible cache/test wiring; the
    label distinguishes this implementation from the SAC-based MBPO of
    Janner et al. 2019, since it uses linear-Gaussian dynamics and a
    two-parameter REINFORCE policy rather than neural dynamics and SAC."""
    mbpo = cp.MBPOPolicy(
        gamma=0.95, explore_std=0.1, warmup=5,
        ensemble_size=5, rollout_horizon=5, n_rollouts=10,
    )
    assert mbpo.name == 'MB-LG-REINFORCE', (
        f"name should be 'MB-LG-REINFORCE'; got {mbpo.name!r}"
    )


def test_real_mbpo_policy_moves_with_training():
    """After 100 steps, policy params should have moved from initialization."""
    mbpo = cp.MBPOPolicy(
        gamma=0.95, explore_std=0.1, warmup=5,
        ensemble_size=5, rollout_horizon=5, n_rollouts=10,
    )
    rp = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1)
    mbpo.reset(rp, seed=0)
    env = cp.CobwebEnv(
        a=rp['a'], b=rp['b'], c=rp['c'], phi=rp['phi'],
        sigma=rp['sigma'], gamma=0.95, T=100, seed=0,
    )
    state = env.reset()
    K0_init = float(mbpo.K0)
    Kq_init = float(mbpo.Kq)
    for t in range(100):
        action = mbpo.act(state, t)
        next_state, reward, done, _ = env.step(action)
        mbpo.observe(state, action, reward, next_state)
        state = next_state
    K0_final = float(mbpo.K0)
    Kq_final = float(mbpo.Kq)
    moved = abs(K0_final - K0_init) > 1e-4 or abs(Kq_final - Kq_init) > 1e-4
    assert moved, (
        f"Policy params did not move (K0: {K0_init:.5f} → {K0_final:.5f}, "
        f"Kq: {Kq_init:.5f} → {Kq_final:.5f}) — REINFORCE update may not "
        "be wired up to the rollout returns."
    )
