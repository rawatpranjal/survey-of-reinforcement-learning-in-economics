"""Arifovic GA must not read true (a, b, c, phi).

The original Arifovic election operator compared offspring to parents using
expected_reward(...; a, b, c, phi) — the true demand/cost parameters. The
audit's claim that the GA has "no parametric knowledge" requires that
selection comes only from realized observed profit. We enforce this by
monkey-patching expected_reward to raise on any call from GA code paths.
"""

import numpy as np
import pytest

import cobweb_paradigms as cp


def test_ga_evolve_does_not_call_expected_reward(monkeypatch):
    calls = []

    def trap(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("expected_reward called from GA code path — leak")

    monkeypatch.setattr(cp, 'expected_reward', trap)

    ga = cp.ArifovicGAPolicy(
        n_pop=10, L_bits=8, p_cross=0.6, p_mut=0.05,
        gen_len=5, q_min=0.0, q_max=4.0,
    )
    rp = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.1)
    ga.reset(rp, seed=0)
    env = cp.CobwebEnv(
        a=rp['a'], b=rp['b'], c=rp['c'], phi=rp['phi'],
        sigma=rp['sigma'], gamma=0.95, T=20, seed=0,
    )
    state = env.reset()
    for t in range(15):  # >= gen_len triggers _evolve at least once
        action = ga.act(state, t)
        next_state, reward, done, _ = env.step(action)
        ga.observe(state, action, reward, next_state)
        state = next_state

    assert not calls, (
        f"GA called expected_reward {len(calls)} times — true demand/cost "
        "parameters were used in selection. Election operator must be removed."
    )


def test_ga_fitness_equals_running_mean_of_realized_rewards():
    """With no env noise and gen_len long enough to skip evolution, each
    chromosome's fitness should equal the mean of the rewards it actually
    received when played — no hypothetical scoring."""
    ga = cp.ArifovicGAPolicy(
        n_pop=4, L_bits=6, p_cross=0.6, p_mut=0.0,
        gen_len=100, q_min=0.0, q_max=4.0,
    )
    rp = dict(a=4.0, b=0.5, c=1.0, phi=0.2, sigma=0.0)
    ga.reset(rp, seed=0)
    env = cp.CobwebEnv(
        a=rp['a'], b=rp['b'], c=rp['c'], phi=rp['phi'],
        sigma=0.0, gamma=0.95, T=20, seed=0,
    )
    state = env.reset()
    rewards_per_chrom = [[] for _ in range(4)]
    for t in range(8):  # 8 < gen_len so no evolve yet
        action = ga.act(state, t)
        idx_played = ga._last_active
        next_state, reward, done, _ = env.step(action)
        ga.observe(state, action, reward, next_state)
        rewards_per_chrom[idx_played].append(reward)
        state = next_state

    for i in range(4):
        if rewards_per_chrom[i]:
            expected = float(np.mean(rewards_per_chrom[i]))
            assert abs(ga.fitness[i] - expected) < 1e-9, (
                f"fitness[{i}]={ga.fitness[i]:.6f} != mean realized "
                f"profit {expected:.6f}"
            )


def test_ga_constructor_does_not_accept_election_kwarg():
    """After the fix, the election operator is removed entirely — the kwarg
    that toggled it should also be gone, so accidental re-introduction fails
    loudly."""
    with pytest.raises(TypeError):
        cp.ArifovicGAPolicy(
            n_pop=10, L_bits=8, p_cross=0.6, p_mut=0.05,
            gen_len=5, election=True,
        )
