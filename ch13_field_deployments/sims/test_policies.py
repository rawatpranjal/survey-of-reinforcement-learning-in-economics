# Teeth tests for the batched reference policies.
import numpy as np
import pytest

from promo_env import (
    PromoEnv,
    PromoConfig,
    DISCOUNTS,
    N_ACTIONS,
    uniform_batch,
    myopic_batch,
    constant_batch,
)


@pytest.fixture
def env():
    return PromoEnv(PromoConfig(K=8), seed=0)


def _obs_batch(env, n, seed):
    rng = np.random.default_rng(seed)
    C = np.clip(rng.normal(size=(n, env.cfg.K)), -4, 4)
    r = rng.uniform(0, 0.25, size=n)
    t = rng.uniform(0, 1, size=n)
    return np.concatenate([C, r[:, None], t[:, None]], axis=1).astype(np.float32)


def test_uniform_is_uniform_and_valid(env):
    rng = np.random.default_rng(0)
    obs = _obs_batch(env, 60000, 1)
    a = uniform_batch(env, obs, rng)
    assert a.min() >= 0 and a.max() < N_ACTIONS
    counts = np.bincount(a, minlength=N_ACTIONS)
    expected = len(a) / N_ACTIONS
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 20.0, (
        f"uniform behavior not uniform: chi2={chi2:.1f}, counts={counts}"
    )


def test_myopic_matches_independent_brute_force(env):
    # myopic must pick the action maximizing immediate expected margin, recomputed
    # here via the scalar buy_prob path (independent of myopic_batch's internals).
    obs = _obs_batch(env, 300, 2)
    got = myopic_batch(env, obs)
    K = env.cfg.K
    for i in range(len(obs)):
        c, r = obs[i, :K], float(obs[i, K])
        er = [env.buy_prob(c, r, d) * float(env.margin(d)) for d in DISCOUNTS]
        assert got[i] == int(np.argmax(er))


def test_constant_policy(env):
    obs = _obs_batch(env, 100, 3)
    for idx in range(N_ACTIONS):
        assert (constant_batch(idx)(env, obs) == idx).all()
