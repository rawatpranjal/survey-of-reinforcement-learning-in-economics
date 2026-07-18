# Teeth tests for the vectorized true-value rollout (the ground-truth oracle).
import numpy as np
import pytest

from promo_env import (
    PromoEnv,
    PromoConfig,
    myopic_batch,
    constant_batch,
    vec_rollout_value,
)


@pytest.fixture
def env():
    return PromoEnv(PromoConfig(K=8), seed=0)


def _scalar_rollout(env, batch_policy, n_episodes, seed):
    # Independent single-episode rollout via the gym-API step path, to cross-check
    # the vectorized dynamics. Uses per-episode env RNG (different draws than vec),
    # so it agrees only in expectation (within a few SE).
    g = env.cfg.gamma
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, disc, tot = False, 1.0, 0.0
        while not done:
            a = int(batch_policy(env, obs[None, :], np.random.default_rng(0))[0])
            obs, r, term, trunc, _ = env.step(a)
            tot += disc * r
            disc *= g
            done = term or trunc
        returns[ep] = tot
    return returns.mean(), returns.std(ddof=1) / np.sqrt(n_episodes)


def test_deterministic(env):
    v1 = vec_rollout_value(env, constant_batch(0), 3000, seed=1000)
    v2 = vec_rollout_value(env, constant_batch(0), 3000, seed=1000)
    assert v1 == v2


def test_vec_matches_scalar_path(env):
    # A vectorization bug would shift the mean well beyond sampling error.
    N = 6000
    vm, vse = vec_rollout_value(env, constant_batch(0), N, seed=2000)
    sm, sse = _scalar_rollout(env, constant_batch(0), N, seed=9000)
    assert abs(vm - sm) < 4.0 * np.hypot(vse, sse), f"vec={vm:.4f} scalar={sm:.4f}"


def test_restraint_beats_myopic(env):
    N = 8000
    v_no, se_no = vec_rollout_value(env, constant_batch(0), N, seed=1000)
    v_my, se_my = vec_rollout_value(env, myopic_batch, N, seed=1000)
    assert v_no - v_my > 3.0 * np.hypot(se_no, se_my), (
        f"no-promo={v_no:.3f} myopic={v_my:.3f}"
    )


def test_standard_error_scales(env):
    _, se_small = vec_rollout_value(env, constant_batch(0), 2000, seed=1000)
    _, se_large = vec_rollout_value(env, constant_batch(0), 8000, seed=1000)
    assert 1.5 < se_small / se_large < 2.5  # se ~ 1/sqrt(N), ratio ~ 2
