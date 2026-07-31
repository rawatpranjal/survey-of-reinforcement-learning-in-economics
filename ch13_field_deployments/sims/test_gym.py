# Teeth tests for gym-API compliance, seeding, and termination.
import numpy as np
import pytest

from promo_env import PromoEnv, PromoConfig, N_ACTIONS


@pytest.fixture
def env():
    return PromoEnv(PromoConfig(K=8), seed=0)


def test_reset_contract(env):
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (env.cfg.K + 2,)
    assert env.observation_space.contains(obs)


def test_step_contract_and_termination(env):
    env.reset(seed=0)
    for t in range(env.cfg.horizon):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (env.cfg.K + 2,)
        assert isinstance(r, float)
        assert truncated is False
        assert terminated == (t == env.cfg.horizon - 1)


def test_seed_reproducible(env):
    e1 = PromoEnv(PromoConfig(K=8))
    e2 = PromoEnv(PromoConfig(K=8))
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.array_equal(o1, o2)
    seq1 = [(e1.step(3)[0].copy(), e1.step(1)[1]) for _ in range(10)]
    e2.reset(seed=123)
    seq2 = [(e2.step(3)[0].copy(), e2.step(1)[1]) for _ in range(10)]
    for (a_obs, a_r), (b_obs, b_r) in zip(seq1, seq2):
        assert np.array_equal(a_obs, b_obs) and a_r == b_r


def test_distinct_seeds_differ(env):
    o1, _ = env.reset(seed=1)
    o2, _ = env.reset(seed=2)
    assert not np.array_equal(o1, o2)


def test_action_space(env):
    assert env.action_space.n == N_ACTIONS
