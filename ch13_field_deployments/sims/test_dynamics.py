# Teeth tests for the reference-discount dynamics and observation encoding.
import numpy as np
import pytest

from promo_env import PromoEnv, PromoConfig, DISCOUNTS, N_ACTIONS


@pytest.fixture
def env():
    return PromoEnv(PromoConfig(K=6), seed=0)


def test_reference_is_running_mean_of_offered_discounts(env):
    env.reset(seed=0)
    actions = [5, 0, 3, 3, 1, 0, 4, 2]
    offered = []
    for a in actions:
        env.step(a)
        offered.append(DISCOUNTS[a])
        assert env._r == pytest.approx(np.mean(offered), rel=0, abs=1e-12)


def test_reference_rises_under_deep_discounting(env):
    env.reset(seed=0)
    prev = env._r
    for _ in range(20):
        env.step(N_ACTIONS - 1)  # always 25%
        assert env._r >= prev - 1e-12
        prev = env._r
    assert env._r > 0.20


def test_reference_decays_under_no_promo(env):
    env.reset(seed=0)
    for _ in range(30):
        env.step(0)
    assert env._r < 0.02


def test_reference_bounded(env):
    rng = np.random.default_rng(4)
    env.reset(seed=0)
    for _ in range(200):
        env.step(int(rng.integers(N_ACTIONS)))
        assert 0.0 <= env._r <= DISCOUNTS[-1] + 1e-12


def test_observation_encodes_reference_and_time(env):
    obs, _ = env.reset(seed=1)
    K = env.cfg.K
    assert obs[K] == pytest.approx(env._r)
    assert obs[K + 1] == pytest.approx(0.0)
    for expected_t in range(1, 6):
        obs, *_ = env.step(2)
        assert obs[K] == pytest.approx(env._r)
        assert obs[K + 1] == pytest.approx(expected_t / env.cfg.horizon)
