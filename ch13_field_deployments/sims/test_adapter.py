# Teeth tests for the OLD-gym adapter that presents PromoEnv at the Scope-RL boundary.
import numpy as np
import gym
import pytest

from promo_env import PromoEnv, PromoConfig, N_ACTIONS
from pipeline import PromoGymAdapter


@pytest.fixture
def cfg():
    return PromoConfig(K=8)


def test_is_old_gym_env(cfg):
    ad = PromoGymAdapter(cfg, seed=0)
    assert isinstance(ad, gym.Env), "Scope-RL hard-checks isinstance(env, gym.Env)"


def test_spaces_are_gym_spaces(cfg):
    ad = PromoGymAdapter(cfg, seed=0)
    assert isinstance(ad.action_space, gym.spaces.Discrete)
    assert ad.action_space.n == N_ACTIONS
    assert isinstance(ad.observation_space, gym.spaces.Box)
    assert ad.observation_space.shape == (cfg.K + 2,)


def test_api_tuples(cfg):
    ad = PromoGymAdapter(cfg, seed=0)
    obs, info = ad.reset(seed=0)
    assert obs.shape == (cfg.K + 2,) and isinstance(info, dict)
    out = ad.step(2)
    assert len(out) == 5  # obs, reward, terminated, truncated, info


def test_parity_with_promoenv(cfg):
    # The adapter must not alter dynamics: identical obs + rewards for the same seed.
    ad = PromoGymAdapter(cfg, seed=0)
    raw = PromoEnv(cfg, seed=0)
    o_ad, _ = ad.reset(seed=7)
    o_raw, _ = raw.reset(seed=7)
    assert np.array_equal(o_ad, o_raw)
    actions = [0, 5, 2, 3, 1, 4, 2, 0, 5, 1]
    for a in actions:
        oa, ra, ta, tra, _ = ad.step(a)
        orw, rr, tr, trr, _ = raw.step(a)
        assert np.array_equal(oa, orw)
        assert ra == rr and ta == tr and tra == trr
