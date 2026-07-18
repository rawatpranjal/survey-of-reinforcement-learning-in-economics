# Teeth tests for log generation: the A/B (uniform, full support) vs observational
# (concentrated, collapsed support) contrast that drives the whole study.
import numpy as np
import pytest

from promo_env import PromoConfig, N_ACTIONS
from pipeline import generate_log


CFG = PromoConfig(K=8)


def _action_hist(lg):
    return np.bincount(lg["action"].astype(int), minlength=N_ACTIONS) / lg["size"]


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@pytest.fixture(scope="module")
def logs():
    return {
        "ab": generate_log(CFG, "ab", n_traj=300, seed=0),
        "obs": generate_log(CFG, "observational", n_traj=300, seed=0),
    }


def test_schema_and_size(logs):
    for lg in logs.values():
        for k in ["state", "action", "reward", "done", "terminal", "pscore", "size"]:
            assert k in lg, f"missing log key {k}"
        assert lg["size"] == 300 * CFG.horizon
        assert lg["state"].shape == (lg["size"], CFG.K + 2)


def test_ab_is_uniform_full_support(logs):
    lg = logs["ab"]
    assert np.allclose(lg["pscore"], 1.0 / N_ACTIONS), "A/B propensities must be 1/N"
    h = _action_hist(lg)
    assert h.min() > 0.12, f"A/B should cover all actions, min share={h.min():.3f}"


def test_observational_collapses_support(logs):
    ab, ob = logs["ab"], logs["obs"]
    h_ab, h_ob = _action_hist(ab), _action_hist(ob)
    assert _entropy(h_ob) < _entropy(h_ab) - 0.2, (
        "observational must be more concentrated"
    )
    assert h_ob.min() < 0.5 * h_ab.min(), "observational must thin some actions"
    assert ob["pscore"].std() > 1e-3, "observational propensities must vary with state"
