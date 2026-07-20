# Teeth tests for log generation: the A/B (uniform, full support), incumbent
# (concentrated, narrow support), and historical-mixture (r-axis-sweeping) regimes.
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
    out = {}
    out["ab"], out["ab_head"] = generate_log(CFG, "ab", n_traj=300, seed=0)
    out["inc"], out["inc_head"] = generate_log(CFG, "incumbent", n_traj=300, seed=0)
    out["mix"], out["mix_head"] = generate_log(CFG, "mixture", n_traj=300, seed=0)
    return out


def test_schema_and_size(logs):
    for key in ("ab", "inc", "mix"):
        lg = logs[key]
        for k in ["state", "action", "reward", "done", "terminal", "pscore", "size"]:
            assert k in lg, f"missing log key {k}"
        assert lg["size"] == 300 * CFG.horizon
        assert lg["state"].shape == (lg["size"], CFG.K + 2)


def test_ab_is_uniform_full_support(logs):
    lg = logs["ab"]
    assert np.allclose(lg["pscore"], 1.0 / N_ACTIONS), "A/B propensities must be 1/N"
    h = _action_hist(lg)
    assert h.min() > 0.12, f"A/B should cover all actions, min share={h.min():.3f}"


def test_incumbent_narrows_support(logs):
    # the faithful epsilon-greedy myopic is state-dependent, so it is less collapsed
    # than the old under-trained BC imitation (which piled 48% of mass on one action);
    # it must still be clearly more concentrated than uniform, with a dominant action
    ab, inc = logs["ab"], logs["inc"]
    h_ab, h_inc = _action_hist(ab), _action_hist(inc)
    assert _entropy(h_inc) < _entropy(h_ab) - 0.15, (
        f"incumbent must be more concentrated (got {_entropy(h_inc):.3f} "
        f"vs uniform {_entropy(h_ab):.3f})"
    )
    assert h_inc.max() > 2.0 * h_ab.max(), "incumbent must have a dominant action"
    assert h_inc.min() < 0.5 * h_ab.min(), "incumbent must thin some actions"
    assert inc["pscore"].std() > 1e-3, "incumbent propensities must vary with state"


def test_incumbent_pscores_are_analytic(logs):
    # every propensity is either 1-eps+eps/N (greedy action) or eps/N (explore)
    inc = logs["inc"]
    eps = logs["inc_head"].epsilon
    hi = 1.0 - eps + eps / N_ACTIONS
    lo = eps / N_ACTIONS
    ps = np.asarray(inc["pscore"], dtype=float)
    assert np.all(np.isclose(ps, hi) | np.isclose(ps, lo)), (
        "incumbent pscores must take exactly the two epsilon-greedy values"
    )
    # the greedy action must actually be played ~1-eps+eps/N of the time
    frac_hi = float(np.isclose(ps, hi).mean())
    assert abs(frac_hi - hi) < 0.02, f"greedy share {frac_hi:.3f} vs expected {hi:.3f}"


def test_mixture_covers_all_actions_heavily(logs):
    # each constant component plays its action >= (1-eps) of its episodes, so every
    # action gets a substantial share of the pooled log (no thin tail like incumbent)
    h = _action_hist(logs["mix"])
    assert h.min() > 0.05, f"mixture log min action share {h.min():.3f} too thin"
    assert len(logs["mix_head"].component_trace) == 300, (
        "one component draw per episode"
    )


def test_mixture_pscore_is_realized_component(logs):
    # reconstruct: for each episode, the traced component's greedy action must carry
    # pscore 1-eps+eps/N when logged, and any other action eps/N
    mix, head = logs["mix"], logs["mix_head"]
    eps = head.epsilon
    hi = 1.0 - eps + eps / N_ACTIONS
    lo = eps / N_ACTIONS
    T = CFG.horizon
    states = np.asarray(mix["state"], dtype=np.float32)
    actions = np.asarray(mix["action"]).astype(int)
    ps = np.asarray(mix["pscore"], dtype=float)
    trace = head.component_trace
    for ep in range(0, 300, 37):  # spot-check a spread of episodes
        k = trace[ep]
        sl = slice(ep * T, (ep + 1) * T)
        greedy = head.component_greedy(k, states[sl])
        match = actions[sl] == greedy
        expected = np.where(match, hi, lo)
        assert np.allclose(ps[sl], expected), (
            f"episode {ep}: logged pscore does not match realized component {k}"
        )
