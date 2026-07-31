# Teeth tests for EpsilonMyopicHead: exact greedy rule, analytic propensities,
# empirical action frequencies matching the stated policy.
import numpy as np

from promo_env import PromoConfig, PromoEnv, myopic_batch, N_ACTIONS
from pipeline import behavior_head


CFG = PromoConfig(K=8)


def _states(n, seed):
    rng = np.random.default_rng(seed)
    C = np.clip(rng.normal(size=(n, CFG.K)), -CFG.ctx_clip, CFG.ctx_clip)
    r = rng.uniform(0.0, 0.25, size=n)
    t = rng.uniform(0.0, 1.0, size=n)
    return np.concatenate([C, r[:, None], t[:, None]], axis=1).astype(np.float32)


def test_greedy_matches_exact_myopic_rule():
    head = behavior_head(CFG, "incumbent", seed=0, incumbent_epsilon=0.3)
    x = _states(1000, 1)
    env = PromoEnv(CFG, seed=0)
    ref = np.asarray(myopic_batch(env, x)).astype(int)
    assert np.array_equal(head.predict(x), ref), (
        "incumbent head greedy must equal the exact myopic rule (no clone, no drift)"
    )


def test_pscores_analytic_and_normalized():
    head = behavior_head(CFG, "incumbent", seed=0, incumbent_epsilon=0.3)
    x = _states(500, 2)
    probs = head.calc_action_choice_probability(x)
    assert probs.shape == (500, N_ACTIONS)
    assert np.allclose(probs.sum(axis=1), 1.0), "rows must sum to 1"
    greedy = head.predict(x)
    hi = 1.0 - 0.3 + 0.3 / N_ACTIONS
    lo = 0.3 / N_ACTIONS
    assert np.allclose(probs[np.arange(500), greedy], hi)
    # pscore-given-action agrees with the row of the full distribution
    a = np.full(500, 3)
    ps = head.calc_pscore_given_action(x, a)
    assert np.allclose(ps, probs[np.arange(500), a])
    assert set(np.round(np.unique(ps), 9)) <= {round(hi, 9), round(lo, 9)}


def test_epsilon_zero_reduces_to_pure_rule():
    head = behavior_head(CFG, "incumbent", seed=0, incumbent_epsilon=0.0)
    x = _states(300, 3)
    assert np.array_equal(head.sample_action(x), head.predict(x))


def test_empirical_frequencies_match_propensities():
    head = behavior_head(CFG, "incumbent", seed=7, incumbent_epsilon=0.3)
    x = np.tile(_states(1, 4), (20000, 1))  # one state, many samples
    a = head.sample_action(x)
    probs = head.calc_action_choice_probability(x[:1])[0]
    freq = np.bincount(a, minlength=N_ACTIONS) / len(a)
    # 3-sigma binomial tolerance per action
    tol = 3 * np.sqrt(probs * (1 - probs) / len(a))
    assert np.all(np.abs(freq - probs) < tol + 1e-12), (
        f"empirical {freq} vs analytic {probs}"
    )


def test_sample_and_pscore_consistent():
    head = behavior_head(CFG, "incumbent", seed=3, incumbent_epsilon=0.3)
    x = _states(400, 5)
    action, pscore = head.sample_action_and_output_pscore(x)
    assert np.allclose(pscore, head.calc_pscore_given_action(x, action))
