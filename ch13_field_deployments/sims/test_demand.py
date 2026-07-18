# Teeth tests for the demand equation (buy_prob_vec) -- the single source of truth.
import numpy as np
import pytest

from promo_env import PromoEnv, PromoConfig, DISCOUNTS


@pytest.fixture
def env():
    return PromoEnv(PromoConfig(K=8), seed=0)


def _logit(p):
    return np.log(p / (1.0 - p))


def test_probs_strictly_interior(env):
    rng = np.random.default_rng(0)
    C = np.clip(rng.normal(size=(3000, env.cfg.K)), -4, 4)
    for d in DISCOUNTS:
        p = env.buy_prob_vec(C, np.full(3000, 0.10), np.full(3000, d))
        assert (p > 0).all() and (p < 1).all()


def test_strictly_increasing_in_discount(env):
    # Lower price + larger reference gain => buy prob strictly increases with discount.
    rng = np.random.default_rng(1)
    C = np.clip(rng.normal(size=(2000, env.cfg.K)), -4, 4)
    r = np.full(2000, 0.10)
    probs = [env.buy_prob_vec(C, r, np.full(2000, d)) for d in DISCOUNTS]
    for lo, hi in zip(probs[:-1], probs[1:]):
        assert (hi > lo).all(), "buy_prob must strictly increase in discount"


def test_decreasing_in_price_sensitivity(env):
    # alpha = alpha_base * exp(theta_scale * (c.theta)/sqrt(n_signal)); theta is unit-norm.
    hi_alpha_ctx = (+2.0 * env._theta)[None, :]  # aligned with theta -> high alpha
    lo_alpha_ctx = (-2.0 * env._theta)[None, :]  # anti-aligned -> low alpha
    d = np.array([0.10])
    r = np.array([0.10])
    p_hi = env.buy_prob_vec(hi_alpha_ctx, r, d)[0]
    p_lo = env.buy_prob_vec(lo_alpha_ctx, r, d)[0]
    assert p_hi < p_lo, (
        "more price-sensitive consumers must buy less at a positive price"
    )


def test_scalar_matches_vector(env):
    rng = np.random.default_rng(2)
    for _ in range(200):
        c = np.clip(rng.normal(size=env.cfg.K), -4, 4)
        r = float(rng.uniform(0, 0.25))
        d = float(rng.choice(DISCOUNTS))
        assert env.buy_prob(c, r, d) == pytest.approx(
            float(env.buy_prob_vec(c[None, :], [r], [d])[0]), rel=0, abs=1e-12
        )


def test_loss_aversion_asymmetry(env):
    # At fixed discount, a reference shortfall (d<r) hurts utility more than an equal
    # surplus (d>r) helps, by exactly the loss_aversion factor (in log-odds).
    c = np.zeros((1, env.cfg.K))  # average consumer, alpha = alpha_base
    d, x = 0.10, 0.05
    p_gain = env.buy_prob_vec(c, [d - x], [d])[0]  # gap = +x
    p_neut = env.buy_prob_vec(c, [d], [d])[0]  # gap = 0
    p_loss = env.buy_prob_vec(c, [d + x], [d])[0]  # gap = -x
    gain_step = _logit(p_gain) - _logit(p_neut)
    loss_step = _logit(p_neut) - _logit(p_loss)
    assert loss_step / gain_step == pytest.approx(env.cfg.loss_aversion, rel=1e-6)


def test_index_is_buried(env):
    # Only the first n_signal context coords drive demand; the rest are pure noise.
    rng = np.random.default_rng(3)
    base = np.clip(rng.normal(size=(500, env.cfg.K)), -4, 4)
    perturbed = base.copy()
    ns = env.cfg.n_signal
    perturbed[:, ns:] = np.clip(rng.normal(size=(500, env.cfg.K - ns)), -4, 4)
    r, d = np.full(500, 0.1), np.full(500, 0.15)
    p_base = env.buy_prob_vec(base, r, d)
    p_pert = env.buy_prob_vec(perturbed, r, d)
    assert np.allclose(p_base, p_pert, atol=1e-12), (
        "noise context coords must not affect demand"
    )
