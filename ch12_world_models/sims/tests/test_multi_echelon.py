# Audit tests for the multi-echelon supply-chain sim. Run with pytest.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_echelon_env import (  # noqa: E402
    MultiEchelonEnv,
    newsvendor_base_stock,
    find_oracle_base_stock,
)


# ---------------------------------------------------------------------------
# 1. Clark-Scarf / base-stock optimality certified by the single-stage closed
#    form. On a K=1 instance the base-stock newsvendor level is provably optimal;
#    simulation optimization on the true model must recover it.
# ---------------------------------------------------------------------------


def test_single_stage_matches_newsvendor():
    for lam, p in [(3.0, 6.0), (4.0, 8.0), (6.0, 12.0)]:
        one = dict(K=1, L=1, lam=lam, p=p, order_cap=16, inv_cap=50, T=100)
        S_nv, pp_nv, crit = newsvendor_base_stock(one)
        S_sim, c_sim = find_oracle_base_stock(one, n_search_seq=12, T_search=2000)
        # base-stock level within 1 unit of the closed-form optimum
        assert abs(int(S_sim[0]) - int(S_nv)) <= 1, (lam, p, S_sim[0], S_nv)
        # simulated per-period cost within 15% of the closed-form cost
        assert abs(c_sim / 2000 - pp_nv) < 0.15 * pp_nv, (lam, p, c_sim / 2000, pp_nv)


def test_newsvendor_critical_ratio():
    # Critical ratio p/(p+h) with default h[0]=0.5 for K=1.
    one = dict(K=1, L=1, lam=4.0, p=8.0, order_cap=16, inv_cap=50, T=100)
    _, _, crit = newsvendor_base_stock(one)
    assert abs(crit - 8.0 / (8.0 + 0.5)) < 1e-9


# ---------------------------------------------------------------------------
# 2. Environment fidelity: conservation and cost accounting.
# ---------------------------------------------------------------------------


def test_backorders_and_onhand_never_coexist_at_retailer():
    env = MultiEchelonEnv(K=3, L=1, lam=5.0, T=200, seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(200):
        env.step(rng.integers(0, 8, size=3))
        # retailer: after a demand fill, on-hand and backorders cannot both be > 0
        assert not (env.inv[0] > 0 and env.b > 0)


def test_cost_is_holding_plus_penalty():
    env = MultiEchelonEnv(K=3, L=1, lam=5.0, p=8.0, T=50, seed=1)
    env.reset()
    _, r, _, info = env.step(np.array([5, 5, 5]))
    expected = float(np.dot(env.h, env.inv)) + env.p * env.b
    assert abs(info["cost"] - expected) < 1e-9
    assert abs(-r - info["cost"]) < 1e-9


def test_oracle_base_stock_is_nested():
    # Echelon base-stock levels must be non-decreasing upstream (Clark-Scarf).
    ep = dict(K=3, L=1, lam=5.0, p=8.0, order_cap=20, inv_cap=60)
    S, _ = find_oracle_base_stock(ep, n_search_seq=8, T_search=400)
    assert S[0] <= S[1] <= S[2], S


# ---------------------------------------------------------------------------
# 3. No information leakage: learners must not read the true demand rate,
#    the oracle base-stock, or env internals during learning.
# ---------------------------------------------------------------------------


def test_no_leakage_learner_source():
    import inspect
    import multi_echelon_paradigms as mp

    # Learners may read known problem bounds (order_cap, inv_cap, K, L, T) and
    # what they observe, but must never read the true demand rate, holding cost,
    # penalty, or the oracle base-stock levels.
    forbidden = ["env.lam", "env.p", "env.h", "S_oracle["]
    for cls in (
        mp.DQNAgent,
        mp.WorldModelPlanner,
        mp.DecentralizedBaseStock,
        mp.NaiveConstant,
    ):
        src = inspect.getsource(cls)
        # strip each method signature line so parameter names (e.g. S_oracle) in
        # signatures are not flagged; only usages in bodies count.
        for token in forbidden:
            assert token not in src, (
                f"{cls.__name__} leaks a true environment parameter: {token}"
            )
        # S_oracle may appear only as the unused reset() parameter name.
        assert src.count("S_oracle") == src.count("def reset(self, env, S_oracle"), (
            f"{cls.__name__} uses S_oracle beyond the reset signature (oracle leakage)"
        )


def test_world_model_uses_no_true_params_in_planning():
    import inspect
    import multi_echelon_paradigms as mp

    src = inspect.getsource(mp.WorldModelPlanner._model_cost_batch)
    # planning cost must come from the learned model (self.model), not the env.
    assert "self.model(" in src
    assert "MultiEchelonEnv" not in src
    assert ".step(" not in src  # never calls the true environment while planning


# ---------------------------------------------------------------------------
# 4. World-model learning: one-step forecast error decreases with data.
# ---------------------------------------------------------------------------


def test_world_model_forecast_improves():
    import torch
    import multi_echelon_paradigms as mp

    ep = dict(K=3, L=1, lam=5.0, p=8.0, order_cap=20, inv_cap=60)
    S, _ = find_oracle_base_stock(ep, n_search_seq=6, T_search=300)
    env = MultiEchelonEnv(**ep, T=2000, seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    data = []  # (obs, action, demand, next_obs)
    for _ in range(2000):
        a = np.clip(env.order_from_echelon(S) + rng.integers(-3, 4, size=3), 0, 20)
        no, r, _, info = env.step(a)
        data.append((obs.copy(), a.copy(), info["demand"], no.copy()))
        obs = no

    oscale = ep["lam"] * (ep["L"] + 1) * ep["K"]
    ascale = ep["order_cap"]
    dscale = ep["lam"]
    torch.manual_seed(0)
    model = mp.WorldModelNet(obs.shape[0], 3, 128)  # demand-conditioned (obs+act+1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    O = np.stack([d[0] for d in data])
    A = np.stack([d[1] for d in data])
    D = np.array([[d[2]] for d in data])
    ON = np.stack([d[3] for d in data])

    def forecast_mse():
        with torch.no_grad():
            o = torch.tensor(O / oscale, dtype=torch.float32)
            a = torch.tensor(A / ascale, dtype=torch.float32)
            d = torch.tensor(D / dscale, dtype=torch.float32)
            dp, _ = model(torch.cat([o, a, d], -1))
            pred = (o + dp).numpy() * oscale
        return float(np.mean((pred - ON) ** 2))

    mse_start = forecast_mse()
    for _ in range(1500):
        idx = rng.integers(0, len(data), size=128)
        o = torch.tensor(O[idx] / oscale, dtype=torch.float32)
        a = torch.tensor(A[idx] / ascale, dtype=torch.float32)
        d = torch.tensor(D[idx] / dscale, dtype=torch.float32)
        on = torch.tensor(ON[idx] / oscale, dtype=torch.float32)
        dp, _ = model(torch.cat([o, a, d], -1))
        loss = torch.nn.functional.mse_loss(dp, on - o)
        opt.zero_grad()
        loss.backward()
        opt.step()
    mse_end = forecast_mse()
    # the learned model must forecast the pipeline transition far better than
    # the untrained model
    assert mse_end < 0.25 * mse_start, (mse_start, mse_end)
    assert mse_end < 5.0, mse_end  # absolute: obs entries up to ~60


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
