# Serial multi-echelon inventory system. Chapter: World Models and
# Model-Based Reinforcement Learning.
#
# A K-stage serial supply chain (stage 0 = retailer facing customer demand,
# stage K-1 = most upstream, replenished from an ample external supplier).
# Each link has a shipment lead time L. Demand is i.i.d. Poisson. Unmet
# customer demand is backordered at the retailer; internal shortfalls are
# backlogged between stages. Costs are installation holding at every stage
# plus a customer-backorder penalty at the retailer.
#
# For a serial system with i.i.d. demand, linear holding/penalty costs, and
# full backlogging, Clark & Scarf (1960) proved the optimal policy is an
# echelon base-stock policy: each stage raises its echelon inventory position
# to a level S_k. The oracle here executes that policy with base-stock levels
# found by simulation-optimization on the true model. The smoke test certifies
# the machinery on a single stage (K=1), where the optimal base-stock has the
# closed-form newsvendor critical-fractile solution, by checking the
# simulation-optimization result matches that closed form.

import numpy as np


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MultiEchelonEnv:
    """K-stage serial inventory system with lead time L and Poisson demand.

    Physical state carried across periods:
      inv[k]      on-hand inventory at stage k, k = 0..K-1
      pipe[k]     length-L vector of shipments in transit to stage k
                  (pipe[k][0] arrives at the start of next period)
      backlog[k]  units stage k+1 owes stage k (internal backlog); backlog[K-1]
                  is unused because the top stage draws from an ample supplier
      b           customer backorders at the retailer

    The learner observes the full physical state. The optimal policy needs only
    the K echelon inventory positions, a sufficient statistic the learner does
    not know a priori.
    """

    def __init__(
        self, K=3, L=1, lam=5.0, h=None, p=8.0, order_cap=20, inv_cap=60, T=200, seed=0
    ):
        self.K = K
        self.L = L
        self.lam = lam
        # Installation holding cost increases downstream (value added moving down).
        if h is None:
            h = [0.5 * (k + 1) for k in range(K)][::-1]  # e.g. K=3 -> [1.5,1.0,0.5]
        self.h = np.asarray(h, dtype=np.float64)
        assert self.h.shape[0] == K
        self.p = p
        self.order_cap = order_cap
        self.inv_cap = inv_cap
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        K, L = self.K, self.L
        # Start each stage with roughly one lead-time of mean demand on hand.
        base = int(round(self.lam * (self.L + 1)))
        self.inv = np.full(K, base, dtype=np.int64)
        self.pipe = np.full((K, L), int(round(self.lam)), dtype=np.int64)
        self.backlog = np.zeros(K, dtype=np.int64)  # backlog[k] = owed by k+1 to k
        self.b = 0
        return self._obs()

    def _obs(self):
        # Flat physical observation: on-hand, pipelines, internal backlog, cust backorder.
        return np.concatenate(
            [
                self.inv.astype(np.float64),
                self.pipe.reshape(-1).astype(np.float64),
                self.backlog[:-1].astype(
                    np.float64
                ),  # top stage has no upstream backlog
                np.array([self.b], dtype=np.float64),
            ]
        )

    def echelon_positions(self):
        """Echelon inventory position of each stage k (sufficient statistic).

        echelon stock_k = sum_{j<=k} on-hand_j + in-transit to stages j<k - b
        echelon IP_k    = echelon stock_k + in-transit into stage k (pipe[k])
        """
        K = self.K
        ip = np.zeros(K, dtype=np.float64)
        cum_onhand = 0
        cum_downstream_transit = 0  # in transit to stages strictly downstream of k
        for k in range(K):
            cum_onhand += self.inv[k]
            ech_stock = cum_onhand + cum_downstream_transit - self.b
            ip[k] = ech_stock + self.pipe[k].sum()
            cum_downstream_transit += self.pipe[k].sum()
        return ip

    def order_from_echelon(self, S):
        """Base-stock order quantities to raise each echelon IP toward S_k."""
        ip = self.echelon_positions()
        q = np.maximum(0, np.asarray(S, dtype=np.float64) - ip)
        return np.clip(q.astype(np.int64), 0, self.order_cap)

    def step(self, orders, demand=None):
        """Advance one period given integer order quantities per stage.

        Event sequence: receive arrivals -> place/ship orders (top-down) ->
        customer demand at retailer -> charge costs.
        Returns (obs, reward, done, info). reward = -cost.
        """
        K, L = self.K, self.L
        orders = np.clip(np.asarray(orders, dtype=np.int64), 0, self.order_cap)

        # (1) Receive arrivals: pipe[k][0] lands as on-hand at stage k.
        arrivals = self.pipe[:, 0].copy()
        self.inv = np.minimum(self.inv + arrivals, self.inv_cap)
        if L > 1:
            self.pipe[:, :-1] = self.pipe[:, 1:]
        self.pipe[:, -1] = 0

        # (2) Shipping, processed top-down. Top stage draws from ample supplier.
        # Top stage K-1: its order is always fully shipped into its pipe.
        self.pipe[K - 1, -1] = orders[K - 1]
        # Internal links: stage k+1 ships to stage k, honoring existing backlog.
        for k in range(K - 2, -1, -1):
            want = orders[k] + self.backlog[k]
            ship = int(min(self.inv[k + 1], want))
            self.inv[k + 1] -= ship
            self.pipe[k, -1] += ship
            self.backlog[k] = want - ship

        # (3) Customer demand at the retailer (stage 0).
        if demand is None:
            demand = int(self.rng.poisson(self.lam))
        need = demand + self.b
        sold = int(min(self.inv[0], need))
        self.inv[0] -= sold
        self.b = need - sold

        # (4) Costs: installation holding at each stage + customer backorder penalty.
        holding = float(np.dot(self.h, self.inv))
        penalty = self.p * self.b
        cost = holding + penalty
        reward = -cost

        self.t += 1
        done = self.t >= self.T
        info = {
            "demand": demand,
            "cost": cost,
            "orders": orders.copy(),
            "holding": holding,
            "penalty": penalty,
        }
        return self._obs(), reward, done, info


# ---------------------------------------------------------------------------
# Oracle: echelon base-stock policy, levels found by simulation optimization
# ---------------------------------------------------------------------------


def simulate_base_stock(env_params, S, demand_seq):
    """Roll a base-stock policy S over a fixed demand sequence; return total cost."""
    env = MultiEchelonEnv(
        **{k: v for k, v in env_params.items() if k != "seed"}, seed=0
    )
    env.reset()
    total = 0.0
    for d in demand_seq:
        orders = env.order_from_echelon(S)
        _, r, _, _ = env.step(orders, demand=int(d))
        total += -r
    return total


def find_oracle_base_stock(env_params, n_search_seq=12, T_search=400, seed=123):
    """Coordinate-ascent search over echelon base-stock levels S using common
    random numbers (shared demand sequences). Returns (S_star, mean_cost)."""
    rng = np.random.default_rng(seed)
    lam = env_params["lam"]
    K = env_params["K"]
    L = env_params["L"]
    demand_seqs = [rng.poisson(lam, size=T_search) for _ in range(n_search_seq)]

    def mean_cost(S):
        return float(
            np.mean([simulate_base_stock(env_params, S, dq) for dq in demand_seqs])
        )

    # Nested base-stock levels; initialize near (L+1)*lam scaled by stage depth.
    S = np.array(
        [int(round(lam * (L + 1) * (k + 1))) for k in range(K)], dtype=np.int64
    )
    best = mean_cost(S)
    hi = int(round(lam * (L + 1) * (K + 2)))
    improved = True
    while improved:
        improved = False
        for k in range(K):
            for cand in range(0, hi + 1):
                S_try = S.copy()
                S_try[k] = cand
                # keep nested monotonicity S_0 <= S_1 <= ... to stay feasible
                S_try = np.maximum.accumulate(S_try)
                c = mean_cost(S_try)
                if c < best - 1e-9:
                    best, S, improved = c, S_try, True
    return S, best


# ---------------------------------------------------------------------------
# Independent single-stage optimum: base-stock newsvendor over lead-time demand
# ---------------------------------------------------------------------------
#
# For a single stage (K=1) with backlogging, linear holding/penalty, and lead
# time L, the optimal policy is a base-stock policy whose level is the critical
# fractile of demand over the L+1-period protection interval (textbook result,
# e.g. Zipkin 2000). This closed form is fully independent of the simulator and
# of Clark-Scarf, so it certifies both the env dynamics and the base-stock
# oracle machinery. Multi-stage optimality of the base-stock FORM (used by the
# K>=2 oracle) rests separately on Clark & Scarf (1960) for serial systems.


def _poisson_pmf(lam, dmax):
    from math import factorial

    pmf = np.array([np.exp(-lam) * lam**d / factorial(d) for d in range(dmax + 1)])
    return pmf / pmf.sum()


def newsvendor_base_stock(env_params):
    """Exact single-stage base-stock level and per-period cost (closed form).

    Protection interval is L+1 periods (the order placed now arrives after L,
    so the position set now covers demand over the current plus L periods).
    Returns (S, per_period_cost, critical_ratio).
    """
    assert env_params["K"] == 1
    lam = env_params["lam"]
    L = env_params["L"]
    h0 = np.asarray(
        env_params.get("h") or [0.5 * (k + 1) for k in range(1)][::-1], dtype=np.float64
    )[0]
    p = env_params["p"]
    lam_lt = lam * (L + 1)  # lead-time (protection-interval) demand rate
    dmax = int(lam_lt * 6) + 10
    pmf = _poisson_pmf(lam_lt, dmax)
    cdf = np.cumsum(pmf)
    crit = p / (p + h0)
    S = int(np.searchsorted(cdf, crit))  # smallest S with P(D_LT <= S) >= crit
    d = np.arange(dmax + 1)
    cost = h0 * np.dot(pmf, np.maximum(S - d, 0)) + p * np.dot(
        pmf, np.maximum(d - S, 0)
    )
    return S, float(cost), crit


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = dict(K=2, L=1, lam=5.0, p=8.0, order_cap=20, inv_cap=60, T=200)
    print(
        f"MultiEchelonEnv: K={params['K']}, L={params['L']}, lam={params['lam']}, "
        f"p={params['p']}"
    )
    env = MultiEchelonEnv(**params, seed=0)
    print(f"  obs dim = {env._obs().shape[0]}")
    print(f"  initial echelon positions = {env.echelon_positions()}")

    # Random-policy sanity roll.
    env.reset()
    rng = np.random.default_rng(1)
    tot = 0.0
    for _ in range(50):
        orders = rng.integers(0, 8, size=params["K"])
        _, r, _, info = env.step(orders)
        tot += -r
    print(f"  random-policy 50-step cost = {tot:.1f}")

    # Oracle base-stock via simulation optimization.
    print("\nSearching oracle base-stock (simulation optimization)...")
    S_star, mean_c = find_oracle_base_stock(params, n_search_seq=8, T_search=300)
    print(f"  S* = {S_star}, mean per-episode cost (T=300) = {mean_c:.1f}")
    print(f"  per-period cost = {mean_c / 300:.3f}")

    # Certify env + oracle against the INDEPENDENT single-stage closed form.
    one = dict(K=1, L=1, lam=4.0, p=8.0, order_cap=14, inv_cap=40, T=100)
    S_nv, pp_nv, crit = newsvendor_base_stock(one)
    print(
        f"\nSingle-stage newsvendor (K=1, lam=4, p=8, h=0.5): "
        f"critical ratio={crit:.4f}, base-stock S={S_nv}, per-period cost={pp_nv:.4f}"
    )
    S_sim, c_sim = find_oracle_base_stock(one, n_search_seq=12, T_search=2000)
    print(
        f"  simulation-opt base-stock S*={S_sim[0]}, per-period cost={c_sim / 2000:.4f}"
    )
    assert abs(int(S_sim[0]) - int(S_nv)) <= 1, (
        f"sim-opt base-stock {S_sim[0]} must match newsvendor optimum {S_nv}"
    )
    assert abs(c_sim / 2000 - pp_nv) < 0.5 * pp_nv, (
        f"sim cost {c_sim / 2000:.3f} must match closed-form {pp_nv:.3f}"
    )
    print("\nSmoke test passed.")
