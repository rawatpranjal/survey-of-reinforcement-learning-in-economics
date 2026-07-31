# Logistic-growth fishery. Chapter: World Models and Model-Based RL.
# Exogenous environment in the §9 dual sim; complements the self-referential
# cobweb by isolating sample efficiency from curvature variation.

import numpy as np


class FisheryEnv:
    """Logistic-growth fishery with quadratic harvest cost.

    State: s_t in [0, s_max] (stock biomass, scalar).
    Action: h_t in [0, min(s_t, h_max)] (harvest).
    Dynamics: s_{t+1} = max(0, s_t + r * s_t * (1 - s_t/K) - h_t + eps_t),
              eps_t ~ N(0, sigma^2).
    Reward:   r_t = p * h_t - (c/2) * h_t^2.
    """

    def __init__(self, r, K, p, c, sigma, gamma, T, s_max=None, h_max=None, seed=0):
        self.r, self.K, self.p, self.c = r, K, p, c
        self.sigma, self.gamma, self.T = sigma, gamma, T
        self.s_max = 1.5 * K if s_max is None else s_max
        self.h_msy = r * K / 4.0
        self.h_max = 1.5 * self.h_msy if h_max is None else h_max
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.s = K

    def reset(self):
        self.t = 0
        self.s = self.K
        return self.s

    def step(self, h_t):
        h_t = float(np.clip(h_t, 0.0, min(self.s, self.h_max)))
        eps = self.rng.normal(0.0, self.sigma)
        growth = self.r * self.s * (1.0 - self.s / self.K)
        s_next = max(0.0, self.s + growth - h_t + eps)
        s_next = min(s_next, self.s_max)
        r_t = self.p * h_t - 0.5 * self.c * h_t**2
        self.s = s_next
        self.t += 1
        done = self.t >= self.T
        return self.s, r_t, done, {"h": h_t}


def solve_oracle_dp(r, K, p, c, sigma, gamma, n_s=60, n_h=31, tol=1e-7, max_iter=400):
    """Grid-based discounted DP for the logistic fishery."""
    s_max = 1.5 * K
    h_msy = r * K / 4.0
    h_max = 1.5 * h_msy
    s_grid = np.linspace(0.0, s_max, n_s)
    h_grid = np.linspace(0.0, h_max, n_h)
    sigma_pts, w_pts = np.polynomial.hermite_e.hermegauss(9)
    sigma_pts = sigma_pts * sigma
    w_pts = w_pts / np.sqrt(2 * np.pi)
    V = np.zeros(n_s)
    converged = False
    for it in range(max_iter):
        V_next = np.empty(n_s)
        for i, s in enumerate(s_grid):
            best_q = -np.inf
            growth = r * s * (1.0 - s / K)
            for h in h_grid:
                if h > s:
                    break
                r_imm = p * h - 0.5 * c * h**2
                s_mean = max(0.0, s + growth - h)
                e_v = 0.0
                for sp_offset, w in zip(sigma_pts, w_pts):
                    s_next = np.clip(s_mean + sp_offset, 0.0, s_max)
                    e_v += w * np.interp(s_next, s_grid, V)
                q = r_imm + gamma * e_v
                if q > best_q:
                    best_q = q
            V_next[i] = best_q
        if np.max(np.abs(V_next - V)) < tol:
            V = V_next
            converged = True
            break
        V = V_next
    g = np.empty(n_s)
    for i, s in enumerate(s_grid):
        best_q = -np.inf
        best_h = 0.0
        growth = r * s * (1.0 - s / K)
        for h in h_grid:
            if h > s:
                break
            r_imm = p * h - 0.5 * c * h**2
            s_mean = max(0.0, s + growth - h)
            e_v = 0.0
            for sp_offset, w in zip(sigma_pts, w_pts):
                s_next = np.clip(s_mean + sp_offset, 0.0, s_max)
                e_v += w * np.interp(s_next, s_grid, V)
            q = r_imm + gamma * e_v
            if q > best_q:
                best_q = q
                best_h = h
        g[i] = best_h
    return dict(
        s_grid=s_grid,
        h_grid=h_grid,
        V=V,
        g=g,
        h_msy=h_msy,
        h_max=h_max,
        s_max=s_max,
        n_iter=it + 1,
        converged=converged,
    )


def oracle_action(s, oracle_dict):
    """Linearly interpolate the greedy policy at stock s."""
    return float(np.interp(s, oracle_dict["s_grid"], oracle_dict["g"]))


if __name__ == "__main__":
    r, K, p, c, sigma, gamma = 0.4, 10.0, 2.0, 0.2, 0.3, 0.95
    print(f"Params: r={r}, K={K}, p={p}, c={c}, sigma={sigma}, gamma={gamma}")
    print(f"MSY: h_MSY = rK/4 = {r * K / 4:.3f} at s* = K/2 = {K / 2:.3f}")
    oracle = solve_oracle_dp(r, K, p, c, sigma, gamma)
    print(f"DP solve: converged={oracle['converged']} in {oracle['n_iter']} iters")
    print(
        f"  V(0)={oracle['V'][0]:.3f}  V(K/2)={np.interp(K / 2, oracle['s_grid'], oracle['V']):.3f}  V(K)={oracle['V'][-1]:.3f}"
    )
    print(
        f"  g(K/2)={np.interp(K / 2, oracle['s_grid'], oracle['g']):.3f} (target MSY={r * K / 4:.3f})"
    )
    print(f"  g(K)={np.interp(K, oracle['s_grid'], oracle['g']):.3f}")
    env = FisheryEnv(r=r, K=K, p=p, c=c, sigma=0.0, gamma=gamma, T=100, seed=0)
    s = env.reset()
    s_history = [s]
    for _ in range(100):
        h = oracle_action(s, oracle)
        s, _, _, _ = env.step(h)
        s_history.append(s)
    print("\nDeterministic oracle rollout (sigma=0):")
    print(
        f"  s_0={s_history[0]:.3f}  s_50={s_history[50]:.3f}  s_100={s_history[-1]:.3f}"
    )
    print(f"  Steady-state target: s*={K / 2:.3f}")
