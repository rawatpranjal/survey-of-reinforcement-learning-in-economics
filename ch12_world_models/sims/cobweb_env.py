# Cobweb model with adjustment cost. Chapter: Forecasting, Dreaming and Learning.
# Self-referential environment: price p_t = a - b q_t + eps_t, with reward
# r_t = p_t q_t - (c/2) q_t^2 - (phi/2)(q_t - q_{t-1})^2.

import numpy as np


class CobwebEnv:
    """Self-referential cobweb with quadratic adjustment cost.

    State: s_t = (q_{t-1}, p_{t-1}) in R^2 (p_{t-1} is observed but ignored
        by the optimal policy; learners see it and must infer its irrelevance).
    Action: q_t in [q_min, q_max].
    Dynamics: p_t = a - b q_t + eps_t,    eps_t ~ N(0, sigma^2).
    Reward:   r_t = p_t q_t - (c/2) q_t^2 - (phi/2) (q_t - q_{t-1})^2.
    """

    def __init__(self, a, b, c, phi, sigma, gamma, T,
                 q_min=0.0, q_max=4.0, seed=0):
        self.a, self.b, self.c, self.phi = a, b, c, phi
        self.sigma, self.gamma, self.T = sigma, gamma, T
        self.q_min, self.q_max = q_min, q_max
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.q_prev = a / (2 * (b + c / 2))
        self.p_prev = a - b * self.q_prev

    def reset(self):
        self.t = 0
        self.q_prev = self.a / (2 * (self.b + self.c / 2))
        self.p_prev = self.a - self.b * self.q_prev
        return np.array([self.q_prev, self.p_prev], dtype=np.float64)

    def step(self, q_t):
        q_t = float(np.clip(q_t, self.q_min, self.q_max))
        eps = self.rng.normal(0.0, self.sigma)
        p_t = self.a - self.b * q_t + eps
        r_t = (p_t * q_t
               - 0.5 * self.c * q_t ** 2
               - 0.5 * self.phi * (q_t - self.q_prev) ** 2)
        self.q_prev, self.p_prev = q_t, p_t
        self.t += 1
        done = self.t >= self.T
        return np.array([self.q_prev, self.p_prev]), r_t, done, {}


def solve_oracle_lq(a, b, c, phi, gamma, max_iter=2000, tol=1e-12):
    """Solve LQ-Bellman for cobweb via quadratic-form fixed-point iteration.

    V(q_prev) = P q_prev^2 + R q_prev + S
    Optimal policy:   q* = K0 + Kq * q_prev
    with   D = 2(A_c - gamma P),   A_c = b + c/2 + phi/2,
           Kq = phi / D,           K0 = (a + gamma R) / D.

    Update rules (matching coefficients of T V = V):
        P' = phi^2 / (2 D) - phi / 2
        R' = (a + gamma R) phi / D
        S' = (a + gamma R)^2 / (2 D) + gamma S
    Steady state: S = (a + gamma R)^2 / (2 D (1 - gamma)).

    Returns dict with P, R, S, K0, Kq, A_c, D, n_iter, converged.
    """
    A_c = b + 0.5 * c + 0.5 * phi
    P, R, S = 0.0, 0.0, 0.0
    converged = False
    for it in range(max_iter):
        D = 2.0 * (A_c - gamma * P)
        if D <= 0:
            raise ValueError(
                f"Riccati update lost positive definiteness: D={D:.4g} "
                f"at iter {it}, P={P:.4g}, gamma={gamma}, A_c={A_c}"
            )
        P_new = phi ** 2 / (2 * D) - 0.5 * phi
        X = a + gamma * R
        R_new = X * phi / D
        S_new = X ** 2 / (2 * D) + gamma * S
        if (abs(P_new - P) + abs(R_new - R) + abs(S_new - S)) < tol:
            P, R, S = P_new, R_new, S_new
            converged = True
            break
        P, R, S = P_new, R_new, S_new
    D_final = 2.0 * (A_c - gamma * P)
    return dict(P=P, R=R, S=S,
                K0=(a + gamma * R) / D_final,
                Kq=phi / D_final,
                A_c=A_c, D=D_final,
                n_iter=it + 1, converged=converged)


def oracle_policy(q_prev, lq):
    """Apply the oracle LQ feedback rule given solved coefficients lq."""
    return lq['K0'] + lq['Kq'] * q_prev


def expected_reward(q_t, q_prev, a, b, c, phi):
    """Expected one-step reward given action q_t and lag q_prev (E[eps] = 0)."""
    return a * q_t - b * q_t ** 2 - 0.5 * c * q_t ** 2 - 0.5 * phi * (q_t - q_prev) ** 2


if __name__ == "__main__":
    # Smoke test: oracle Riccati against numerical Bellman iteration on a grid.
    a, b, c, phi, gamma = 4.0, 0.5, 1.0, 0.2, 0.95
    lq = solve_oracle_lq(a, b, c, phi, gamma)
    print(f"Closed-form solve: converged={lq['converged']} in {lq['n_iter']} iters")
    print(f"  P = {lq['P']:.6f}, R = {lq['R']:.6f}, S = {lq['S']:.6f}")
    print(f"  K0 = {lq['K0']:.6f}, Kq = {lq['Kq']:.6f}")

    # Numerical Bellman iteration on a fine grid as independent validation.
    qg = np.linspace(0.0, 4.0, 401)
    V_grid = np.zeros_like(qg)
    for it in range(5000):
        V_new = np.empty_like(V_grid)
        for i, qp in enumerate(qg):
            # Inner max over q
            r_vec = (a * qg - 0.5 * c * qg ** 2 - 0.5 * phi * (qg - qp) ** 2
                     - b * qg ** 2)
            obj = r_vec + gamma * V_grid
            V_new[i] = obj.max()
        if np.max(np.abs(V_new - V_grid)) < 1e-7:
            print(f"  Grid Bellman: converged at iter {it}")
            V_grid = V_new
            break
        V_grid = V_new
    # Compare closed-form V at sample points to grid V
    test_points = np.linspace(0.5, 3.5, 5)
    print("\n  q_prev    V_closed_form    V_grid    diff")
    for qp in test_points:
        v_cf = lq['P'] * qp ** 2 + lq['R'] * qp + lq['S']
        v_gd = np.interp(qp, qg, V_grid)
        print(f"  {qp:.3f}    {v_cf:>12.6f}    {v_gd:>10.6f}    {v_cf - v_gd:+.4e}")
