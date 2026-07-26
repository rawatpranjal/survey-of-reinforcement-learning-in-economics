# The Engine: the book-wide running example
# Shared module. A bus-engine replacement MDP at two mileage grades (the canonical
# instance every chapter uses), the K-grade grid, and the +EV and +U variants, with exact
# float and exact rational solvers. No plotting, no file outputs; chapter scripts import
# from here. The canonical numbers are pinned by tests/test_engine.py in exact rational
# arithmetic: V* = (155/29, 125/29), action gaps 30/29 and 67/290.

from fractions import Fraction

import numpy as np

# State 0 is the low-mileage (good) grade, state 1 the high-mileage (worn) grade.
# Appendix A narrates the same instance as good/worn; the indices agree.
LOW, HIGH = 0, 1
KEEP, REPLACE = 0, 1
STATE_NAMES = ["low", "high"]
ACTION_NAMES = ["keep", "replace"]

GAMMA = 0.9
GAMMA_FRAC = Fraction(9, 10)

# The canonical primitives: r(s, keep) is per-period output at mileage grade s,
# replace_cost is the replacement cost RC, degrade_prob is the mileage transition.
ENGINE_PARAMS = {
    "gamma": GAMMA,
    "r_keep_low": 1.0,
    "r_keep_high": 0.2,
    "replace_cost": 0.5,
    "degrade_prob": 0.5,
}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def build_mdp(r_keep_low=1.0, r_keep_high=0.2, replace_cost=0.5, degrade_prob=0.5):
    """Return (P, r) with P[s, a, s'] and r[s, a] for the canonical two-grade engine.

    Keeping a low-mileage engine earns r_keep_low and degrades it with probability
    degrade_prob. Keeping a high-mileage engine earns r_keep_high and leaves it there.
    Replacing costs replace_cost from either grade and returns the engine to low mileage.
    """
    p = degrade_prob
    P = np.zeros((2, 2, 2))
    P[LOW, KEEP] = [1.0 - p, p]
    P[LOW, REPLACE] = [1.0, 0.0]
    P[HIGH, KEEP] = [0.0, 1.0]
    P[HIGH, REPLACE] = [1.0, 0.0]
    r = np.array(
        [[r_keep_low, -replace_cost], [r_keep_high, -replace_cost]], dtype=float
    )
    return P, r


def build_mdp_grid(
    K, r_keep_low=1.0, r_keep_high=0.2, replace_cost=0.5, degrade_prob=0.5
):
    """The K-grade engine: mileage grades 0..K-1, keep degrades one grade with
    probability degrade_prob (the top grade is absorbing under keep), replace resets to
    grade 0 at replace_cost. Keep-reward interpolates linearly from r_keep_low at grade 0
    to r_keep_high at grade K-1. K = 2 reduces exactly to build_mdp."""
    P = np.zeros((K, 2, K))
    r = np.zeros((K, 2))
    keep_rewards = np.linspace(r_keep_low, r_keep_high, K)
    for s in range(K):
        if s == K - 1:
            P[s, KEEP, s] = 1.0
        else:
            P[s, KEEP, s] = 1.0 - degrade_prob
            P[s, KEEP, s + 1] = degrade_prob
        P[s, REPLACE, 0] = 1.0
        r[s, KEEP] = keep_rewards[s]
        r[s, REPLACE] = -replace_cost
    return P, r


def build_mdp_confounded(delta, q=0.5, **kwargs):
    """The +U variant (ch10): a binary latent U shifts the degradation probability to
    degrade_prob - delta when U = 0 and degrade_prob + delta when U = 1, with
    P(U = 1) = q. Returns (P_by_u, r, q) where P_by_u[u] is a (2, 2, 2) kernel.
    delta = 0 gives the canonical engine twice."""
    base = ENGINE_PARAMS["degrade_prob"]
    p0 = kwargs.pop("degrade_prob", base)
    P_lo, r = build_mdp(degrade_prob=p0 - delta, **kwargs)
    P_hi, _ = build_mdp(degrade_prob=p0 + delta, **kwargs)
    return np.stack([P_lo, P_hi]), r, q


# ---------------------------------------------------------------------------
# Policies and induced chains
# ---------------------------------------------------------------------------


def policy_matrices(P, r, policy):
    """P^pi and r^pi for a deterministic policy given as an action per state."""
    n = P.shape[0]
    P_pi = np.array([P[s, policy[s]] for s in range(n)])
    r_pi = np.array([r[s, policy[s]] for s in range(n)])
    return P_pi, r_pi


def policy_kernel(P, r, b):
    """P^pi and r^pi for an arbitrary stochastic policy b[s, a] = pi(a | s)."""
    P_pi = np.einsum("sa,sat->st", b, P)
    r_pi = np.einsum("sa,sa->s", b, r)
    return P_pi, r_pi


def stochastic_policy_matrices(P, r, keep_prob):
    """P^mu, r^mu and the policy matrix for the logging policy that keeps with
    probability keep_prob in every state."""
    b = np.array([[keep_prob, 1.0 - keep_prob]] * P.shape[0])
    P_mu = np.einsum("sa,sat->st", b, P)
    r_mu = np.einsum("sa,sa->s", b, r)
    return P_mu, r_mu, b


def policy_from_logits(theta):
    """The book's fixed policy parameterization: one logit per state,
    pi(replace | s) = sigmoid(theta_s). Returns b with rows [pi_keep, pi_replace].
    The overparameterized softmax (one logit per state-action pair) has a rank-one
    per-state Fisher block with null direction (1, 1); this parameterization is the
    redundancy-free one under which the Fisher matrix is diagonal and invertible."""
    p_replace = 1.0 / (1.0 + np.exp(-np.asarray(theta, dtype=float)))
    return np.stack([1.0 - p_replace, p_replace], axis=1)


# ---------------------------------------------------------------------------
# Exact float solvers
# ---------------------------------------------------------------------------


def exact_value(P_pi, r_pi, gamma):
    """The resolvent solve V = (I - gamma P^pi)^{-1} r^pi."""
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


def bellman_optimality(P, r, V, gamma):
    """(T*V)(s) and the greedy action per state."""
    q = r + gamma * P @ V
    return q.max(axis=1), q.argmax(axis=1), q


def solve_optimal(P, r, gamma, tol=1e-12, max_iter=10000):
    V = np.zeros(P.shape[0])
    for _ in range(max_iter):
        V_new, _, _ = bellman_optimality(P, r, V, gamma)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    _, greedy, q = bellman_optimality(P, r, V, gamma)
    return V, greedy, q


def q_values(P, r, V, gamma):
    """Q(s, a) = r(s, a) + gamma sum_s' P(s' | s, a) V(s')."""
    return r + gamma * P @ V


def stationary_distribution(P_pi):
    """Left eigenvector of P^pi for eigenvalue one, normalized to a probability vector."""
    w, vl = np.linalg.eig(P_pi.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    d = np.real(vl[:, idx])
    d = d / d.sum()
    return d


def discounted_occupancy(P_pi, gamma, nu):
    """d^pi_nu = (1 - gamma) nu (I - gamma P^pi)^{-1}."""
    return (1.0 - gamma) * nu @ np.linalg.inv(np.eye(P_pi.shape[0]) - gamma * P_pi)


def projected_modulus(phi, d, P_pi, gamma):
    """Modulus of Pi_d T^pi restricted to span(phi), a scalar because phi has one column.

    Pi_d T^pi (phi theta) = phi theta' with theta' = gamma (phi^T D P^pi phi / phi^T D phi) theta
    plus a constant, so the contraction modulus is the absolute value of that coefficient.
    """
    D = np.diag(d)
    num = phi @ D @ P_pi @ phi
    den = phi @ D @ phi
    return abs(gamma * num / den), gamma * num / den


# ---------------------------------------------------------------------------
# Policy gradient objects (used from ch03 on)
# ---------------------------------------------------------------------------


def policy_performance(P, r, gamma, b, nu):
    """J(pi) = nu . V^pi, with V^pi and the unnormalized occupancy
    rho(s) = sum_t gamma^t P(s_t = s) = nu (I - gamma P^pi)^{-1}. The discounted
    occupancy measure is (1 - gamma) rho."""
    P_b, r_b = policy_kernel(P, r, b)
    V = exact_value(P_b, r_b, gamma)
    rho = nu @ np.linalg.inv(np.eye(P.shape[0]) - gamma * P_b)
    return float(nu @ V), V, rho


def policy_gradient(P, r, gamma, theta, nu):
    """Exact gradient of J(theta) = nu . V^{pi_theta} under one logit per state:
    grad_s = rho(s) pi_keep(s) pi_replace(s) (Q(s, replace) - Q(s, keep)).
    Returns (grad, aux) with the pieces (rho, Q, gap, b) for reuse."""
    b = policy_from_logits(theta)
    _, V, rho = policy_performance(P, r, gamma, b, nu)
    Q = q_values(P, r, V, gamma)
    gap = Q[:, REPLACE] - Q[:, KEEP]
    grad = rho * b[:, KEEP] * b[:, REPLACE] * gap
    return grad, {"rho": rho, "Q": Q, "gap": gap, "b": b, "V": V}


def fisher_matrix(theta, rho):
    """F = diag(rho(s) pi_keep(s) pi_replace(s)) under the one-logit-per-state
    parameterization. Diagonal, and genuinely invertible at any interior policy with a
    reaching occupancy; the determinant is the product of the diagonal."""
    b = policy_from_logits(theta)
    return np.diag(rho * b[:, KEEP] * b[:, REPLACE])


def natural_gradient(P, r, gamma, theta, nu):
    """F^{-1} grad J, inverting the diagonal Fisher entrywise (never a pseudo-inverse).
    Equals the action-value gap Q(s, replace) - Q(s, keep) exactly."""
    grad, aux = policy_gradient(P, r, gamma, theta, nu)
    F = fisher_matrix(theta, aux["rho"])
    diag = np.diag(F)
    if np.any(diag <= 0.0):
        raise ValueError(
            "Fisher matrix is singular: interior policy and reaching occupancy required"
        )
    return grad / diag, aux


# ---------------------------------------------------------------------------
# The +EV variant (ch05): Type-I extreme-value shocks, the logit fixed point
# ---------------------------------------------------------------------------


def solve_ev(P, r, gamma, sigma, tol=1e-12, max_iter=100000):
    """Fixed point of the smoothed Bellman operator with iid T1EV(scale sigma) shocks.

    Returns (W, v, ccp): the integrated (ex-ante) value W(s), the choice-specific value
    v(s, a) = r(s, a) + gamma sum P(s'|s,a) W(s'), and the logit conditional choice
    probabilities ccp(s, a) = exp(v/sigma) / sum_a' exp(v/sigma). The Euler constant is
    omitted; it shifts W by sigma * gamma_E / (1 - gamma) and cancels in the CCPs.
    As sigma -> 0, W -> V* and the CCPs concentrate on the greedy actions."""
    W = np.zeros(P.shape[0])
    for _ in range(max_iter):
        v = r + gamma * P @ W
        m = v.max(axis=1)
        W_new = m + sigma * np.log(np.exp((v - m[:, None]) / sigma).sum(axis=1))
        if np.max(np.abs(W_new - W)) < tol:
            W = W_new
            break
        W = W_new
    v = r + gamma * P @ W
    z = np.exp((v - v.max(axis=1)[:, None]) / sigma)
    ccp = z / z.sum(axis=1, keepdims=True)
    return W, v, ccp


# ---------------------------------------------------------------------------
# Exact rational solvers (fractions.Fraction), the oracle layer
# ---------------------------------------------------------------------------


def build_mdp_grid_frac(
    K=2,
    r_keep_low=Fraction(1),
    r_keep_high=Fraction(1, 5),
    replace_cost=Fraction(1, 2),
    degrade_prob=Fraction(1, 2),
):
    """The K-grade engine with Fraction entries. K = 2 is the canonical instance."""
    P = [[[Fraction(0)] * K for _ in range(2)] for _ in range(K)]
    r = [[Fraction(0)] * 2 for _ in range(K)]
    for s in range(K):
        if s == K - 1:
            P[s][KEEP][s] = Fraction(1)
        else:
            P[s][KEEP][s] = 1 - degrade_prob
            P[s][KEEP][s + 1] = degrade_prob
        P[s][REPLACE][0] = Fraction(1)
        r[s][KEEP] = r_keep_low + (r_keep_high - r_keep_low) * Fraction(s, K - 1)
        r[s][REPLACE] = -replace_cost
    return P, r


def linsolve_frac(A, b):
    """Solve A x = b by Gaussian elimination over Fractions. A: list of rows, b: list."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = next(i for i in range(col, n) if M[i][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        inv = Fraction(1) / M[col][col]
        M[col] = [x * inv for x in M[col]]
        for i in range(n):
            if i != col and M[i][col] != 0:
                f = M[i][col]
                M[i] = [M[i][j] - f * M[col][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def exact_value_frac(P, r, policy, gamma=GAMMA_FRAC):
    """V^pi over Fractions for a deterministic policy: solve (I - gamma P^pi) V = r^pi."""
    n = len(r)
    A = [
        [
            (Fraction(1) if i == j else Fraction(0)) - gamma * P[i][policy[i]][j]
            for j in range(n)
        ]
        for i in range(n)
    ]
    b = [r[i][policy[i]] for i in range(n)]
    return linsolve_frac(A, b)


def q_values_frac(P, r, V, gamma=GAMMA_FRAC):
    n = len(r)
    return [
        [r[s][a] + gamma * sum(P[s][a][t] * V[t] for t in range(n)) for a in range(2)]
        for s in range(n)
    ]


def solve_optimal_frac(P, r, gamma=GAMMA_FRAC):
    """Howard policy iteration over Fractions: exact V*, greedy policy and Q*.
    Terminates finitely because every comparison is exact."""
    n = len(r)
    policy = [KEEP] * n
    while True:
        V = exact_value_frac(P, r, policy, gamma)
        Q = q_values_frac(P, r, V, gamma)
        new_policy = [max(range(2), key=lambda a: Q[s][a]) for s in range(n)]
        if new_policy == policy:
            return V, policy, Q
        policy = new_policy


def stationary_frac(P_pi):
    """Stationary distribution of a Fraction transition matrix: solve d (P - I) = 0
    with sum d = 1, by replacing one equation with the normalization."""
    n = len(P_pi)
    A = [
        [P_pi[j][i] - (Fraction(1) if i == j else Fraction(0)) for j in range(n)]
        for i in range(n)
    ]
    A[n - 1] = [Fraction(1)] * n
    b = [Fraction(0)] * (n - 1) + [Fraction(1)]
    return linsolve_frac(A, b)


def occupancy_frac(P_pi, nu, gamma=GAMMA_FRAC):
    """Discounted occupancy (1 - gamma) nu (I - gamma P^pi)^{-1} over Fractions,
    via the transposed solve (I - gamma P^pi)^T x = nu."""
    n = len(P_pi)
    A = [
        [
            (Fraction(1) if i == j else Fraction(0)) - gamma * P_pi[j][i]
            for j in range(n)
        ]
        for i in range(n)
    ]
    x = linsolve_frac(A, list(nu))
    return [(1 - gamma) * xi for xi in x]
