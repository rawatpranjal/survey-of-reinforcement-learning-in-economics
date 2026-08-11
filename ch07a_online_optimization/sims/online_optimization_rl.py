#!/usr/bin/env python3
"""Online optimization and RL: six reproducible paired experiments.

The script uses expected occupancies whenever the transition kernel is known and
separate random-number streams for environment and algorithm randomization when
feedback is sampled.  It produces six publication figures and a short LaTeX
diagnostics table next to this file.
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import COLORS, FIG_DOUBLE, apply_style
from sims.engine import build_mdp

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(__file__)
S, A, H = 3, 2, 12
RHO = np.array([1.0, 0.0, 0.0])
P_TRUE = np.array(
    [
        [[0.90, 0.10, 0.00], [0.35, 0.55, 0.10]],
        [[0.35, 0.55, 0.10], [0.05, 0.45, 0.50]],
        [[0.00, 0.45, 0.55], [0.00, 0.10, 0.90]],
    ]
)
D_COST = np.array([0.0, 0.15, 0.90])
M_VALUE = np.array([1.0, 0.60, 0.15])
PROMO_COST = 0.25

# Cold-storage maintenance experiment.  States are normal, strained, and
# failing; actions are monitor and service.  Service has an immediate cost but
# improves the distribution of the next equipment state.
P_STORAGE = np.array(
    [
        [[0.75, 0.25, 0.00], [0.95, 0.05, 0.00]],
        [[0.05, 0.65, 0.30], [0.60, 0.38, 0.02]],
        [[0.00, 0.05, 0.95], [0.45, 0.45, 0.10]],
    ]
)
STORAGE_LOSS = np.array([[0.00, 0.24], [0.18, 0.34], [1.00, 0.55]])


def losses(value):
    raw = D_COST[:, None] + PROMO_COST * np.arange(A)[None, :] - value * (
        np.arange(A)[None, :] * M_VALUE[:, None]
    )
    # A single affine normalization, fixed over both demand regimes.
    return (raw + 1.0) / 2.0


def episode_values(k):
    return np.where((np.arange(k) % 100) < 50, 1.25, 0.55)


def flow_matrix(P, horizon=H, rho=RHO):
    n = horizon * S * A
    aeq = np.zeros((horizon * S, n))
    beq = np.zeros(horizon * S)
    for h in range(horizon):
        for s in range(S):
            row = h * S + s
            for a in range(A):
                aeq[row, (h * S + s) * A + a] = 1.0
            if h == 0:
                beq[row] = rho[s]
            else:
                for sp in range(S):
                    for a in range(A):
                        aeq[row, ((h - 1) * S + sp) * A + a] -= P[sp, a, s]
    return aeq, beq


def occupancy(P, policy, horizon=H, rho=RHO):
    q = np.zeros((horizon, S, A))
    state = rho.copy()
    for h in range(horizon):
        q[h] = state[:, None] * policy[h]
        state = np.einsum("sa,san->n", q[h], P)
    return q


def policy_from_q(q):
    denom = q.sum(axis=2, keepdims=True)
    return np.divide(q, denom, out=np.full_like(q, 1.0 / A), where=denom > 1e-14)


def dynamic_program(P, loss):
    value = np.zeros(S)
    policy = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        qval = loss + np.einsum("san,n->sa", P, value)
        acts = np.argmin(qval, axis=1)
        policy[h, np.arange(S), acts] = 1.0
        value = qval[np.arange(S), acts]
    return policy, float(RHO @ value)


def policy_loss_by_bellman(P, policy, loss):
    """Evaluate a finite-horizon policy by the ordinary MDP recursion."""
    value = np.zeros(S)
    for h in range(H - 1, -1, -1):
        continuation = np.einsum("san,n->sa", P, value)
        value = np.sum(policy[h] * (loss + continuation), axis=1)
    return float(RHO @ value)


def oreps_projection(q0, loss, eta, aeq, beq):
    full0 = q0.ravel()
    active = full0 > 1e-14
    x0 = full0[active]
    ell_full = np.broadcast_to(loss, (H, S, A)).ravel()
    ell = ell_full[active]
    active_eq = aeq[:, active]
    # Flow conservation has one redundant mass equation after each first layer.
    # Select an independent row basis so SLSQP receives a full-rank Jacobian.
    independent = []
    rank = 0
    for row in range(active_eq.shape[0]):
        candidate = active_eq[independent + [row]]
        new_rank = np.linalg.matrix_rank(candidate, tol=1e-11)
        if new_rank > rank:
            independent.append(row)
            rank = new_rank
    reduced_eq = active_eq[independent]
    reduced_b = beq[independent]

    def objective(x):
        return eta * float(ell @ x) + float(np.sum(x * np.log(x / x0) - x + x0))

    def gradient(x):
        return eta * ell + np.log(x / x0)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(1e-12, None)] * len(x0),
        constraints=[LinearConstraint(reduced_eq, reduced_b, reduced_b)],
        options={"ftol": 1e-10, "maxiter": 600},
    )
    if not result.success:
        raise RuntimeError(f"O-REPS projection failed: {result.message}")
    full = np.zeros_like(full0)
    full[active] = result.x
    residual = float(np.max(np.abs(aeq @ full - beq)))
    if residual > 2e-7:
        raise AssertionError(f"occupancy flow residual {residual:.3e}")
    return full.reshape(H, S, A), residual


def sim_stability():
    horizons = np.array([50, 100, 200, 500, 1000, 2000])
    rows = {name: [] for name in ("FTL", "OGD", "Hedge")}
    for T in horizons:
        seq = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), (T // 2 + 1, 1))[:T]
        benchmark = seq.sum(axis=0).min()
        cumulative = np.zeros(2)
        ftl_loss = 0.0
        p_ogd = np.array([0.5, 0.5])
        ogd_loss = 0.0
        weights = np.ones(2)
        hedge_loss = 0.0
        eta = np.sqrt(2.0 / T)
        for ell in seq:
            act = int(np.argmin(cumulative))
            ftl_loss += ell[act]
            cumulative += ell
            ogd_loss += p_ogd @ ell
            z = p_ogd - eta * ell
            # Exact projection on the two-dimensional simplex.
            p_ogd = np.maximum(z - (z.sum() - 1.0) / 2.0, 0.0)
            p_ogd /= p_ogd.sum()
            p = weights / weights.sum()
            hedge_loss += p @ ell
            weights *= np.exp(-eta * ell)
        rows["FTL"].append(ftl_loss - benchmark)
        rows["OGD"].append(ogd_loss - benchmark)
        rows["Hedge"].append(hedge_loss - benchmark)
    assert abs(rows["FTL"][-1] - horizons[-1] / 2) < 1e-10
    return horizons, {k: np.asarray(v) for k, v in rows.items()}


def sim_one_step_equivalence(horizon=2000):
    """Verify the OCO and one-state, one-step MDP representations round by round."""
    seq = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), (horizon // 2 + 1, 1))[:horizon]
    p_oco = np.array([0.5, 0.5])
    p_mdp = np.array([0.5, 0.5])
    eta = np.sqrt(2.0 / horizon)
    max_policy_residual = 0.0
    max_value_residual = 0.0
    for ell in seq:
        # In the one-step MDP, the state-action occupancy is the action
        # distribution itself.  Expected MDP loss and the OCO inner product
        # are therefore the same scalar.
        max_value_residual = max(
            max_value_residual, abs(float(p_oco @ ell) - float(p_mdp @ ell))
        )
        z_oco = p_oco - eta * ell
        z_mdp = p_mdp - eta * ell
        p_oco = np.maximum(z_oco - (z_oco.sum() - 1.0) / 2.0, 0.0)
        p_mdp = np.maximum(z_mdp - (z_mdp.sum() - 1.0) / 2.0, 0.0)
        p_oco /= p_oco.sum()
        p_mdp /= p_mdp.sum()
        max_policy_residual = max(
            max_policy_residual, float(np.max(np.abs(p_oco - p_mdp)))
        )
    return max_policy_residual, max_value_residual


def sim_feedback(reps=200, horizon=1500):
    rng_env = np.random.default_rng(711)
    rng_alg = np.random.default_rng(712)
    results = {}
    for arms in (5, 10):
        t = np.arange(horizon)
        base = 0.45 + 0.18 * np.sin(t[:, None] / 31.0 + np.arange(arms)[None, :])
        bias = np.linspace(0.0, 0.16, arms)[None, :]
        noise = rng_env.normal(0.0, 0.025, size=(horizon, arms))
        loss_seq = np.clip(base + bias + noise, 0.0, 1.0)
        best_curve = np.minimum.accumulate(np.cumsum(loss_seq, axis=0), axis=1)[:, -1]
        eta_h = np.sqrt(2 * np.log(arms) / horizon)
        w = np.ones(arms)
        hedge = np.zeros(horizon)
        for i, ell in enumerate(loss_seq):
            p = w / w.sum()
            hedge[i] = p @ ell
            w *= np.exp(-eta_h * ell)
        hedge_regret = np.cumsum(hedge) - best_curve

        exp3_regrets = np.zeros((reps, horizon))
        est_variance = np.zeros((reps, horizon))
        gamma = min(0.5, np.sqrt(arms * np.log(arms) / horizon))
        eta_e = gamma / arms
        for rep in range(reps):
            w = np.ones(arms)
            incurred = np.zeros(horizon)
            for i, ell in enumerate(loss_seq):
                p = (1.0 - gamma) * w / w.sum() + gamma / arms
                action = rng_alg.choice(arms, p=p)
                estimate = np.zeros(arms)
                estimate[action] = ell[action] / p[action]
                incurred[i] = ell[action]
                est_variance[rep, i] = np.mean((estimate - ell) ** 2)
                w *= np.exp(-eta_e * estimate)
            exp3_regrets[rep] = np.cumsum(incurred) - best_curve
        results[arms] = {
            "hedge": hedge_regret,
            "exp3_mean": exp3_regrets.mean(axis=0),
            "exp3_se": exp3_regrets.std(axis=0, ddof=1) / np.sqrt(reps),
            "variance": est_variance.mean(axis=0),
        }
    return results


def sim_soil_policy_regret(horizon=120):
    """Compare realized-state external regret with a counterfactual RL comparator."""
    # States: fertile, depleted. Actions: harvest, rest.
    reward = np.array([[1.0, 0.2], [0.0, 0.2]])
    transition = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    policies = {
        "Always harvest": np.array([0, 0]),
        "Always rest": np.array([1, 1]),
        "Rotate": np.array([0, 1]),
        "Rest when fertile": np.array([1, 0]),
    }

    learner_policy = policies["Always harvest"]
    learner_states = np.zeros(horizon + 1, dtype=int)
    learner_rewards = np.zeros(horizon)
    state = 0
    for t in range(horizon):
        action = learner_policy[state]
        learner_rewards[t] = reward[state, action]
        state = int(np.argmax(transition[state, action]))
        learner_states[t + 1] = state

    realized_comparator_rewards = {}
    counterfactual_rewards = {}
    counterfactual_states = {}
    for name, policy in policies.items():
        realized_comparator_rewards[name] = np.array(
            [reward[learner_states[t], policy[learner_states[t]]] for t in range(horizon)]
        )
        rewards = np.zeros(horizon)
        states = np.zeros(horizon + 1, dtype=int)
        state = 0
        for t in range(horizon):
            action = policy[state]
            rewards[t] = reward[state, action]
            state = int(np.argmax(transition[state, action]))
            states[t + 1] = state
        counterfactual_rewards[name] = rewards
        counterfactual_states[name] = states

    learner_cumulative = np.cumsum(learner_rewards)
    external_benchmark = np.maximum.reduce(
        [np.cumsum(v) for v in realized_comparator_rewards.values()]
    )
    policy_totals = {name: float(values.sum()) for name, values in counterfactual_rewards.items()}
    best_policy = max(policy_totals, key=policy_totals.get)
    policy_benchmark = np.cumsum(counterfactual_rewards[best_policy])
    return {
        "external": external_benchmark - learner_cumulative,
        "policy": policy_benchmark - learner_cumulative,
        "learner_states": learner_states,
        "comparator_states": counterfactual_states[best_policy],
        "best_policy": best_policy,
        "reward": reward,
        "transition": transition,
    }


def sim_known_mdp(K=300):
    aeq, beq = flow_matrix(P_TRUE)
    uniform = np.full((H, S, A), 0.5)
    q_oreps = occupancy(P_TRUE, uniform)
    eta = 0.12 / np.sqrt(H)
    weights_blind = np.ones(A)
    weights_state = np.ones((S, A))
    cumulative = {k: np.zeros(K) for k in ("Myopic", "State-blind", "Per-state Hedge", "O-REPS")}
    state_shares = {k: np.zeros(S) for k in cumulative}
    promo = {k: np.zeros(S) for k in cumulative}
    values = episode_values(K)
    avg_loss = np.mean([losses(v) for v in values], axis=0)
    hindsight_policy, hindsight_episode = dynamic_program(P_TRUE, avg_loss)
    benchmark = np.arange(1, K + 1) * hindsight_episode
    max_residual = 0.0
    max_representation_residual = 0.0
    for k, value in enumerate(values):
        ell = losses(value)
        myopic = np.zeros((H, S, A))
        acts = np.argmin(ell, axis=1)
        myopic[:, np.arange(S), acts] = 1.0
        p_blind = weights_blind / weights_blind.sum()
        blind = np.broadcast_to(p_blind, (H, S, A)).copy()
        p_state = weights_state / weights_state.sum(axis=1, keepdims=True)
        per_state = np.broadcast_to(p_state, (H, S, A)).copy()
        policies = {
            "Myopic": myopic,
            "State-blind": blind,
            "Per-state Hedge": per_state,
            "O-REPS": policy_from_q(q_oreps),
        }
        for name, policy in policies.items():
            q = occupancy(P_TRUE, policy)
            episode_loss = float(np.sum(q * ell[None, :, :]))
            bellman_loss = policy_loss_by_bellman(P_TRUE, policy, ell)
            max_representation_residual = max(
                max_representation_residual, abs(episode_loss - bellman_loss)
            )
            cumulative[name][k] = episode_loss + (cumulative[name][k - 1] if k else 0.0)
            state_shares[name] += q.sum(axis=(0, 2))
            promo[name] += q[:, :, 1].sum(axis=0)
        weights_blind *= np.exp(-0.18 * ell.mean(axis=0))
        weights_state *= np.exp(-0.18 * ell)
        q_oreps, residual = oreps_projection(q_oreps, ell, eta, aeq, beq)
        max_residual = max(max_residual, residual)
    for name in state_shares:
        state_shares[name] /= state_shares[name].sum()
        visits = state_shares[name] * K * H
        promo[name] = np.divide(promo[name], visits, out=np.zeros(S), where=visits > 0)
        cumulative[name] -= benchmark
    q_star = occupancy(P_TRUE, hindsight_policy)
    return (
        cumulative,
        state_shares,
        promo,
        q_star,
        max_residual,
        max_representation_residual,
    )


def project_simplex(x):
    order = np.sort(x)[::-1]
    cssv = np.cumsum(order) - 1.0
    rho = np.nonzero(order - cssv / (np.arange(len(x)) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(x - theta, 0.0)


def discounted_optimal(P, reward, gamma, rho):
    v = np.zeros(S)
    for _ in range(10000):
        q = reward + gamma * np.einsum("san,n->sa", P, v)
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < 1e-12:
            break
        v = v_new
    return v_new, np.argmax(q, axis=1)


def sim_bellman(iterations=5000):
    gamma = 0.90
    transition, reward = build_mdp()
    ns, na = reward.shape
    rho = np.array([1.0, 0.0])
    v_star = np.zeros(ns)
    for _ in range(10000):
        q_star = reward + gamma * np.einsum("san,n->sa", transition, v_star)
        v_new = q_star.max(axis=1)
        if np.max(np.abs(v_new - v_star)) < 1e-12:
            break
        v_star = v_new
    v_star = v_new
    B = 1.0 / (1.0 - gamma)
    v = np.zeros(ns)
    mu = np.full(ns * na, 1.0 / (ns * na))
    v_sum = np.zeros(ns)
    mu_sum = np.zeros(ns * na)
    played = 0.0
    grad_v_sum = np.zeros(ns)
    adv_mu_sum = np.zeros(ns * na)
    checkpoints = np.unique(np.geomspace(10, iterations, 90).astype(int))
    out = {"n": [], "gap": [], "policy": [], "regret": []}
    # The two domains have different diameters: the value box is wide while the
    # occupancy player lives on a simplex.  Tune their standard no-regret steps
    # separately for the fixed iteration budget.
    eta_v = 0.10
    eta_mu = 0.015
    for n in range(1, iterations + 1):
        adv = (reward + gamma * np.einsum("san,n->sa", transition, v) - v[:, None]).ravel()
        grad_v = (1.0 - gamma) * rho.copy()
        for s in range(ns):
            grad_v[s] += gamma * np.sum(mu.reshape(ns, na) * transition[:, :, s]) - mu.reshape(ns, na)[s].sum()
        current = (1.0 - gamma) * rho @ v + mu @ adv
        played += current
        grad_v_sum += grad_v
        adv_mu_sum += adv
        v = np.clip(v - eta_v * grad_v, -B, B)
        logits = np.log(np.maximum(mu, 1e-300)) + eta_mu * adv
        logits -= logits.max()
        mu = np.exp(logits)
        mu /= mu.sum()
        v_sum += v
        mu_sum += mu
        if n in checkpoints:
            vb = v_sum / n
            mub = (mu_sum / n).reshape(ns, na)
            max_l = (1.0 - gamma) * rho @ vb + np.max(
                (reward + gamma * np.einsum("san,n->sa", transition, vb) - vb[:, None])
            )
            coeff = (1.0 - gamma) * rho.copy()
            for s in range(ns):
                coeff[s] += gamma * np.sum(mub * transition[:, :, s]) - mub[s].sum()
            min_l = float(mub.ravel() @ reward.ravel() - B * np.sum(np.abs(coeff)))
            gap = max_l - min_l
            policy = np.divide(mub, mub.sum(axis=1, keepdims=True), out=np.full_like(mub, 1.0 / na), where=mub.sum(axis=1, keepdims=True) > 0)
            ppi = np.einsum("sa,san->sn", policy, transition)
            rpi = np.sum(policy * reward, axis=1)
            vpi = np.linalg.solve(np.eye(ns) - gamma * ppi, rpi)
            policy_gap = float(rho @ (v_star - vpi))
            min_v_loss = -B * np.sum(np.abs(grad_v_sum))
            max_mu_gain = np.max(adv_mu_sum)
            regret = (played - min_v_loss + max_mu_gain - played) / n
            out["n"].append(n)
            out["gap"].append(gap / (1.0 - gamma))
            out["policy"].append(policy_gap)
            out["regret"].append(regret / (1.0 - gamma))
    result = {k: np.asarray(v) for k, v in out.items()}

    vi_iterations = np.arange(1, 181)
    vi_value = np.zeros(ns)
    vi_gap = np.zeros_like(vi_iterations, dtype=float)
    for i, _ in enumerate(vi_iterations):
        vi_q = reward + gamma * np.einsum("san,n->sa", transition, vi_value)
        vi_value = vi_q.max(axis=1)
        vi_gap[i] = max(float(rho @ (v_star - vi_value)), 1e-14)
    result["vi_n"] = vi_iterations
    result["vi_gap"] = vi_gap
    result["v_star"] = v_star
    return result


def plan_estimated(P, loss, bonus=None):
    optimistic = loss.copy()
    if bonus is not None:
        optimistic = optimistic - bonus
    return dynamic_program(P, optimistic)[0]


def learn_unknown(seed, optimistic, K=250):
    rng = np.random.default_rng(seed)
    counts = np.zeros((S, A, S))
    visits = np.zeros((S, A))
    regret = np.zeros(K)
    total = 0.0
    for k in range(K):
        ell = STORAGE_LOSS
        phat = np.divide(counts, visits[:, :, None], out=np.zeros_like(counts), where=visits[:, :, None] > 0)
        unseen = visits == 0
        phat[unseen] = np.eye(S)[np.where(unseen)[0]]
        bonus = None
        if optimistic:
            bonus = 0.55 * np.sqrt(np.log(K * S * A + 1.0) / np.maximum(visits, 1.0))
            bonus = np.broadcast_to(bonus[None, :, :], (H, S, A))
            # dynamic_program expects a stationary stage cost; use stage-varying planner below.
            vnext = np.zeros(S)
            policy = np.zeros((H, S, A))
            for h in range(H - 1, -1, -1):
                qv = ell - bonus[h] + np.einsum("san,n->sa", phat, vnext)
                acts = np.argmin(qv, axis=1)
                policy[h, np.arange(S), acts] = 1.0
                vnext = qv[np.arange(S), acts]
        else:
            policy = plan_estimated(phat, ell)
        oracle_policy, oracle_loss = dynamic_program(P_STORAGE, ell)
        del oracle_policy
        state = 0
        episode_loss = 0.0
        for h in range(H):
            action = rng.choice(A, p=policy[h, state])
            episode_loss += ell[state, action]
            next_state = rng.choice(S, p=P_STORAGE[state, action])
            counts[state, action, next_state] += 1
            visits[state, action] += 1
            state = next_state
        total += episode_loss - oracle_loss
        regret[k] = total
    return regret


def learn_known_storage(seed, K=250):
    """Known-transition benchmark on the same sampled cold-storage problem."""
    rng = np.random.default_rng(seed)
    policy, expected_loss = dynamic_program(P_STORAGE, STORAGE_LOSS)
    regret = np.zeros(K)
    total = 0.0
    for k in range(K):
        state = 0
        episode_loss = 0.0
        for h in range(H):
            action = rng.choice(A, p=policy[h, state])
            episode_loss += STORAGE_LOSS[state, action]
            state = rng.choice(S, p=P_STORAGE[state, action])
        total += episode_loss - expected_loss
        regret[k] = total
    return regret


def riverswim(seed, optimistic, episodes=450, horizon=20):
    rng = np.random.default_rng(seed)
    ns = 6
    counts = np.zeros((ns, 2, ns))
    visits = np.zeros((ns, 2))
    rewards = np.zeros((ns, 2))
    rewards[0, 0] = 0.02
    rewards[-1, 1] = 1.0
    reached = np.zeros(episodes)
    cumulative = 0
    for ep in range(episodes):
        phat = np.divide(counts, visits[:, :, None], out=np.zeros_like(counts), where=visits[:, :, None] > 0)
        for s in range(ns):
            for a in range(2):
                if visits[s, a] == 0:
                    phat[s, a, s] = 1.0
        value = np.zeros(ns)
        policies = []
        bonus = 0.25 * np.sqrt(np.log(episodes * ns * 2 + 1) / np.maximum(visits, 1.0)) if optimistic else 0.0
        for _ in range(horizon):
            q = rewards + bonus + np.einsum("san,n->sa", phat, value)
            act = np.argmax(q, axis=1)
            policies.append(act)
            value = q[np.arange(ns), act]
        state = 0
        for h in range(horizon):
            action = policies[horizon - h - 1][state]
            if action == 0:
                nxt = max(0, state - 1)
            else:
                u = rng.random()
                nxt = min(ns - 1, state + 1) if u < 0.35 else (max(0, state - 1) if u < 0.40 else state)
            counts[state, action, nxt] += 1
            visits[state, action] += 1
            state = nxt
            cumulative += int(state == ns - 1)
        reached[ep] = cumulative
    return reached


def sim_unknown(reps=100):
    known = np.vstack([learn_known_storage(7000 + i) for i in range(reps)])
    ce = np.vstack([learn_unknown(9000 + i, False) for i in range(reps)])
    opt = np.vstack([learn_unknown(12000 + i, True) for i in range(reps)])
    river_ce = np.vstack([riverswim(15000 + i, False) for i in range(reps)])
    river_opt = np.vstack([riverswim(18000 + i, True) for i in range(reps)])
    return known, ce, opt, river_ce, river_opt


def band(mean, se, ax, color):
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, color=color)
    ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.18)


def save_figures(data):
    horizons, stability = data["stability"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for name, color in zip(stability, (COLORS["red"], COLORS["blue"], COLORS["green"])):
        axes[0].plot(horizons, stability[name], marker="o", label=name, color=color)
        axes[1].plot(horizons, stability[name] / horizons, marker="o", color=color)
        axes[2].plot(horizons, stability[name] / np.sqrt(horizons), marker="o", color=color)
    axes[0].set_ylabel("Cumulative regret")
    axes[1].set_ylabel(r"$R_T/T$")
    axes[2].set_ylabel(r"$R_T/\sqrt{T}$")
    for ax in axes:
        ax.set_xlabel("Rounds")
        ax.set_xscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stability_regret.png"), bbox_inches="tight")
    plt.close(fig)

    feedback = data["feedback"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    for arms, color in ((5, COLORS["blue"]), (10, COLORS["orange"])):
        x = np.arange(1, len(feedback[arms]["hedge"]) + 1)
        axes[0].plot(x, feedback[arms]["hedge"], linestyle="--", color=color, label=f"Hedge, A={arms}")
        axes[0].plot(x, feedback[arms]["exp3_mean"], color=color, label=f"Exp3, A={arms}")
        se = feedback[arms]["exp3_se"]
        axes[0].fill_between(x, feedback[arms]["exp3_mean"] - 1.96 * se, feedback[arms]["exp3_mean"] + 1.96 * se, color=color, alpha=0.12)
        window = 50
        var = np.convolve(feedback[arms]["variance"], np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window, len(x) + 1), var, color=color, label=f"A={arms}")
    axes[0].set(xlabel="Rounds", ylabel="Regret", title="Full information and chosen-action feedback")
    axes[1].set(xlabel="Rounds", ylabel="Mean squared estimation error", title="Importance-weight variance (moving average)")
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feedback_regret.png"), bbox_inches="tight")
    plt.close(fig)

    soil = data["soil"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    rounds = np.arange(1, len(soil["external"]) + 1)
    axes[0].plot(rounds, soil["external"], color=COLORS["orange"], label="Realized-state external regret")
    axes[0].plot(rounds, soil["policy"], color=COLORS["blue"], label="Counterfactual policy regret")
    axes[0].set(xlabel="Seasons", ylabel="Regret", title="The comparator changes with the state path")
    state_rounds = np.arange(len(soil["learner_states"]))
    axes[1].step(state_rounds, soil["learner_states"], where="post", color=COLORS["red"], label="Always harvest")
    axes[1].step(state_rounds, soil["comparator_states"], where="post", color=COLORS["green"], label=soil["best_policy"])
    axes[1].set_yticks([0, 1], ["Fertile", "Depleted"])
    axes[1].set(xlabel="Seasons", ylabel="Soil state", title="Actual and counterfactual state paths")
    axes[1].set_xlim(0, 16)
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "soil_policy_regret.png"), bbox_inches="tight")
    plt.close(fig)

    regret, shares, promo, qstar, _, _ = data["known"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = [COLORS["red"], COLORS["gray"], COLORS["orange"], COLORS["blue"]]
    for (name, curve), color in zip(regret.items(), colors):
        axes[0].plot(np.arange(1, len(curve) + 1), curve, label=name, color=color)
    x = np.arange(S); width = 0.18
    for j, (name, vals) in enumerate(shares.items()):
        axes[1].bar(x + (j - 1.5) * width, vals, width, label=name, color=colors[j])
    axes[2].imshow(qstar.sum(axis=0), aspect="auto", cmap="viridis")
    axes[2].set_xticks([0, 1], ["Hold", "Admit"])
    axes[2].set_yticks([0, 1, 2], ["Quiet", "Busy", "Crowded"])
    axes[0].set(xlabel="Days", ylabel="Policy regret", title="Timed-entry admissions")
    axes[1].set_xticks(x, ["Quiet", "Busy", "Crowded"])
    axes[1].set(ylabel="Share of visits", title="Induced state occupancy")
    axes[2].set_title("Hindsight occupancy")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "occupancy_online_mdp.png"), bbox_inches="tight")
    plt.close(fig)

    bell = data["bellman"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(bell["n"], bell["gap"], label="Saddle gap", color=COLORS["blue"])
    ax.loglog(bell["n"], bell["policy"], label="Policy value gap", color=COLORS["orange"])
    ax.loglog(bell["n"], bell["regret"], label="Average two-player regret", color=COLORS["green"], linestyle="--")
    ax.loglog(bell["vi_n"], bell["vi_gap"], label="Value-iteration error", color=COLORS["red"], linestyle=":")
    ax.set(xlabel="Iterations", ylabel="Gap", title="Two solvers on the Engine Replacement MDP")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bellman_no_regret.png"), bbox_inches="tight")
    plt.close(fig)

    known, ce, opt, rce, ropt = data["unknown"]
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    band(known.mean(axis=0), known.std(axis=0, ddof=1) / np.sqrt(len(known)), axes[0], COLORS["green"])
    band(ce.mean(axis=0), ce.std(axis=0, ddof=1) / np.sqrt(len(ce)), axes[0], COLORS["red"])
    band(opt.mean(axis=0), opt.std(axis=0, ddof=1) / np.sqrt(len(opt)), axes[0], COLORS["blue"])
    axes[0].lines[0].set_label("Known-model planner")
    axes[0].lines[1].set_label("Certainty equivalent")
    axes[0].lines[2].set_label("Optimistic")
    band(rce.mean(axis=0), rce.std(axis=0, ddof=1) / np.sqrt(len(rce)), axes[1], COLORS["red"])
    band(ropt.mean(axis=0), ropt.std(axis=0, ddof=1) / np.sqrt(len(ropt)), axes[1], COLORS["blue"])
    axes[1].lines[0].set_label("Certainty equivalent"); axes[1].lines[1].set_label("Optimistic")
    axes[0].set(xlabel="Episodes", ylabel="Cumulative regret", title="Unknown cold-storage dynamics")
    axes[1].set(xlabel="Episodes", ylabel="Cumulative visits to far-right state", title="RiverSwim information stress test")
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "unknown_dynamics.png"), bbox_inches="tight")
    plt.close(fig)


def write_table(data):
    regret, shares, promo, _, flow_residual, representation_residual = data["known"]
    rows = []
    for name in regret:
        rows.append(
            f"{name} & {regret[name][-1]:.2f} & {shares[name][2]:.3f} & "
            f"{promo[name][0]:.3f} & {promo[name][1]:.3f} & {promo[name][2]:.3f} \\\\"
        )
    text = "\n".join([
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Final regret & Crowded share & Quiet admit & Busy admit & Crowded admit \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        f"% maximum occupancy flow residual: {flow_residual:.3e}",
        f"% maximum Bellman/occupancy value residual: {representation_residual:.3e}",
    ])
    with open(os.path.join(OUT, "online_optimization_summary.tex"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        raise FileNotFoundError("This compact study has no retained cache; run without --plots-only.")
    data = {
        "stability": sim_stability(),
        "one_step_equivalence": sim_one_step_equivalence(),
        "feedback": sim_feedback(),
        "soil": sim_soil_policy_regret(),
        "known": sim_known_mdp(),
        "bellman": sim_bellman(),
        "unknown": sim_unknown(),
    }
    print(f"One-step OO/MDP policy residual: {data['one_step_equivalence'][0]:.3e}")
    print(f"One-step OO/MDP value residual: {data['one_step_equivalence'][1]:.3e}")
    print(f"Maximum occupancy-flow residual: {data['known'][-2]:.3e}")
    print(f"Maximum occupancy/Bellman value residual: {data['known'][-1]:.3e}")
    print(f"Soil external regret: {data['soil']['external'][-1]:.4f}")
    print(f"Soil policy regret: {data['soil']['policy'][-1]:.4f}")
    print(f"Known-MDP final O-REPS regret: {data['known'][0]['O-REPS'][-1]:.4f}")
    print(f"Known-model cold-storage final regret: {data['unknown'][0].mean(axis=0)[-1]:.4f}")
    print(f"Unknown-MDP CE final regret: {data['unknown'][1].mean(axis=0)[-1]:.4f}")
    print(f"Unknown-MDP optimistic final regret: {data['unknown'][2].mean(axis=0)[-1]:.4f}")
    if not args.data_only:
        save_figures(data)
        write_table(data)
        print("Saved six figures and online_optimization_summary.tex")


if __name__ == "__main__":
    main()
