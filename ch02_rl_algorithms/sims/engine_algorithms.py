# Eight algorithm families on the running example
# Chapter 2 - Reinforcement Learning Algorithms
# Runs MC, TD(0), SARSA, Q-learning, REINFORCE, actor-critic, FQI and DQN on the
# two-grade bus engine replacement MDP and shows every family landing on the same
# hand-computed V* = (5.3448, 4.3103). Prediction methods evaluate pi*; control methods
# learn from epsilon-greedy or random behavior; policy methods are scored by the exact
# value of their final policy (computed by the resolvent, disclosed in the table).

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    exact_value,
    policy_kernel,
    policy_from_logits,
    solve_optimal,
)

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCRIPT_NAME = "engine_algorithms"
OUTPUT_DIR = os.path.dirname(__file__)

STATE_NAMES = ["low", "high"]
PI_STAR = [KEEP, REPLACE]

SHARED_CONFIG = {"gamma": GAMMA, "n_seeds": 10, "version": 1}
MC_CONFIG = {**SHARED_CONFIG, "episodes": 2000, "horizon": 200}
TD_CONFIG = {**SHARED_CONFIG, "steps": 20000, "alpha_pow": 0.7}
SARSA_CONFIG = {**SHARED_CONFIG, "steps": 50000, "alpha_pow": 0.7, "eps_pow": 0.5}
QL_CONFIG = {**SHARED_CONFIG, "steps": 50000, "alpha_pow": 0.7, "epsilon": 0.1}
REINFORCE_CONFIG = {
    **SHARED_CONFIG,
    "updates": 500,
    "batch": 10,
    "horizon": 100,
    "lr": 0.1,
}
AC_CONFIG = {
    **SHARED_CONFIG,
    "episodes": 2000,
    "horizon": 200,
    "lr_actor": 0.02,
    "alpha_pow": 0.6,
}
FQI_CONFIG = {**SHARED_CONFIG, "transitions": 20000, "iterations": 200}
DQN_CONFIG = {
    **SHARED_CONFIG,
    "steps": 40000,
    "hidden": 32,
    "lr": 1e-3,
    "buffer": 5000,
    "batch": 64,
    "target_every": 250,
    "warmup": 500,
    "epsilon": 0.1,
}

# Every algorithm must land within this sup-norm distance of V*; the assert fails loudly
# if one drifts, so a broken learner cannot ship a green table.
TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# Environment sampling
# ---------------------------------------------------------------------------


def sample_next(P, s, a, rng):
    return LOW if rng.random() < P[s, a, LOW] else HIGH


def compute_shared():
    P, r = build_mdp()
    V_star, greedy, Q_star = solve_optimal(P, r, GAMMA)
    print("Shared: exact DP solution")
    print(f"  V* = [low {V_star[LOW]:.4f}, high {V_star[HIGH]:.4f}]")
    print(
        f"  greedy = ({['keep', 'replace'][greedy[LOW]]}, {['keep', 'replace'][greedy[HIGH]]})"
    )
    return {"P": P, "r": r, "V_star": V_star, "Q_star": Q_star}


# ---------------------------------------------------------------------------
# The eight algorithms. Each returns per-seed V-estimates, shape (n_seeds, 2).
# ---------------------------------------------------------------------------


def run_mc(shared):
    """First-visit Monte Carlo prediction of pi*, episodes started uniformly."""
    P, r = shared["P"], shared["r"]
    cfg = MC_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(100 + seed)
        V_sum = np.zeros(2)
        V_cnt = np.zeros(2)
        for ep in range(cfg["episodes"]):
            s0 = ep % 2  # alternate start states so both are first-visited
            traj = []
            s = s0
            for _ in range(cfg["horizon"]):
                a = PI_STAR[s]
                s_next = sample_next(P, s, a, rng)
                traj.append((s, r[s, a]))
                s = s_next
            G = 0.0
            returns = [None] * len(traj)
            for t in range(len(traj) - 1, -1, -1):
                G = traj[t][1] + GAMMA * G
                returns[t] = G
            seen = set()
            for t, (s_t, _) in enumerate(traj):
                if s_t not in seen:
                    seen.add(s_t)
                    V_sum[s_t] += returns[t]
                    V_cnt[s_t] += 1
        out[seed] = V_sum / V_cnt
    return out


def run_td0(shared):
    """TD(0) prediction of pi*, step size n^-0.7 in per-state visit counts."""
    P, r = shared["P"], shared["r"]
    cfg = TD_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(200 + seed)
        V = np.zeros(2)
        visits = np.zeros(2)
        s = LOW
        for _ in range(cfg["steps"]):
            a = PI_STAR[s]
            s_next = sample_next(P, s, a, rng)
            visits[s] += 1
            alpha = visits[s] ** -cfg["alpha_pow"]
            V[s] += alpha * (r[s, a] + GAMMA * V[s_next] - V[s])
            s = s_next
        out[seed] = V
    return out


def run_sarsa(shared):
    """SARSA control with GLIE epsilon(s) = visits(s)^-0.5."""
    P, r = shared["P"], shared["r"]
    cfg = SARSA_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(300 + seed)
        Q = np.zeros((2, 2))
        visits_s = np.zeros(2)
        visits_sa = np.zeros((2, 2))

        def act(s):
            eps = (visits_s[s] + 1.0) ** -cfg["eps_pow"]
            if rng.random() < eps:
                return int(rng.integers(2))
            return int(Q[s].argmax())

        s = LOW
        a = act(s)
        for _ in range(cfg["steps"]):
            s_next = sample_next(P, s, a, rng)
            a_next = act(s_next)
            visits_s[s] += 1
            visits_sa[s, a] += 1
            alpha = visits_sa[s, a] ** -cfg["alpha_pow"]
            Q[s, a] += alpha * (r[s, a] + GAMMA * Q[s_next, a_next] - Q[s, a])
            s, a = s_next, a_next
        out[seed] = Q.max(axis=1)
    return out


def run_qlearning(shared):
    """Q-learning control under fixed epsilon-greedy behavior."""
    P, r = shared["P"], shared["r"]
    cfg = QL_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(400 + seed)
        Q = np.zeros((2, 2))
        visits_sa = np.zeros((2, 2))
        s = LOW
        for _ in range(cfg["steps"]):
            if rng.random() < cfg["epsilon"]:
                a = int(rng.integers(2))
            else:
                a = int(Q[s].argmax())
            s_next = sample_next(P, s, a, rng)
            visits_sa[s, a] += 1
            alpha = visits_sa[s, a] ** -cfg["alpha_pow"]
            Q[s, a] += alpha * (r[s, a] + GAMMA * Q[s_next].max() - Q[s, a])
            s = s_next
        out[seed] = Q.max(axis=1)
    return out


def run_reinforce(shared):
    """REINFORCE on the one-logit-per-state policy; scored by the exact value of the
    final policy."""
    P, r = shared["P"], shared["r"]
    cfg = REINFORCE_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(500 + seed)
        theta = np.zeros(2)
        for _ in range(cfg["updates"]):
            grad = np.zeros(2)
            for _ in range(cfg["batch"]):
                p_replace = 1.0 / (1.0 + np.exp(-theta))
                s = LOW
                traj = []
                for _ in range(cfg["horizon"]):
                    a = REPLACE if rng.random() < p_replace[s] else KEEP
                    s_next = sample_next(P, s, a, rng)
                    traj.append((s, a, r[s, a]))
                    s = s_next
                G = 0.0
                returns = [None] * len(traj)
                for t in range(len(traj) - 1, -1, -1):
                    G = traj[t][2] + GAMMA * G
                    returns[t] = G
                for t, (s_t, a_t, _) in enumerate(traj):
                    score = (1.0 if a_t == REPLACE else 0.0) - p_replace[s_t]
                    grad[s_t] += (GAMMA**t) * returns[t] * score
            theta += cfg["lr"] * grad / cfg["batch"]
        b = policy_from_logits(theta)
        P_b, r_b = policy_kernel(P, r, b)
        out[seed] = exact_value(P_b, r_b, GAMMA)
    return out


def run_actor_critic(shared):
    """One-step actor-critic: tabular TD(0) critic, policy logits actor."""
    P, r = shared["P"], shared["r"]
    cfg = AC_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(600 + seed)
        theta = np.zeros(2)
        V = np.zeros(2)
        visits = np.zeros(2)
        for ep in range(cfg["episodes"]):
            s = ep % 2
            I = 1.0
            for _ in range(cfg["horizon"]):
                p_replace = 1.0 / (1.0 + np.exp(-theta[s]))
                a = REPLACE if rng.random() < p_replace else KEEP
                s_next = sample_next(P, s, a, rng)
                visits[s] += 1
                alpha_w = visits[s] ** -cfg["alpha_pow"]
                delta = r[s, a] + GAMMA * V[s_next] - V[s]
                V[s] += alpha_w * delta
                score = (1.0 if a == REPLACE else 0.0) - p_replace
                theta[s] += cfg["lr_actor"] * I * delta * score
                I *= GAMMA
                s = s_next
        b = policy_from_logits(theta)
        P_b, r_b = policy_kernel(P, r, b)
        out[seed] = exact_value(P_b, r_b, GAMMA)
    return out


def run_fqi(shared):
    """Fitted Q-iteration on a fixed random-behavior dataset, tabular regression."""
    P, r = shared["P"], shared["r"]
    cfg = FQI_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(700 + seed)
        data = []
        s = LOW
        for _ in range(cfg["transitions"]):
            a = int(rng.integers(2))
            s_next = sample_next(P, s, a, rng)
            data.append((s, a, r[s, a], s_next))
            s = s_next
        data = np.array(data)
        Q = np.zeros((2, 2))
        for _ in range(cfg["iterations"]):
            targets = data[:, 2] + GAMMA * Q[data[:, 3].astype(int)].max(axis=1)
            Q_new = np.zeros((2, 2))
            for s_i in range(2):
                for a_i in range(2):
                    mask = (data[:, 0] == s_i) & (data[:, 1] == a_i)
                    Q_new[s_i, a_i] = targets[mask].mean()
            Q = Q_new
        out[seed] = Q.max(axis=1)
    return out


def run_dqn(shared):
    """DQN: one-hot state MLP, replay buffer, target network, epsilon-greedy."""
    import torch
    import torch.nn as nn

    P, r = shared["P"], shared["r"]
    cfg = DQN_CONFIG
    out = np.zeros((cfg["n_seeds"], 2))
    eye = np.eye(2, dtype=np.float32)
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(800 + seed)
        torch.manual_seed(800 + seed)
        net = nn.Sequential(
            nn.Linear(2, cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden"], cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden"], 2),
        )
        target = nn.Sequential(
            nn.Linear(2, cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden"], cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden"], 2),
        )
        target.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        buf_s = np.zeros(cfg["buffer"], dtype=int)
        buf_a = np.zeros(cfg["buffer"], dtype=int)
        buf_r = np.zeros(cfg["buffer"], dtype=np.float32)
        buf_sn = np.zeros(cfg["buffer"], dtype=int)
        n_in_buf = 0
        s = LOW
        for t in range(cfg["steps"]):
            if rng.random() < cfg["epsilon"]:
                a = int(rng.integers(2))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(eye[s])).argmax())
            s_next = sample_next(P, s, a, rng)
            idx = t % cfg["buffer"]
            buf_s[idx], buf_a[idx], buf_r[idx], buf_sn[idx] = s, a, r[s, a], s_next
            n_in_buf = min(n_in_buf + 1, cfg["buffer"])
            s = s_next
            if t >= cfg["warmup"]:
                batch = rng.integers(n_in_buf, size=cfg["batch"])
                bs = torch.from_numpy(eye[buf_s[batch]])
                ba = torch.from_numpy(buf_a[batch]).long()
                br = torch.from_numpy(buf_r[batch])
                bsn = torch.from_numpy(eye[buf_sn[batch]])
                with torch.no_grad():
                    y = br + GAMMA * target(bsn).max(dim=1).values
                q = net(bs).gather(1, ba[:, None]).squeeze(1)
                loss = nn.functional.mse_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (t + 1) % cfg["target_every"] == 0:
                target.load_state_dict(net.state_dict())
        with torch.no_grad():
            out[seed] = net(torch.from_numpy(eye)).max(dim=1).values.numpy()
    return out


ALGO_REGISTRY = {
    "MC": (
        run_mc,
        MC_CONFIG,
        "prediction of $\\pi^\\star$",
        f"{MC_CONFIG['episodes']} episodes",
    ),
    "TD(0)": (
        run_td0,
        TD_CONFIG,
        "prediction of $\\pi^\\star$",
        f"{TD_CONFIG['steps']} steps",
    ),
    "SARSA": (
        run_sarsa,
        SARSA_CONFIG,
        "on-policy control",
        f"{SARSA_CONFIG['steps']} steps",
    ),
    "Q-learning": (
        run_qlearning,
        QL_CONFIG,
        "off-policy control",
        f"{QL_CONFIG['steps']} steps",
    ),
    "REINFORCE": (
        run_reinforce,
        REINFORCE_CONFIG,
        "policy gradient",
        f"{REINFORCE_CONFIG['updates'] * REINFORCE_CONFIG['batch']} episodes",
    ),
    "Actor-critic": (
        run_actor_critic,
        AC_CONFIG,
        "policy gradient",
        f"{AC_CONFIG['episodes']} episodes",
    ),
    "FQI": (
        run_fqi,
        FQI_CONFIG,
        "batch value iteration",
        f"{FQI_CONFIG['transitions']} transitions",
    ),
    "DQN": (run_dqn, DQN_CONFIG, "deep value learning", f"{DQN_CONFIG['steps']} steps"),
}


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        SHARED_CONFIG,
        compute_shared,
        force=("shared" in force),
    )
    cascade = "shared" in force
    results = {}
    for name, (fn, cfg, _, _) in ALGO_REGISTRY.items():
        key = name.replace("(", "").replace(")", "").replace("-", "_").lower()
        results[name] = compute_or_load(
            CACHE_DIR,
            SCRIPT_NAME,
            key,
            cfg,
            fn,
            shared,
            force=(key in force or cascade),
        )
    V_star = shared["V_star"]
    # guard: the hardcoded prediction target PI_STAR must be the DP greedy policy
    _, greedy, _ = solve_optimal(shared["P"], shared["r"], GAMMA)
    assert list(greedy) == PI_STAR, "PI_STAR no longer matches the DP greedy policy"
    print()
    print(
        "Per-algorithm value estimates against V* "
        f"= [low {V_star[LOW]:.4f}, high {V_star[HIGH]:.4f}], "
        f"{SHARED_CONFIG['n_seeds']} seeds:"
    )
    print(
        f"  {'algorithm':>14s}  {'V(low)':>8s}  {'se':>7s}  {'V(high)':>8s}  {'se':>7s}"
        f"  {'sup err':>8s}  {'se':>7s}"
    )
    summary = {}
    for name, est in results.items():
        mean = est.mean(axis=0)
        se = est.std(axis=0, ddof=1) / np.sqrt(est.shape[0])
        sup = np.abs(est - V_star).max(axis=1)
        summary[name] = {
            "V_mean": mean,
            "V_se": se,
            "sup_mean": float(sup.mean()),
            "sup_se": float(sup.std(ddof=1) / np.sqrt(len(sup))),
            "sup_max": float(sup.max()),
        }
        print(
            f"  {name:>14s}  {mean[LOW]:8.4f}  {se[LOW]:7.4f}  {mean[HIGH]:8.4f}"
            f"  {se[HIGH]:7.4f}  {sup.mean():8.4f}  {summary[name]['sup_se']:7.4f}"
        )
    print()
    for name, row in summary.items():
        assert row["sup_max"] < TOLERANCE, (
            f"{name} worst-seed sup error {row['sup_max']:.4f} exceeds tolerance {TOLERANCE}"
        )
    print(
        f"All {len(summary)} algorithms within sup-norm tolerance {TOLERANCE} of V* "
        "on every seed."
    )
    return {"shared": shared, "results": results, "summary": summary}


def generate_outputs(data):
    """One consolidated LaTeX table, rows ranked by mean sup-norm error."""
    V_star = data["shared"]["V_star"]
    summary = data["summary"]
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["sup_mean"])
    tex_path = os.path.join(OUTPUT_DIR, "engine_algorithms.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Eight algorithm families on the running example, "
            f"{SHARED_CONFIG['n_seeds']} seeds each, ranked by mean sup-norm distance "
            f"from $V^\\star = ({V_star[LOW]:.4f},\\ {V_star[HIGH]:.4f})$. Standard "
            "errors across seeds in brackets. Rows carry different data budgets and "
            "three scoring modes, so the ordering is not a race. Prediction rows "
            "evaluate $\\pi^\\star$; control rows learn from exploratory behavior and "
            "report $\\max_a \\hat{Q}$; policy-gradient rows report the exact value of "
            "the final learned policy.}\n"
        )
        f.write("\\label{tab:engine_algorithms}\n")
        f.write("\\begin{tabular}{llrrr}\n\\hline\n")
        f.write(
            "algorithm & family & $\\hat{V}(\\text{low})$ & $\\hat{V}(\\text{high})$"
            " & $\\|\\hat{V} - V^\\star\\|_\\infty$ \\\\\n\\hline\n"
        )
        for name, row in ranked:
            _, _, family, budget = ALGO_REGISTRY[name]
            f.write(
                f"{name} & {family}, {budget} & "
                f"{row['V_mean'][LOW]:.4f} ({row['V_se'][LOW]:.4f}) & "
                f"{row['V_mean'][HIGH]:.4f} ({row['V_se'][HIGH]:.4f}) & "
                f"{row['sup_mean']:.4f} ({row['sup_se']:.4f}) \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Eight algorithms on the running example"
    )
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print("=" * 70)
    print("EIGHT ALGORITHM FAMILIES ON THE RUNNING EXAMPLE")
    print("=" * 70)
    print()
    if args.plots_only:
        generate_outputs(compute_data())
    elif args.data_only:
        compute_data(force=force)
    else:
        generate_outputs(compute_data(force=force))
    print("\nDone.")


if __name__ == "__main__":
    main()
