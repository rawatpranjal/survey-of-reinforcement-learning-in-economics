# Fishery panel of the §9 dual simulation: seven paradigms (oracle, naive,
# myopic, RLS, model-based DP, Q-learning, GA) on a logistic-growth fishery
# with quadratic harvest cost. Chapter:
# World Models and Model-Based Reinforcement Learning.

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_DOUBLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()
import matplotlib.pyplot as plt

from fishery_env import FisheryEnv, solve_oracle_dp, oracle_action

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "fishery_paradigms"

ENV_PARAMS = dict(r=0.4, K=10.0, p=2.0, c=0.2, sigma=0.3)
SHARED_CONFIG = {
    **ENV_PARAMS,
    "GAMMA": 0.95,
    "T_EPISODE": 500,
    "N_SEEDS": 20,
    "N_S_GRID": 50,
    "N_H_GRID": 25,
    "INNER_DP_NS": 25,
    "INNER_DP_NH": 15,
    "INNER_DP_ITER": 60,
    "H_TRAJ": "realized",  # cache buster: h_traj records post-clip harvest
}

RLS_CONFIG = {**SHARED_CONFIG, "INIT_VAR": 100.0, "REFIT_EVERY": 25}
QL_CONFIG = {
    **SHARED_CONFIG,
    "G_S": 30,
    "G_H": 21,
    "ALPHA": 0.1,
    "EPS_HI": 0.3,
    "EPS_LO": 0.01,
}
GA_CONFIG = {
    **SHARED_CONFIG,
    "N_POP": 30,
    "L_BITS": 10,
    "P_CROSS": 0.6,
    "P_MUT": 0.0033,
    "GEN_LEN": 10,
}
MBPO_CONFIG = {**SHARED_CONFIG, "EXPLORE_STD": 0.15, "WARMUP": 10, "REFIT_EVERY": 25}
NAIVE_CONFIG = {**SHARED_CONFIG, "H_FIXED": 0.5}
MYOPIC_CONFIG = {**SHARED_CONFIG}
ORACLE_CONFIG = {**SHARED_CONFIG}


# ---------------------------------------------------------------------------
# Paradigm base
# ---------------------------------------------------------------------------


class Paradigm:
    name = "base"

    def reset(self, params, seed=0):
        pass

    def act(self, s, t):
        raise NotImplementedError

    def observe(self, s, h, r, s_next):
        pass


class OraclePolicy(Paradigm):
    name = "Oracle"

    def __init__(self, gamma, n_s, n_h):
        self.gamma = gamma
        self.n_s, self.n_h = n_s, n_h
        self.oracle = None

    def reset(self, params, seed=0):
        self.oracle = solve_oracle_dp(
            r=params["r"],
            K=params["K"],
            p=params["p"],
            c=params["c"],
            sigma=params["sigma"],
            gamma=self.gamma,
            n_s=self.n_s,
            n_h=self.n_h,
        )

    def act(self, s, t):
        return oracle_action(s, self.oracle)


class NaivePolicy(Paradigm):
    name = "Naive"

    def __init__(self, h_fixed):
        self.h_fixed = h_fixed

    def reset(self, params, seed=0):
        pass

    def act(self, s, t):
        return min(s, self.h_fixed)


class MyopicPolicy(Paradigm):
    """Open-access / per-period profit maximizer. Knows (p, c) but ignores
    stock dynamics. Requests the unconstrained interior solution
    h = p / c (= 10 with p = 2, c = 0.2), but the environment clips every
    realized harvest to min(s, h_max) with h_max = 1.5 * rK/4 = 1.5, i.e.
    50% above MSY. The stock therefore declines gradually (near zero within
    roughly fifteen steps from s_0 = K = 10), not on the first step; the
    additive noise then regrows small amounts of stock each period, which
    the agent immediately scavenges, so the fishery never recovers."""

    name = "Myopic"

    def __init__(self):
        self.p = None
        self.c = None

    def reset(self, params, seed=0):
        self.p = params["p"]
        self.c = params["c"]

    def act(self, s, t):
        # Unconstrained myopic optimum: h* = p/c
        h_star = self.p / self.c
        return float(min(s, max(0.0, h_star)))


class RLSPolicy(Paradigm):
    """Recursive LS on the linearized logistic. Plans by re-solving DP."""

    name = "RLS"

    def __init__(self, gamma, init_var, inner_ns, inner_nh, inner_iter, refit_every):
        self.gamma = gamma
        self.init_var = init_var
        self.inner_ns, self.inner_nh = inner_ns, inner_nh
        self.inner_iter = inner_iter
        self.refit_every = refit_every
        self.theta = None
        self.R = None
        self.p = None
        self.c = None
        self.sigma_hat = 0.3
        self.oracle = None
        self.n_obs = 0

    def reset(self, params, seed=0):
        self.theta = np.array([0.2, 0.02], dtype=np.float64)
        self.R = (1.0 / self.init_var) * np.eye(2)
        self.p = params["p"]
        self.c = params["c"]
        self.sigma_hat = params["sigma"]
        self.n_obs = 0
        self._refit_oracle()

    def _refit_oracle(self):
        r_hat = max(0.05, float(self.theta[0]))
        r_over_K = max(1e-3, float(self.theta[1]))
        K_hat = r_hat / r_over_K
        K_hat = float(np.clip(K_hat, 1.0, 50.0))
        self.oracle = solve_oracle_dp(
            r=r_hat,
            K=K_hat,
            p=self.p,
            c=self.c,
            sigma=self.sigma_hat,
            gamma=self.gamma,
            n_s=self.inner_ns,
            n_h=self.inner_nh,
            max_iter=self.inner_iter,
        )

    def act(self, s, t):
        return oracle_action(s, self.oracle)

    def observe(self, s, h, r, s_next):
        # Linear-in-parameters growth: delta_s + h = r*s - (r/K)*s^2 + eps
        target = (s_next - s) + h
        x = np.array([s, -(s**2)])
        self.R = self.R + np.outer(x, x)
        try:
            R_inv = np.linalg.inv(self.R)
            innov = target - x @ self.theta
            self.theta = self.theta + R_inv @ x * innov
        except np.linalg.LinAlgError:
            pass
        self.n_obs += 1
        if self.n_obs % self.refit_every == 0:
            self._refit_oracle()


class QLearningPolicy(Paradigm):
    """Tabular Q-learning on bucketed (s, h)."""

    name = "Q-Learning"

    def __init__(
        self, gamma, g_s, g_h, alpha, eps_hi, eps_lo, T_episode=500, h_max=1.5
    ):
        self.gamma = gamma
        self.g_s, self.g_h = g_s, g_h
        self.alpha = alpha
        self.eps_hi, self.eps_lo = eps_hi, eps_lo
        self.T = T_episode
        self.h_max = h_max
        self.Q = None
        self.actions = None
        self.rng = None
        self.s_max = None

    def reset(self, params, seed=0):
        self.s_max = 1.5 * params["K"]
        self.h_max = 1.5 * params["r"] * params["K"] / 4.0
        self.Q = np.zeros((self.g_s, self.g_h), dtype=np.float64)
        self.actions = np.linspace(0.0, self.h_max, self.g_h)
        self.rng = np.random.default_rng(seed + 12345)

    def _bucket_s(self, s):
        s = np.clip(s, 0.0, self.s_max)
        i = int(s / self.s_max * (self.g_s - 1))
        return min(max(i, 0), self.g_s - 1)

    def _eps(self, t):
        frac = min(1.0, t / max(1, self.T))
        return self.eps_hi + (self.eps_lo - self.eps_hi) * frac

    def act(self, s, t):
        i = self._bucket_s(s)
        eps = self._eps(t)
        if self.rng.random() < eps:
            j = self.rng.integers(0, self.g_h)
        else:
            j = int(np.argmax(self.Q[i]))
        self._last = (i, j)
        return float(min(s, self.actions[j]))

    def observe(self, s, h, r, s_next):
        i, j = self._last
        ni = self._bucket_s(s_next)
        td = r + self.gamma * self.Q[ni].max() - self.Q[i, j]
        self.Q[i, j] += self.alpha * td


class ArifovicGAPolicy(Paradigm):
    """Population of constant harvest rules."""

    name = "Arifovic GA"

    def __init__(self, n_pop, L_bits, p_cross, p_mut, gen_len):
        self.n_pop, self.L = n_pop, L_bits
        self.p_cross, self.p_mut = p_cross, p_mut
        self.gen_len = gen_len
        self.pop = None
        self.fitness = None
        self.n_obs = None
        self.rng = None
        self.h_max = None
        self.params = None
        self.idx_cycle = 0
        self.recent_obs = []

    def _decode(self, chrom):
        bits = chrom.astype(int)
        weights = 2 ** np.arange(self.L - 1, -1, -1)
        val = (bits * weights).sum()
        return val / (2**self.L - 1) * self.h_max

    def reset(self, params, seed=0):
        self.rng = np.random.default_rng(seed + 54321)
        self.h_max = 1.5 * params["r"] * params["K"] / 4.0
        self.params = params
        self.pop = self.rng.integers(0, 2, size=(self.n_pop, self.L))
        self.fitness = np.zeros(self.n_pop)
        self.n_obs = np.zeros(self.n_pop, dtype=int)
        self.idx_cycle = 0
        self.recent_obs = []

    def act(self, s, t):
        self._last_active = self.idx_cycle % self.n_pop
        self.idx_cycle += 1
        return min(s, float(self._decode(self.pop[self._last_active])))

    def observe(self, s, h, r, s_next):
        i = self._last_active
        self.fitness[i] = (self.fitness[i] * self.n_obs[i] + r) / (self.n_obs[i] + 1)
        self.n_obs[i] += 1
        self.recent_obs.append((s, h, r))
        if self.idx_cycle % self.gen_len == 0:
            self._evolve()

    def _evolve(self):
        fit = self.fitness.copy()
        fit_shift = fit - fit.min() + 1e-6
        probs = fit_shift / fit_shift.sum()
        new_pop = []
        last_s, _, _ = self.recent_obs[-1] if self.recent_obs else (5.0, 0, 0)
        p_, c_ = self.params["p"], self.params["c"]
        for _ in range(self.n_pop // 2):
            parents = self.rng.choice(self.n_pop, size=2, replace=True, p=probs)
            p1, p2 = self.pop[parents[0]].copy(), self.pop[parents[1]].copy()
            if self.rng.random() < self.p_cross:
                pt = self.rng.integers(1, self.L)
                p1[pt:], p2[pt:] = p2[pt:].copy(), p1[pt:].copy()
            mask1 = self.rng.random(self.L) < self.p_mut
            mask2 = self.rng.random(self.L) < self.p_mut
            p1 = np.where(mask1, 1 - p1, p1)
            p2 = np.where(mask2, 1 - p2, p2)
            for child, parent_idx in [(p1, parents[0]), (p2, parents[1])]:
                parent_chrom = self.pop[parent_idx]
                hc = min(last_s, self._decode(child))
                hp = min(last_s, self._decode(parent_chrom))
                pi_c = p_ * hc - 0.5 * c_ * hc**2
                pi_p = p_ * hp - 0.5 * c_ * hp**2
                new_pop.append(child if pi_c >= pi_p else parent_chrom.copy())
        new_pop = np.array(new_pop[: self.n_pop])
        if new_pop.shape[0] < self.n_pop:
            extra = self.rng.integers(
                0, 2, size=(self.n_pop - new_pop.shape[0], self.L)
            )
            new_pop = np.vstack([new_pop, extra])
        self.pop = new_pop
        self.fitness = np.zeros(self.n_pop)
        self.n_obs = np.zeros(self.n_pop, dtype=int)


class MBPOPolicy(Paradigm):
    """Learn (r, K, p, c) by LS; plan by re-solving DP with point estimates.
    Despite the class name retained from the cobweb sibling, the planner on
    the non-linear fishery is grid-based dynamic programming, not LQ Riccati;
    the paradigm's display name is therefore "Model-Based DP"."""

    name = "Model-Based DP"

    def __init__(
        self, gamma, explore_std, warmup, refit_every, inner_ns, inner_nh, inner_iter
    ):
        self.gamma = gamma
        self.explore_std = explore_std
        self.warmup = warmup
        self.refit_every = refit_every
        self.inner_ns, self.inner_nh = inner_ns, inner_nh
        self.inner_iter = inner_iter
        self.data = []
        self.rng = None
        self.r_hat = 0.2
        self.K_hat = 10.0
        self.p_hat = 1.0
        self.c_hat = 0.5
        self.sigma_hat = 0.3
        self.oracle = None
        self.h_max = None

    def reset(self, params, seed=0):
        self.data = []
        self.rng = np.random.default_rng(seed + 98765)
        self.r_hat = 0.2
        self.K_hat = 10.0
        self.p_hat = 1.0
        self.c_hat = 0.5
        self.sigma_hat = params["sigma"]
        self.h_max = 1.5 * params["r"] * params["K"] / 4.0
        self._refit_oracle()

    def _refit_oracle(self):
        try:
            self.oracle = solve_oracle_dp(
                r=max(0.05, self.r_hat),
                K=float(np.clip(self.K_hat, 1.0, 50.0)),
                p=max(0.1, self.p_hat),
                c=max(0.01, self.c_hat),
                sigma=max(0.05, self.sigma_hat),
                gamma=self.gamma,
                n_s=self.inner_ns,
                n_h=self.inner_nh,
                max_iter=self.inner_iter,
            )
        except Exception:
            pass

    def _refit_model(self):
        if len(self.data) < 8:
            return
        s_arr = np.array([d[0] for d in self.data])
        h_arr = np.array([d[1] for d in self.data])
        r_arr = np.array([d[2] for d in self.data])
        sn_arr = np.array([d[3] for d in self.data])
        # Growth: delta_s + h = r*s - (r/K)*s^2
        target = (sn_arr - s_arr) + h_arr
        X = np.column_stack([s_arr, -(s_arr**2)])
        try:
            coef, *_ = np.linalg.lstsq(X, target, rcond=None)
            r_hat = max(0.05, float(coef[0]))
            r_over_K = max(1e-3, float(coef[1]))
            self.r_hat = r_hat
            self.K_hat = float(np.clip(r_hat / r_over_K, 1.0, 50.0))
        except np.linalg.LinAlgError:
            pass
        # Reward: r_t = p h - (c/2) h^2
        Xr = np.column_stack([h_arr, -0.5 * h_arr**2])
        try:
            coef2, *_ = np.linalg.lstsq(Xr, r_arr, rcond=None)
            self.p_hat = max(0.1, float(coef2[0]))
            self.c_hat = max(0.01, float(coef2[1]))
        except np.linalg.LinAlgError:
            pass

    def act(self, s, t):
        if t < self.warmup or self.oracle is None:
            return float(self.rng.uniform(0, min(s, self.h_max)))
        h = oracle_action(s, self.oracle)
        decay = max(0.1, 1.0 - t / 250.0)
        h = h + self.rng.normal(0, self.explore_std * decay)
        return float(np.clip(h, 0.0, min(s, self.h_max)))

    def observe(self, s, h, r, s_next):
        self.data.append((s, h, r, s_next))
        if len(self.data) % self.refit_every == 0:
            self._refit_model()
            self._refit_oracle()


# ---------------------------------------------------------------------------
# Rollout + compute
# ---------------------------------------------------------------------------


def rollout(paradigm, params, T, gamma, seed, track_traj=False):
    env = FisheryEnv(
        r=params["r"],
        K=params["K"],
        p=params["p"],
        c=params["c"],
        sigma=params["sigma"],
        gamma=gamma,
        T=T,
        seed=seed,
    )
    paradigm.reset(params, seed=seed)
    s = env.reset()
    rewards = np.zeros(T)
    if track_traj:
        s_traj = np.zeros(T)
        h_traj = np.zeros(T)
    for t in range(T):
        h = paradigm.act(s, t)
        if track_traj:
            s_traj[t] = s
        s_next, r, done, info = env.step(h)
        if track_traj:
            h_traj[t] = info["h"]  # realized (post-clip) harvest, not the request
        paradigm.observe(s, h, r, s_next)
        rewards[t] = r
        s = s_next
        if done:
            break
    if track_traj:
        return rewards, s_traj, h_traj
    return rewards


def _extract_param_estimates(paradigm):
    """Return final (r_hat, K_hat) for paradigms that estimate them, else None."""
    if isinstance(paradigm, RLSPolicy):
        r_hat = max(0.05, float(paradigm.theta[0]))
        r_over_K = max(1e-3, float(paradigm.theta[1]))
        return (r_hat, r_hat / r_over_K)
    if isinstance(paradigm, MBPOPolicy):
        return (float(paradigm.r_hat), float(paradigm.K_hat))
    return None


def make_paradigm(name, config):
    if name == "Oracle":
        return OraclePolicy(
            gamma=config["GAMMA"], n_s=config["N_S_GRID"], n_h=config["N_H_GRID"]
        )
    if name == "Naive":
        return NaivePolicy(h_fixed=config["H_FIXED"])
    if name == "Myopic":
        return MyopicPolicy()
    if name == "RLS":
        return RLSPolicy(
            gamma=config["GAMMA"],
            init_var=config["INIT_VAR"],
            inner_ns=config["INNER_DP_NS"],
            inner_nh=config["INNER_DP_NH"],
            inner_iter=config["INNER_DP_ITER"],
            refit_every=config["REFIT_EVERY"],
        )
    if name == "Q-Learning":
        return QLearningPolicy(
            gamma=config["GAMMA"],
            g_s=config["G_S"],
            g_h=config["G_H"],
            alpha=config["ALPHA"],
            eps_hi=config["EPS_HI"],
            eps_lo=config["EPS_LO"],
            T_episode=config["T_EPISODE"],
        )
    if name == "Arifovic GA":
        return ArifovicGAPolicy(
            n_pop=config["N_POP"],
            L_bits=config["L_BITS"],
            p_cross=config["P_CROSS"],
            p_mut=config["P_MUT"],
            gen_len=config["GEN_LEN"],
        )
    if name == "Model-Based DP":
        return MBPOPolicy(
            gamma=config["GAMMA"],
            explore_std=config["EXPLORE_STD"],
            warmup=config["WARMUP"],
            refit_every=config["REFIT_EVERY"],
            inner_ns=config["INNER_DP_NS"],
            inner_nh=config["INNER_DP_NH"],
            inner_iter=config["INNER_DP_ITER"],
        )
    raise ValueError(name)


def compute_shared(config):
    N = config["N_SEEDS"]
    T = config["T_EPISODE"]
    params = {k: config[k] for k in ("r", "K", "p", "c", "sigma")}
    oracle_rewards = np.zeros((N, T))
    for s in range(N):
        oracle = make_paradigm("Oracle", config)
        oracle_rewards[s] = rollout(oracle, params, T, config["GAMMA"], seed=s)
    return dict(params=params, oracle_rewards=oracle_rewards)


def compute_paradigm(config, shared, name):
    N = config["N_SEEDS"]
    T = config["T_EPISODE"]
    params = shared["params"]
    oracle_rewards = shared["oracle_rewards"]
    regret_curves = np.zeros((N, T))
    final_regret = np.zeros(N)
    r_hats = np.full(N, np.nan)
    K_hats = np.full(N, np.nan)
    s_traj = None
    h_traj = None
    for s in range(N):
        np.random.seed(s)
        paradigm = make_paradigm(name, config)
        if s == 0:
            rewards, s_traj, h_traj = rollout(
                paradigm, params, T, config["GAMMA"], seed=s, track_traj=True
            )
        else:
            rewards = rollout(paradigm, params, T, config["GAMMA"], seed=s)
        est = _extract_param_estimates(paradigm)
        if est is not None:
            r_hats[s], K_hats[s] = est
        curve = np.cumsum(oracle_rewards[s] - rewards)
        regret_curves[s] = curve
        final_regret[s] = curve[-1]
    out = dict(
        regret_curves=regret_curves,
        final_regret=final_regret,
        mean_curve=regret_curves.mean(axis=0),
        se_curve=regret_curves.std(axis=0, ddof=1) / np.sqrt(N),
        final_mean=float(final_regret.mean()),
        final_se=float(final_regret.std(ddof=1) / np.sqrt(N)),
        s_traj=s_traj,
        h_traj=h_traj,
    )
    if not np.all(np.isnan(r_hats)):
        out["r_hats"] = r_hats
        out["K_hats"] = K_hats
    return out


PARADIGM_REGISTRY = {
    "Oracle": (compute_paradigm, ORACLE_CONFIG),
    "Naive": (compute_paradigm, NAIVE_CONFIG),
    "Myopic": (compute_paradigm, MYOPIC_CONFIG),
    "RLS": (compute_paradigm, RLS_CONFIG),
    "Q-Learning": (compute_paradigm, QL_CONFIG),
    "Arifovic GA": (compute_paradigm, GA_CONFIG),
    "Model-Based DP": (compute_paradigm, MBPO_CONFIG),
}


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "shared",
        SHARED_CONFIG,
        compute_shared,
        SHARED_CONFIG,
        force=("shared" in force),
    )
    results = {}
    for name, (fn, cfg) in PARADIGM_REGISTRY.items():
        results[name] = compute_or_load(
            CACHE_DIR,
            SCRIPT_NAME,
            name.replace(" ", "_"),
            cfg,
            fn,
            cfg,
            shared,
            name,
            force=(name in force or "shared" in force),
        )
    return dict(shared=shared, results=results)


# Rank order by expected performance (lower regret = better)
PARADIGM_ORDER = [
    "Oracle",
    "RLS",
    "Model-Based DP",
    "Q-Learning",
    "Naive",
    "Arifovic GA",
    "Myopic",
]
PARADIGM_COLORS = {
    "Oracle": COLORS["black"],
    "RLS": COLORS["red"],
    "Model-Based DP": COLORS["purple"],
    "Naive": COLORS["gray"],
    "Arifovic GA": COLORS["green"],
    "Q-Learning": COLORS["blue"],
    "Myopic": COLORS["orange"],
}


def generate_outputs(data):
    apply_style()
    T = SHARED_CONFIG["T_EPISODE"]
    t_axis = np.arange(1, T + 1)

    # Sort paradigms by final regret ascending (Oracle first at 0)
    ranked = sorted(PARADIGM_ORDER, key=lambda nm: data["results"][nm]["final_mean"])

    fig, (ax_reg, ax_traj) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Left panel: cumulative regret (mean ± SE over 20 seeds)
    for name in ranked:
        res = data["results"][name]
        mean, se = res["mean_curve"], res["se_curve"]
        ax_reg.plot(
            t_axis, mean, label=name, color=PARADIGM_COLORS[name], linewidth=1.7
        )
        ax_reg.fill_between(
            t_axis, mean - se, mean + se, color=PARADIGM_COLORS[name], alpha=0.15
        )
    ax_reg.axhline(0, **BENCH_STYLE)
    ax_reg.set_xlabel("environment step $t$")
    ax_reg.set_ylabel("cumulative regret")
    ax_reg.set_title("Cumulative regret (20 seeds, mean $\\pm$ SE)")
    ax_reg.legend(loc="upper left", fontsize=8)

    # Right panel: seed-0 stock and harvest trajectories
    # Two line groups: solid = stock (left y-axis), dashed = harvest (right y-axis)
    ax_h = ax_traj.twinx()
    K_val = SHARED_CONFIG["K"]
    msy_val = SHARED_CONFIG["r"] * K_val / 4.0  # r*K/4 for logistic growth
    for name in ranked:
        res = data["results"][name]
        s_tr = res.get("s_traj")
        h_tr = res.get("h_traj")
        if s_tr is None or h_tr is None:
            continue
        ax_traj.plot(
            t_axis,
            s_tr,
            color=PARADIGM_COLORS[name],
            linewidth=1.4,
            linestyle="-",
            alpha=0.85,
        )
        ax_h.plot(
            t_axis,
            h_tr,
            color=PARADIGM_COLORS[name],
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )
    ax_traj.axhline(K_val, **BENCH_STYLE, label=f"$K={K_val:.0f}$")
    ax_h.axhline(
        msy_val,
        color=COLORS["gray"],
        linewidth=1.0,
        linestyle=":",
        label=f"MSY harvest $rK/4={msy_val:.1f}$",
    )
    ax_traj.set_xlabel("environment step $t$")
    ax_traj.set_ylabel("stock $s_t$ (solid)")
    ax_h.set_ylabel("harvest $h_t$ (dashed)")
    ax_traj.set_title("Seed-0 trajectories: stock (solid) and harvest (dashed)")
    # Combined legend from both axes
    lines_s, labels_s = ax_traj.get_legend_handles_labels()
    lines_h, labels_h = ax_h.get_legend_handles_labels()
    ax_traj.legend(
        lines_s + lines_h, labels_s + labels_h, fontsize=7, loc="upper right"
    )

    fig.suptitle("Fishery paradigms: seven learning approaches", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fishery_paradigms.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tbl_path = os.path.join(OUTPUT_DIR, "fishery_paradigms_results.tex")
    with open(tbl_path, "w") as f:
        f.write("% Cumulative regret at T=500, mean ± SE over 20 seeds.\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Paradigm & Final regret \\\\\n")
        f.write("\\midrule\n")
        for name in ranked:
            r = data["results"][name]
            f.write(f"{name} & {r['final_mean']:.2f} $\\pm$ {r['final_se']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Table saved: {tbl_path}")

    print("\n=== Cumulative regret at T=500 (mean ± SE, n=20 seeds) ===\n")
    print(f"{'Paradigm':<16} {'Final regret':>22}")
    print("-" * 40)
    for name in ranked:
        r = data["results"][name]
        print(f"{name:<16}  {r['final_mean']:>10.2f} ± {r['final_se']:>5.2f}")

    # Diagnostic: collapse incidence per paradigm.
    print(
        "\n=== Stock-collapse incidence: fraction of seeds with mean final"
        " regret >= 0.95 * Myopic floor (proxy for sustained collapse) ===\n"
    )
    myopic_floor = data["results"]["Myopic"]["final_mean"]
    if myopic_floor > 1.0:
        print(f"{'Paradigm':<16} {'frac collapsed':>16}")
        print("-" * 34)
        for name in ranked:
            fr = data["results"][name]["final_regret"]
            frac = float(np.mean(fr >= 0.95 * myopic_floor))
            print(f"{name:<16} {frac:>16.2f}")

    # Parameter recovery for the two structured learners.
    r_true = SHARED_CONFIG["r"]
    K_true = SHARED_CONFIG["K"]
    recovery_rows = []
    for name in PARADIGM_ORDER:
        res = data["results"][name]
        if "r_hats" not in res:
            continue
        r_hats = res["r_hats"]
        K_hats = res["K_hats"]
        r_err = np.abs(r_hats - r_true)
        K_err = np.abs(K_hats - K_true)
        recovery_rows.append(
            (
                name,
                float(np.nanmean(r_hats)),
                float(np.nanstd(r_hats, ddof=1) / np.sqrt(np.sum(~np.isnan(r_hats)))),
                float(np.nanmean(K_hats)),
                float(np.nanstd(K_hats, ddof=1) / np.sqrt(np.sum(~np.isnan(K_hats)))),
                float(np.nanmean(r_err)),
                float(np.nanmean(K_err)),
            )
        )
    if recovery_rows:
        print(
            f"\n=== Parameter recovery at t = T (true r = {r_true}, K = {K_true}) ===\n"
        )
        print(
            f"{'Paradigm':<16} {'r_hat':>16} {'K_hat':>16} {'|r_err|':>10} {'|K_err|':>10}"
        )
        print("-" * 72)
        for row in recovery_rows:
            name, r_m, r_se, K_m, K_se, r_err, K_err = row
            print(
                f"{name:<16} {r_m:>7.3f}±{r_se:.3f}  {K_m:>7.3f}±{K_se:.3f}  "
                f"{r_err:>10.3f} {K_err:>10.3f}"
            )
        # Write recovery table
        rec_path = os.path.join(OUTPUT_DIR, "fishery_paradigms_recovery.tex")
        with open(rec_path, "w") as f:
            f.write(
                f"% Parameter recovery on fishery, mean +- SE over 20 seeds."
                f" True r = {r_true}, K = {K_true}.\n"
            )
            f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
            f.write(
                "Paradigm & $\\hat r$ & $\\hat K$ & "
                "$|\\hat r - r|$ & $|\\hat K - K|$ \\\\\n"
            )
            f.write("\\midrule\n")
            for row in recovery_rows:
                name, r_m, r_se, K_m, K_se, r_err, K_err = row
                f.write(
                    f"{name} & {r_m:.3f} $\\pm$ {r_se:.3f} & "
                    f"{K_m:.3f} $\\pm$ {K_se:.3f} & "
                    f"{r_err:.3f} & {K_err:.3f} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"  Recovery table saved: {rec_path}")


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print(f"=== {SCRIPT_NAME} ===")
    print(f"Paradigms: {PARADIGM_ORDER}")
    print(
        f"N_SEEDS = {SHARED_CONFIG['N_SEEDS']}, T_EPISODE = {SHARED_CONFIG['T_EPISODE']}\n"
    )

    if args.plots_only:
        data = compute_data(force=set())
        generate_outputs(data)
        return
    data = compute_data(force=force)
    if args.data_only:
        print("Data-only run complete.")
        return
    generate_outputs(data)


if __name__ == "__main__":
    main()
