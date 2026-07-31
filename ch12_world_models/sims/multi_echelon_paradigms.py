# Multi-echelon supply-chain panel: five paradigms on a serial inventory
# network with a lead-time pipeline and Poisson demand. Chapter: World Models
# and Model-Based Reinforcement Learning.
#
# The state the learner sees is the full physical pipeline (on-hand, in-transit,
# backlog), a 6-dimensional vector for K=2, L=1. The optimal policy needs only
# the K echelon inventory positions, a sufficient statistic the learner does
# not know a priori. The panel contrasts:
#   Oracle              echelon base-stock (Clark-Scarf optimal form), levels
#                       found by simulation optimization on the true model.
#   NN World Model      a neural net learns the pipeline+demand dynamics from
#                       interaction and plans by base-stock search on the model.
#   Model-Free DQN      function-approximation Q-learning over a coarse action
#                       grid; sample-inefficient under a tight online budget.
#   Decentralized       each stage runs a local newsvendor base-stock ignoring
#                       echelon coupling; a plausible but misspecified heuristic.
#   Naive               constant order equal to mean demand; no-learning floor.
#
# The thesis (matching Gijsbrechts 2022): a learned world model plus planning
# recovers near-oracle cost from interaction data, beating model-free RL by a
# wide margin and beating the misspecified heuristic on terminal (asymptotic)
# per-period cost, though it pays a larger exploration transient than the
# no-learning heuristic on cumulative regret; it does not beat the Clark-Scarf
# oracle, which is optimal for the serial system.

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_TRIPLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from multi_echelon_env import (
    MultiEchelonEnv,
    find_oracle_base_stock,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "multi_echelon_paradigms"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_PARAMS = dict(K=2, L=1, lam=5.0, p=8.0, order_cap=20, inv_cap=60)
SHARED_CONFIG = {
    **ENV_PARAMS,
    "GAMMA": 0.99,
    "T_EPISODE": 500,
    "N_SEEDS": 20,
    "ORACLE_SEARCH_SEQ": 12,
    "ORACLE_SEARCH_T": 500,
    "TERMINAL_WINDOW": 80,  # last-W-step window for asymptotic per-period cost
    "COST_SCALE": 50.0,  # cache buster / reward normalization for NN
}

NAIVE_CONFIG = {**SHARED_CONFIG}
DECENTRAL_CONFIG = {**SHARED_CONFIG, "REFIT_EVERY": 20, "SAFETY_Z": 1.0}
DQN_CONFIG = {
    **SHARED_CONFIG,
    "HIDDEN": 128,
    "LR": 1e-3,
    "BUFFER": 5000,
    "BATCH": 64,
    "EPS_HI": 1.0,
    "EPS_LO": 0.05,
    "TARGET_SYNC": 50,
    "WARMUP": 50,
    "N_UPDATES": 4,  # replay ratio: gradient steps per environment step
    "N_ACTION_LEVELS": 5,  # uniform levels on [0,order_cap]; K=2 -> 5^2 = 25 actions
}
WM_CONFIG = {
    **SHARED_CONFIG,
    "HIDDEN": 128,
    "LR": 3e-3,
    "BUFFER": 5000,
    "BATCH": 64,
    "WARMUP": 50,
    "REFIT_EVERY": 5,
    "REFIT_STEPS": 20,
    "REPLAN_EVERY": 20,
    "SIM_HORIZON": 50,
    "SIM_WARM": 15,
    "SEARCH_RADIUS": 3,
    "SEARCH_PASSES": 2,
    "DAMP": 0.5,  # blend new search result with prior S_hat to damp planner noise
    "N_DEMAND_PATHS": 24,  # Monte-Carlo demand samples per candidate (variance)
    "EXPLORE_STD": 2.0,
}
ORACLE_CONFIG = {**SHARED_CONFIG}


# ---------------------------------------------------------------------------
# Paradigm base
# ---------------------------------------------------------------------------


class Paradigm:
    name = "base"

    def reset(self, env, S_oracle, seed=0):
        pass

    def act(self, obs, env, t):
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, info):
        pass


class OracleBaseStock(Paradigm):
    """Echelon base-stock with the oracle levels S* (Clark-Scarf optimal form)."""

    name = "Oracle"

    def reset(self, env, S_oracle, seed=0):
        self.S = np.asarray(S_oracle)

    def act(self, obs, env, t):
        return env.order_from_echelon(self.S)


class NaiveConstant(Paradigm):
    """Order the running mean of observed demand every period, regardless of
    inventory state. A no-learning floor: it tracks average demand from data
    (the retailer observes demand) but never reacts to the pipeline state."""

    name = "Naive"

    def reset(self, env, S_oracle, seed=0):
        self.K = env.K
        self.order_cap = env.order_cap
        self.q = max(1, env.order_cap // 4)  # neutral start, no demand knowledge
        self.demands = []

    def act(self, obs, env, t):
        return np.full(self.K, self.q, dtype=np.int64)

    def observe(self, obs, action, reward, next_obs, info):
        self.demands.append(info["demand"])
        self.q = int(np.clip(round(np.mean(self.demands)), 1, self.order_cap))


class DecentralizedBaseStock(Paradigm):
    """Each stage runs a LOCAL base-stock on its own observed inflow demand,
    ignoring the echelon coupling. Stage 0 sees customer demand; each upstream
    stage sees its downstream stage's orders as its 'demand'. Base-stock level
    set to a local newsvendor estimate = mean + z*std over a one-period lead.
    This is a plausible practitioner heuristic that is provably suboptimal for
    serial systems and amplifies order variance upstream (bullwhip)."""

    name = "Decentralized"

    def __init__(self, refit_every, safety_z):
        self.refit_every = refit_every
        self.safety_z = safety_z

    def reset(self, env, S_oracle, seed=0):
        self.K = env.K
        self.L = env.L
        self.local_demand = [[] for _ in range(env.K)]  # observed inflow per stage
        # Neutral start from the order bound (no demand knowledge); refit from
        # observed local demand once data accumulates.
        self.S_local = np.array(
            [(env.L + 1) * env.order_cap * 0.4 * (k + 1) for k in range(env.K)],
            dtype=np.float64,
        )
        self.n = 0

    def act(self, obs, env, t):
        # Order each stage's echelon position up to its local base-stock.
        return env.order_from_echelon(self.S_local)

    def observe(self, obs, action, reward, next_obs, info):
        # Stage 0's 'demand' = customer demand; stage k>0 'demand' = orders it
        # received from stage k-1 last period (the local view).
        d0 = info["demand"]
        self.local_demand[0].append(d0)
        for k in range(1, self.K):
            self.local_demand[k].append(int(info["orders"][k - 1]))
        self.n += 1
        if self.n % self.refit_every == 0:
            for k in range(self.K):
                arr = np.array(self.local_demand[k][-100:], dtype=np.float64)
                if len(arr) >= 3:
                    mu, sd = arr.mean(), arr.std() + 1e-6
                    # local base-stock covers L+1 periods of local demand
                    self.S_local[k] = (self.L + 1) * mu + self.safety_z * sd * np.sqrt(
                        self.L + 1
                    )


# ---------------------------------------------------------------------------
# Model-free DQN (function approximation, coarse joint action grid)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(Paradigm):
    """Online DQN over the raw physical observation with a coarse joint action
    grid (N_ACTION_LEVELS per stage). Model-free: no dynamics model, learns a
    value function directly, so it is data-hungry under a tight online budget."""

    name = "Model-Free DQN"

    def __init__(
        self,
        gamma,
        hidden,
        lr,
        buffer,
        batch,
        eps_hi,
        eps_lo,
        target_sync,
        warmup,
        n_updates,
        n_levels,
        order_cap,
        K,
        cost_scale,
        T_episode,
    ):
        self.gamma = gamma
        self.hidden = hidden
        self.lr = lr
        self.buffer_cap = buffer
        self.batch = batch
        self.eps_hi, self.eps_lo = eps_hi, eps_lo
        self.target_sync = target_sync
        self.warmup = warmup
        self.cost_scale = cost_scale
        self.n_updates = n_updates
        self.n_levels = n_levels
        self.order_cap = order_cap
        self.K = K
        self.T = T_episode
        # Per-stage order levels are a uniform grid on the known action bound
        # [0, order_cap]; for order_cap=20 and n_levels=5 this is {0,5,10,15,20}.
        # No knowledge of the demand rate enters the action set. K=2 gives 5^2=25
        # joint actions.
        levels = np.unique(np.round(np.linspace(0, order_cap, n_levels)).astype(int))
        self.levels = levels
        grids = np.meshgrid(*[levels] * K, indexing="ij")
        self.action_table = np.stack([g.reshape(-1) for g in grids], axis=1)  # (A,K)
        self.n_actions = self.action_table.shape[0]

    def reset(self, env, S_oracle, seed=0):
        torch.manual_seed(seed + 777)
        self.rng = np.random.default_rng(seed + 777)
        self.obs_dim = env._obs().shape[0]
        self.scale = float(env.inv_cap)  # state bound (known)
        self.q = MLP(self.obs_dim, self.n_actions, self.hidden)
        self.q_target = MLP(self.obs_dim, self.n_actions, self.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.buffer = []
        self.t_step = 0
        self._last_a_idx = 0

    def _feat(self, obs):
        return torch.as_tensor(obs / self.scale, dtype=torch.float32)

    def _eps(self, t):
        frac = min(1.0, t / max(1, self.T * 0.7))
        return self.eps_hi + (self.eps_lo - self.eps_hi) * frac

    def act(self, obs, env, t):
        if self.rng.random() < self._eps(t) or self.t_step < self.warmup:
            a_idx = int(self.rng.integers(0, self.n_actions))
        else:
            with torch.no_grad():
                qv = self.q(self._feat(obs).unsqueeze(0)).squeeze(0).numpy()
            a_idx = int(np.argmax(qv))
        self._last_a_idx = a_idx
        return self.action_table[a_idx].copy()

    def observe(self, obs, action, reward, next_obs, info):
        self.buffer.append((obs, self._last_a_idx, reward, next_obs))
        if len(self.buffer) > self.buffer_cap:
            self.buffer = self.buffer[-self.buffer_cap :]
        self.t_step += 1
        if len(self.buffer) < max(self.batch, self.warmup):
            return
        for _ in range(self.n_updates):  # replay ratio > 1 to use samples efficiently
            idx = self.rng.integers(0, len(self.buffer), size=self.batch)
            batch = [self.buffer[i] for i in idx]
            s = torch.stack([self._feat(b[0]) for b in batch])
            a = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
            r = torch.as_tensor(
                [b[2] / self.cost_scale for b in batch], dtype=torch.float32
            )
            sn = torch.stack([self._feat(b[3]) for b in batch])
            q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.q_target(sn).max(dim=1).values
                target = r + self.gamma * q_next
            loss = nn.functional.smooth_l1_loss(q_sa, target)  # Huber, robust
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        if self.t_step % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())


# ---------------------------------------------------------------------------
# NN World Model + CEM model-predictive control (the star)
# ---------------------------------------------------------------------------


class WorldModelNet(nn.Module):
    """Two-head model: (scaled obs, scaled action, scaled demand) -> (obs delta,
    reward). Conditioning on the realized demand makes the transition a nearly
    deterministic function the net can learn sharply; demand is resampled from an
    empirical distribution at planning time so model rollouts carry the demand
    variance that drives backorder costs (an unconditioned mean model would be
    optimistically biased)."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + act_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.delta_head(h), self.reward_head(h).squeeze(-1)


def _echelon_from_obs_batch(obs, K, L, order_cap):
    """Echelon inventory positions and base-stock orders from a batch of flat
    observations. obs: (P, obs_dim). Returns orders (P, K) for target S (P, K)."""
    inv = obs[:, :K]
    pipe = obs[:, K : K + K * L].reshape(obs.shape[0], K, L)
    b = obs[:, -1:]
    cum_on = np.cumsum(inv, axis=1)  # sum_{j<=k} on-hand
    pipe_sum = pipe.sum(axis=2)  # (P,K)
    cum_dt = np.cumsum(pipe_sum, axis=1) - pipe_sum  # in-transit strictly downstream
    ip = cum_on + cum_dt - b + pipe_sum
    return ip


class WorldModelPlanner(Paradigm):
    """Learn the pipeline+demand dynamics with a neural world model, then plan
    by choosing echelon base-stock levels that minimize predicted cost when the
    learned model is rolled forward. This mirrors the oracle exactly, except the
    oracle searches base-stock levels on the TRUE model and this agent searches
    on its LEARNED model, so the regret gap is the price of model error. The
    base-stock policy class is the Clark-Scarf optimal form. A genuine learned
    NN world model drives planning on the physical pipeline state."""

    name = "NN World Model"

    def __init__(
        self,
        gamma,
        hidden,
        lr,
        buffer,
        batch,
        warmup,
        refit_every,
        refit_steps,
        replan_every,
        sim_horizon,
        sim_warm,
        search_radius,
        search_passes,
        damp,
        n_demand_paths,
        explore_std,
        order_cap,
        K,
        L,
        cost_scale,
    ):
        self.gamma = gamma
        self.hidden = hidden
        self.lr = lr
        self.buffer_cap = buffer
        self.batch = batch
        self.warmup = warmup
        self.refit_every = refit_every
        self.refit_steps = refit_steps
        self.replan_every = replan_every
        self.sim_H = sim_horizon
        self.sim_warm = sim_warm
        self.search_radius = search_radius
        self.search_passes = search_passes
        self.damp = damp
        self.n_demand_paths = n_demand_paths
        self.explore_std = explore_std
        self.order_cap = order_cap
        self.K = K
        self.L = L
        self.cost_scale = cost_scale

    def reset(self, env, S_oracle, seed=0):
        # The agent knows only the problem bounds (order_cap, inv_cap, K, L) and
        # what it observes; it must not read the true demand rate, holding, or
        # penalty. Normalization uses the known state/action bounds; the demand
        # scale and the search band are estimated from observed demand.
        torch.manual_seed(seed + 4242)
        self.rng = np.random.default_rng(seed + 4242)
        self.obs_dim = env._obs().shape[0]
        self.oscale = float(env.inv_cap)  # state bound (known)
        self.ascale = float(self.order_cap)  # action bound (known)
        self.dscale = float(self.order_cap)  # known bound; demand <= order_cap-scale
        self.inv_cap = env.inv_cap  # state bound (known)
        self.model = WorldModelNet(self.obs_dim, self.K, self.hidden)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer = []  # (obs, action, demand, reward, next_obs)
        self.demand_hist = []  # observed demands, for empirical resampling
        self.t_step = 0
        self.forecast_log = []
        self._fixed_paths = None  # demand paths fixed at first replan (CRN)
        self.lam_hat = None  # mean demand estimated from data at first replan
        # Neutral base-stock start from the order bound (no demand knowledge); the
        # model-based search moves it to the optimum. The search band S_max is set
        # from the estimated mean demand at the first replan (see _replan).
        self.S_hat = np.full(env.K, self.order_cap, dtype=np.int64)
        self.S_max = np.full(env.K, env.inv_cap, dtype=np.int64)
        self.reset_obs = env.reset().astype(np.float64)

    def _fit(self):
        if len(self.buffer) < self.batch:
            return
        for _ in range(self.refit_steps):
            idx = self.rng.integers(0, len(self.buffer), size=self.batch)
            batch = [self.buffer[i] for i in idx]
            o = torch.as_tensor(
                np.stack([b[0] for b in batch]) / self.oscale, dtype=torch.float32
            )
            a = torch.as_tensor(
                np.stack([b[1] for b in batch]) / self.ascale, dtype=torch.float32
            )
            dem = torch.as_tensor(
                np.array([[b[2]] for b in batch]) / self.dscale, dtype=torch.float32
            )
            on = torch.as_tensor(
                np.stack([b[4] for b in batch]) / self.oscale, dtype=torch.float32
            )
            rew = torch.as_tensor(
                np.array([b[3] for b in batch]) / self.cost_scale, dtype=torch.float32
            )
            delta_pred, r_pred = self.model(torch.cat([o, a, dem], dim=-1))
            loss = nn.functional.mse_loss(delta_pred, on - o) + nn.functional.mse_loss(
                r_pred, rew
            )
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def _model_cost_batch(self, S_batch, obs0, demand_paths):
        """Roll the learned model under each candidate base-stock policy S_batch
        (P, K) with sampled demand, returning mean per-period predicted cost per
        candidate. demand_paths (M, sim_H) is shared across candidates (common
        random numbers): row layout is (candidate p, path m) -> p*M + m."""
        P = S_batch.shape[0]
        M = demand_paths.shape[0]
        S_rep = np.repeat(S_batch, M, axis=0)  # (P*M, K)
        dem_rep = np.tile(demand_paths, (P, 1))  # (P*M, sim_H)
        o = np.tile(obs0, (P * M, 1))
        tot = np.zeros(P * M)
        n = 0
        with torch.no_grad():
            for t in range(self.sim_H):
                ip = _echelon_from_obs_batch(o, self.K, self.L, self.order_cap)
                orders = np.clip(S_rep - ip, 0, self.order_cap)
                inp = torch.as_tensor(
                    np.concatenate(
                        [
                            o / self.oscale,
                            orders / self.ascale,
                            dem_rep[:, t : t + 1] / self.dscale,
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                )
                delta, r = self.model(inp)
                o = np.clip(
                    (o / self.oscale + delta.numpy()) * self.oscale, 0, self.inv_cap
                )
                if t >= self.sim_warm:
                    tot += -r.numpy() * self.cost_scale
                    n += 1
        return (tot / max(1, n)).reshape(P, M).mean(axis=1)

    def _replan(self):
        """Coordinate search over echelon base-stock levels, evaluating each
        candidate on the learned model with common-random-number demand paths
        bootstrapped from observed demand."""
        if len(self.demand_hist) < 5:
            return
        # Estimate the mean demand from observed data (the retailer sees demand)
        # and set the search band from it; this replaces any knowledge of the
        # true demand rate. Done once, at the first replan.
        if self.lam_hat is None:
            self.lam_hat = max(1.0, float(np.mean(self.demand_hist)))
            # Standard base-stock start: one protection interval of estimated mean
            # demand per echelon depth. This is the same data-driven heuristic the
            # decentralized baseline uses; the model-based search below adds the
            # safety stock and echelon-coordination correction the heuristic lacks.
            self.S_hat = np.array(
                [
                    int(round(self.lam_hat * (self.L + 1) * (k + 1)))
                    for k in range(self.K)
                ],
                dtype=np.int64,
            )
            self.S_max = np.array(
                [
                    int(self.lam_hat * (self.L + 1) * (k + 1) * 1.8) + 4
                    for k in range(self.K)
                ],
                dtype=np.int64,
            )
        # Fix the demand paths once (common random numbers across all replans) so
        # the model cost surface is deterministic and S_hat converges rather than
        # chasing resampling noise. Demand is i.i.d., so an early sample suffices.
        if self._fixed_paths is None:
            dhist = np.array(self.demand_hist)
            self._fixed_paths = self.rng.choice(
                dhist, size=(self.n_demand_paths, self.sim_H)
            )
        demand_paths = self._fixed_paths
        S = self.S_hat.copy()
        for _ in range(self.search_passes):
            for k in range(self.K):
                lo = max(0, S[k] - self.search_radius)
                hi = min(self.S_max[k], S[k] + self.search_radius)
                cands = []
                for cand in range(lo, hi + 1):
                    St = S.copy()
                    St[k] = cand
                    St = np.maximum.accumulate(St)  # keep nested (feasible)
                    cands.append(St)
                cands = np.array(cands)
                costs = self._model_cost_batch(cands, self.reset_obs, demand_paths)
                S = cands[int(np.argmin(costs))]
        # damp the update toward the prior estimate to suppress replan-to-replan
        # jitter from the finite demand-sample cost estimate
        self.S_hat = np.round(self.damp * self.S_hat + (1 - self.damp) * S).astype(
            np.int64
        )
        self.S_hat = np.minimum(np.maximum.accumulate(self.S_hat), self.S_max)

    def act(self, obs, env, t):
        if self.t_step < self.warmup:
            # Sweep base-stock targets across the sane band so the model gets
            # coverage of high- and low-inventory operating points (not just the
            # region around the current estimate); this is what lets the planner
            # correctly price the upstream base-stock on every seed.
            frac = self.rng.uniform(0.4, 1.0, size=self.K)
            S_explore = np.round(frac * self.S_max).astype(np.int64)
            S_explore = np.maximum.accumulate(S_explore)
            return env.order_from_echelon(S_explore)
        orders = env.order_from_echelon(self.S_hat)
        decay = max(0.0, 1.0 - t / (0.6 * env.T))
        noise = self.rng.normal(0, self.explore_std * decay, size=self.K)
        return np.clip(orders + np.round(noise), 0, self.order_cap).astype(np.int64)

    def observe(self, obs, action, reward, next_obs, info):
        demand = info["demand"]
        if self.t_step >= self.warmup:
            with torch.no_grad():
                o = torch.as_tensor(obs / self.oscale, dtype=torch.float32)
                a = torch.as_tensor(action / self.ascale, dtype=torch.float32)
                dem = torch.as_tensor([demand / self.dscale], dtype=torch.float32)
                delta, _ = self.model(torch.cat([o, a, dem]).unsqueeze(0))
                pred_next = (o + delta.squeeze(0)).numpy() * self.oscale
            self.forecast_log.append(
                (self.t_step, float(np.mean((pred_next - next_obs) ** 2)))
            )
        self.buffer.append((obs, action, demand, reward, next_obs))
        self.demand_hist.append(demand)
        if len(self.buffer) > self.buffer_cap:
            self.buffer = self.buffer[-self.buffer_cap :]
        self.t_step += 1
        if self.t_step % self.refit_every == 0:
            self._fit()
        if self.t_step >= self.warmup and self.t_step % self.replan_every == 0:
            self._replan()


# ---------------------------------------------------------------------------
# Rollout + compute
# ---------------------------------------------------------------------------


def make_paradigm(name, config):
    if name == "Oracle":
        return OracleBaseStock()
    if name == "Naive":
        return NaiveConstant()
    if name == "Decentralized":
        return DecentralizedBaseStock(
            refit_every=config["REFIT_EVERY"], safety_z=config["SAFETY_Z"]
        )
    if name == "Model-Free DQN":
        return DQNAgent(
            gamma=config["GAMMA"],
            hidden=config["HIDDEN"],
            lr=config["LR"],
            buffer=config["BUFFER"],
            batch=config["BATCH"],
            eps_hi=config["EPS_HI"],
            eps_lo=config["EPS_LO"],
            target_sync=config["TARGET_SYNC"],
            warmup=config["WARMUP"],
            n_updates=config["N_UPDATES"],
            n_levels=config["N_ACTION_LEVELS"],
            order_cap=config["order_cap"],
            K=config["K"],
            cost_scale=config["COST_SCALE"],
            T_episode=config["T_EPISODE"],
        )
    if name == "NN World Model":
        return WorldModelPlanner(
            gamma=config["GAMMA"],
            hidden=config["HIDDEN"],
            lr=config["LR"],
            buffer=config["BUFFER"],
            batch=config["BATCH"],
            warmup=config["WARMUP"],
            refit_every=config["REFIT_EVERY"],
            refit_steps=config["REFIT_STEPS"],
            replan_every=config["REPLAN_EVERY"],
            sim_horizon=config["SIM_HORIZON"],
            sim_warm=config["SIM_WARM"],
            search_radius=config["SEARCH_RADIUS"],
            search_passes=config["SEARCH_PASSES"],
            damp=config["DAMP"],
            n_demand_paths=config["N_DEMAND_PATHS"],
            explore_std=config["EXPLORE_STD"],
            order_cap=config["order_cap"],
            K=config["K"],
            L=config["L"],
            cost_scale=config["COST_SCALE"],
        )
    raise ValueError(name)


def _env_from_config(config, seed):
    return MultiEchelonEnv(
        K=config["K"],
        L=config["L"],
        lam=config["lam"],
        p=config["p"],
        order_cap=config["order_cap"],
        inv_cap=config["inv_cap"],
        T=config["T_EPISODE"],
        seed=seed,
    )


def rollout(paradigm, config, S_oracle, seed, track=False):
    env = _env_from_config(config, seed)
    obs = env.reset()
    paradigm.reset(env, S_oracle, seed=seed)
    T = config["T_EPISODE"]
    rewards = np.zeros(T)
    orders_log = np.zeros((T, config["K"])) if track else None
    for t in range(T):
        action = paradigm.act(obs, env, t)
        next_obs, r, done, info = env.step(action)
        paradigm.observe(obs, action, r, next_obs, info)
        rewards[t] = r
        if track:
            orders_log[t] = info["orders"]
        obs = next_obs
        if done:
            break
    return rewards, orders_log


def compute_shared(config):
    env_params = dict(
        K=config["K"],
        L=config["L"],
        lam=config["lam"],
        p=config["p"],
        order_cap=config["order_cap"],
        inv_cap=config["inv_cap"],
    )
    S_star, mean_cost = find_oracle_base_stock(
        env_params,
        n_search_seq=config["ORACLE_SEARCH_SEQ"],
        T_search=config["ORACLE_SEARCH_T"],
    )
    print(
        f"    Oracle base-stock S* = {S_star}, "
        f"per-period cost = {mean_cost / config['ORACLE_SEARCH_T']:.3f}"
    )
    N, T = config["N_SEEDS"], config["T_EPISODE"]
    oracle_rewards = np.zeros((N, T))
    oracle = OracleBaseStock()
    for s in range(N):
        rew, _ = rollout(oracle, config, S_star, seed=s)
        oracle_rewards[s] = rew
    return dict(
        S_star=np.asarray(S_star), oracle_rewards=oracle_rewards, env_params=env_params
    )


def compute_paradigm(config, shared, name):
    N, T = config["N_SEEDS"], config["T_EPISODE"]
    W = config["TERMINAL_WINDOW"]
    S_star = shared["S_star"]
    oracle_rewards = shared["oracle_rewards"]
    regret_curves = np.zeros((N, T))
    final_regret = np.zeros(N)
    term_cost = np.zeros(N)  # asymptotic per-period cost (last-W window)
    order_var = np.zeros((N, config["K"]))  # variance of orders per stage
    forecast_curve = None
    for s in range(N):
        np.random.seed(s)
        paradigm = make_paradigm(name, config)
        track = s == 0
        rew, orders_log = rollout(paradigm, config, S_star, seed=s, track=track)
        curve = np.cumsum(oracle_rewards[s] - rew)
        regret_curves[s] = curve
        final_regret[s] = curve[-1]
        term_cost[s] = float(np.mean(-rew[-W:]))  # cost = -reward, last W steps
        if track and orders_log is not None:
            order_var[s] = orders_log[:, :].var(axis=0)
            if isinstance(paradigm, WorldModelPlanner) and paradigm.forecast_log:
                forecast_curve = np.array(paradigm.forecast_log)  # (steps, 2)
        else:
            order_var[s] = np.nan
    out = dict(
        regret_curves=regret_curves,
        final_regret=final_regret,
        mean_curve=regret_curves.mean(axis=0),
        se_curve=regret_curves.std(axis=0, ddof=1) / np.sqrt(N),
        final_mean=float(final_regret.mean()),
        final_se=float(final_regret.std(ddof=1) / np.sqrt(N)),
        term_cost_mean=float(term_cost.mean()),
        term_cost_se=float(term_cost.std(ddof=1) / np.sqrt(N)),
        order_var_seed0=order_var[0],
    )
    if forecast_curve is not None:
        out["forecast_curve"] = forecast_curve
    return out


PARADIGM_REGISTRY = {
    "Oracle": (compute_paradigm, ORACLE_CONFIG),
    "Naive": (compute_paradigm, NAIVE_CONFIG),
    "Decentralized": (compute_paradigm, DECENTRAL_CONFIG),
    "Model-Free DQN": (compute_paradigm, DQN_CONFIG),
    "NN World Model": (compute_paradigm, WM_CONFIG),
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


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

PARADIGM_ORDER = [
    "Oracle",
    "NN World Model",
    "Model-Free DQN",
    "Decentralized",
    "Naive",
]
PARADIGM_COLORS = {
    "Oracle": COLORS["black"],
    "NN World Model": COLORS["blue"],
    "Model-Free DQN": COLORS["orange"],
    "Decentralized": COLORS["green"],
    "Naive": COLORS["gray"],
}


def generate_outputs(data):
    apply_style()
    T = SHARED_CONFIG["T_EPISODE"]
    t_axis = np.arange(1, T + 1)
    ranked = sorted(PARADIGM_ORDER, key=lambda nm: data["results"][nm]["final_mean"])

    fig, (ax_reg, ax_pp, ax_fc) = plt.subplots(1, 3, figsize=FIG_TRIPLE)

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
    ax_reg.set_ylabel("cumulative regret (cost above oracle)")
    ax_reg.set_title("Cumulative regret (20 seeds, mean $\\pm$ SE)")
    ax_reg.legend(loc="upper left", fontsize=8)

    # Middle panel: per-period cost above the oracle over time, for the three
    # near-oracle methods only (the two failing methods are off-scale). The
    # per-period regret is the increment of the cumulative curve; it shows the
    # world model paying a high exploration cost early and then converging below
    # the decentralized heuristic, the asymptotic win the table reports as a
    # lower terminal cost.
    near_oracle = ["Oracle", "Decentralized", "NN World Model"]
    kpp = max(1, T // 25)  # rolling-mean window
    for name in near_oracle:
        pp = np.diff(data["results"][name]["mean_curve"], prepend=0.0)
        sm = np.convolve(pp, np.ones(kpp) / kpp, mode="valid")
        ax_pp.plot(
            t_axis[: len(sm)],
            sm,
            label=name,
            color=PARADIGM_COLORS[name],
            linewidth=1.7,
        )
    ax_pp.axhline(0, **BENCH_STYLE)
    ax_pp.set_ylim(bottom=-1.0)
    ax_pp.set_xlabel("environment step $t$")
    ax_pp.set_ylabel("per-period cost above oracle")
    ax_pp.set_title("Per-period regret, near-oracle methods")
    ax_pp.legend(loc="upper right", fontsize=8)

    # Right panel: NN world-model one-step forecast error over training,
    # showing the learned model sharpens as interaction data accumulates.
    wm = data["results"]["NN World Model"]
    if "forecast_curve" in wm:
        fc = wm["forecast_curve"]
        steps, mse = fc[:, 0], fc[:, 1]
        k = max(1, len(mse) // 12)  # rolling mean for readability
        sm = np.convolve(mse, np.ones(k) / k, mode="valid")
        ax_fc.plot(
            steps[: len(sm)], sm, color=PARADIGM_COLORS["NN World Model"], linewidth=1.6
        )
        ax_fc.set_yscale("log")
        ax_fc.set_xlabel("environment step $t$")
        ax_fc.set_ylabel("one-step forecast MSE (seed 0)")
        ax_fc.set_title("NN world-model forecast error over training")

    fig.suptitle("Multi-echelon supply chain: five learning paradigms", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "multi_echelon_paradigms.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # Results table (rank-ordered by final regret): integrated learning cost
    # (cumulative regret) plus asymptotic policy quality (terminal per-period cost).
    W = SHARED_CONFIG["TERMINAL_WINDOW"]
    tbl_path = os.path.join(OUTPUT_DIR, "multi_echelon_paradigms_results.tex")
    with open(tbl_path, "w") as f:
        f.write(
            f"% Cumulative regret vs Clark-Scarf oracle at T={T} and terminal "
            f"per-period cost (last {W} steps), mean +- SE over 20 seeds.\n"
        )
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Paradigm & Cumulative regret & Terminal cost/period \\\\\n\\midrule\n")
        for name in ranked:
            r = data["results"][name]
            f.write(
                f"{name} & {r['final_mean']:.1f} $\\pm$ {r['final_se']:.1f} & "
                f"{r['term_cost_mean']:.2f} $\\pm$ {r['term_cost_se']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  Table saved: {tbl_path}")

    # Stdout summary.
    print(
        f"\n=== Regret vs oracle at T={T} and terminal per-period cost "
        f"(last {W} steps), mean ± SE, 20 seeds ===\n"
    )
    print(f"{'Paradigm':<18} {'Cumulative regret':>20} {'Terminal cost/period':>22}")
    print("-" * 62)
    for name in ranked:
        r = data["results"][name]
        print(
            f"{name:<18}  {r['final_mean']:>10.1f} ± {r['final_se']:>5.1f}"
            f"  {r['term_cost_mean']:>12.2f} ± {r['term_cost_se']:>5.2f}"
        )

    print("\n=== Order variance by stage (seed 0; bullwhip) ===\n")
    print(
        f"{'Paradigm':<18}"
        + "".join(f"{'stage ' + str(k + 1):>10}" for k in range(SHARED_CONFIG["K"]))
    )
    print("-" * (18 + 10 * SHARED_CONFIG["K"]))
    for name in ranked:
        ov = data["results"][name].get("order_var_seed0")
        if ov is None or np.all(np.isnan(ov)):
            continue
        print(f"{name:<18}" + "".join(f"{v:>10.2f}" for v in ov))

    print(f"\n  Oracle base-stock S* = {data['shared']['S_star']}")


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
