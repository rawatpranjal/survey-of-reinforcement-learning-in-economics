# Planning on the Sutton blocking maze. Chapter: Forecasting, Dreaming and Learning.
# §3 simulation study. Compares five agents: plain Q-learning (K=0); Dyna-Q with
# K=5 and K=50; Dyna-Q+ (K=50, with the Sutton-Barto curiosity bonus); and a
# Schmidhuber 1990 controller-model agent with two small neural networks.

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import (
    apply_style, COLORS, BENCH_STYLE, FIG_SINGLE,
)
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()
import matplotlib.pyplot as plt

from dyna_maze_env import BlockingMaze

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'dyna_maze'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_CONFIG = dict(t_switch=1000, t_total=3000, episode_cap=200)

SHARED_CONFIG = dict(
    **ENV_CONFIG,
    N_SEEDS=30,
    ALPHA=0.1,
    GAMMA=0.95,
    EPSILON=0.1,
    BONUS_KAPPA=1e-4,
)

SCHMID_CONFIG = dict(
    **ENV_CONFIG,
    N_SEEDS=30,
    GAMMA=0.95,
    LR_M=3e-3,
    LR_C=3e-3,
    HIDDEN_DIM=32,
    K_PLAN_INTERVAL=10,
    H_PLAN=10,
    N_IMAGINED=16,
    ENTROPY_COEF=0.01,
    BUFFER_CAP=10_000,
)

AGENT_CONFIGS = {
    'Q-learning (K=0)':       {**SHARED_CONFIG, 'K': 0,  'bonus': False},
    'Dyna-Q (K=5)':           {**SHARED_CONFIG, 'K': 5,  'bonus': False},
    'Dyna-Q (K=50)':          {**SHARED_CONFIG, 'K': 50, 'bonus': False},
    'Dyna-Q+ (K=50)':         {**SHARED_CONFIG, 'K': 50, 'bonus': True},
    'Schmidhuber 1990 (NN)':  SCHMID_CONFIG,
}

AGENT_ORDER = [
    'Dyna-Q (K=50)',
    'Dyna-Q+ (K=50)',
    'Dyna-Q (K=5)',
    'Schmidhuber 1990 (NN)',
    'Q-learning (K=0)',
]
AGENT_COLORS = {
    'Q-learning (K=0)':       COLORS['gray'],
    'Dyna-Q (K=5)':           COLORS['orange'],
    'Dyna-Q (K=50)':          COLORS['blue'],
    'Dyna-Q+ (K=50)':         COLORS['green'],
    'Schmidhuber 1990 (NN)':  COLORS['purple'],
}

N_STATES = BlockingMaze.N_ROWS * BlockingMaze.N_COLS
N_ACTIONS = 4
GOAL_STATE_ID = BlockingMaze.GOAL[0] * BlockingMaze.N_COLS + BlockingMaze.GOAL[1]


# ---------------------------------------------------------------------------
# Dyna agent
# ---------------------------------------------------------------------------

class DynaAgent:
    """Tabular Dyna-Q / Dyna-Q+.

    Q[s, a]: value table.
    Model: dict (s, a) -> (r, s', t_last) where t_last is the env step at which
           this transition was last observed (used by Dyna-Q+).
    With K planning steps, after each real step the agent samples K
    previously-visited (s, a) pairs and applies the Q-update on the model's
    cached (r, s'). For Dyna-Q+, the planning reward gets a bonus
    kappa * sqrt(t_current - t_last(s, a)).
    """

    def __init__(self, n_states, n_actions, alpha, gamma, epsilon,
                 K, bonus, bonus_kappa, seed):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.K = K
        self.bonus = bonus
        self.bonus_kappa = bonus_kappa
        self.model = {}        # (s, a) -> (r, s', t_last)
        self.visited_sa = []   # list of (s, a) for uniform sampling
        self.visited_set = set()
        self.t_step = 0
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # Greedy with random tie-breaking among argmax.
        q_s = self.Q[s]
        max_q = q_s.max()
        ties = np.flatnonzero(q_s == max_q)
        return int(self.rng.choice(ties))

    def _q_update(self, s, a, r, s_next):
        td = r + self.gamma * self.Q[s_next].max() - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

    def observe(self, s, a, r, s_next):
        self.t_step += 1
        self._q_update(s, a, r, s_next)
        if (s, a) not in self.visited_set:
            self.visited_set.add((s, a))
            self.visited_sa.append((s, a))
        self.model[(s, a)] = (r, s_next, self.t_step)
        # Sutton & Barto §8.3: Dyna-Q+ registers all actions at every visited
        # state with t_last=0 so the bonus kappa * sqrt(t_step) drives the
        # agent toward untried actions. Without this, the bonus only mutates
        # the value of already-tried (s, a) and cannot enable discovery.
        if self.bonus:
            for state in (s, s_next):
                for a_prime in range(self.n_actions):
                    if (state, a_prime) not in self.visited_set:
                        self.visited_set.add((state, a_prime))
                        self.visited_sa.append((state, a_prime))
                        self.model[(state, a_prime)] = (0.0, state, 0)
        # Planning
        if self.K > 0 and self.visited_sa:
            idx = self.rng.integers(0, len(self.visited_sa), size=self.K)
            for i in idx:
                sp, ap = self.visited_sa[i]
                rp, snp, t_last = self.model[(sp, ap)]
                if self.bonus:
                    tau = max(0, self.t_step - t_last)
                    rp_p = rp + self.bonus_kappa * np.sqrt(tau)
                else:
                    rp_p = rp
                self._q_update(sp, ap, rp_p, snp)


# ---------------------------------------------------------------------------
# Schmidhuber 1990 controller-model agent (neural)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Schmidhuber 1990 model network with two heads.

    trunk:        one-hot(state) || one-hot(action) -> hidden
    state_head:   hidden -> next-state logits (n_states-way classification)
    reward_head:  hidden -> scalar reward prediction
    """

    def __init__(self, n_states, n_actions, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, n_states)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.state_head(h), self.reward_head(h).squeeze(-1)


class Schmidhuber1990Agent:
    """Two-network controller-model architecture (Schmidhuber 1990).

    Model M_theta:  one-hot(state) + one-hot(action) -> (next-state logits, reward).
    Controller C_phi: one-hot(state) -> action logits (4-way).

    Each real step: store (s, a, r, s') in a replay buffer; fit M for one
    minibatch step on the joint loss (cross-entropy on next state + MSE on
    reward). Every K_plan_interval real steps: roll H_plan-step imagined
    trajectories under M starting from random replay states, compute the
    REINFORCE policy gradient with a moving-average baseline plus entropy
    regularization, and apply one gradient step on C. Reward inside imagined
    rollouts comes from the learned reward head, not the true goal location.
    """

    def __init__(self, n_states, n_actions, gamma,
                 lr_m, lr_c, hidden_dim, K_plan_interval, H_plan,
                 n_imagined, entropy_coef, goal_state_id,
                 buffer_cap, seed):
        torch.manual_seed(seed)
        self.n_states, self.n_actions = n_states, n_actions
        self.gamma = gamma
        self.K_plan_interval = K_plan_interval
        self.H_plan = H_plan
        self.n_imagined = n_imagined
        self.entropy_coef = entropy_coef
        # Stored for diagnostics only (e.g. plotting). Never used in training.
        self.goal_state_id_for_diagnostics = goal_state_id
        self.buffer = []
        self.buffer_cap = buffer_cap
        self.t_step = 0
        self.baseline = 0.0
        self.baseline_alpha = 0.05

        self.M = WorldModel(n_states, n_actions, hidden_dim)
        self.C = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.opt_m = torch.optim.Adam(self.M.parameters(), lr=lr_m)
        self.opt_c = torch.optim.Adam(self.C.parameters(), lr=lr_c)
        self.rng = np.random.default_rng(seed + 31415)

    def _one_hot_state(self, s):
        v = torch.zeros(self.n_states)
        v[s] = 1.0
        return v

    def _one_hot_states(self, s_arr):
        v = torch.zeros(len(s_arr), self.n_states)
        v[torch.arange(len(s_arr)),
          torch.as_tensor(np.asarray(s_arr), dtype=torch.long)] = 1.0
        return v

    def _one_hot_actions(self, a_arr):
        v = torch.zeros(len(a_arr), self.n_actions)
        v[torch.arange(len(a_arr)),
          torch.as_tensor(np.asarray(a_arr), dtype=torch.long)] = 1.0
        return v

    def act(self, s):
        with torch.no_grad():
            logits = self.C(self._one_hot_state(s).unsqueeze(0))
            probs = F.softmax(logits.squeeze(0), dim=-1).numpy()
        return int(self.rng.choice(self.n_actions, p=probs))

    def observe(self, s, a, r, s_next):
        self.t_step += 1
        self.buffer.append((s, a, r, s_next))
        if len(self.buffer) > self.buffer_cap:
            self.buffer = self.buffer[-self.buffer_cap:]
        self._train_m_step()
        if self.t_step % self.K_plan_interval == 0:
            self._train_c_reinforce()

    def _train_m_step(self):
        if len(self.buffer) < 8:
            return
        batch_size = min(32, len(self.buffer))
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s_arr = [b[0] for b in batch]
        a_arr = [b[1] for b in batch]
        r_arr = [b[2] for b in batch]
        sn_arr = [b[3] for b in batch]
        x = torch.cat([self._one_hot_states(s_arr),
                       self._one_hot_actions(a_arr)], dim=-1)
        s_target = torch.as_tensor(np.asarray(sn_arr), dtype=torch.long)
        r_target = torch.as_tensor(np.asarray(r_arr), dtype=torch.float32)
        s_logits, r_pred = self.M(x)
        loss_state = F.cross_entropy(s_logits, s_target)
        loss_reward = F.mse_loss(r_pred, r_target)
        loss = loss_state + loss_reward
        self.opt_m.zero_grad()
        loss.backward()
        self.opt_m.step()

    def _train_c_reinforce(self):
        if len(self.buffer) < 8:
            return
        idx = self.rng.integers(0, len(self.buffer), size=self.n_imagined)
        s_init = [self.buffer[i][0] for i in idx]
        s = self._one_hot_states(s_init)
        total_logprob = torch.zeros(self.n_imagined)
        total_return = torch.zeros(self.n_imagined)
        total_entropy = torch.zeros(self.n_imagined)
        gamma_t = 1.0
        for _ in range(self.H_plan):
            logits_c = self.C(s)
            dist = torch.distributions.Categorical(logits=logits_c)
            a = dist.sample()
            total_logprob = total_logprob + dist.log_prob(a)
            total_entropy = total_entropy + dist.entropy()
            a_oh = F.one_hot(a, num_classes=self.n_actions).float()
            with torch.no_grad():
                s_logits, r_pred = self.M(torch.cat([s, a_oh], dim=-1))
                s_next_idx = torch.distributions.Categorical(logits=s_logits).sample()
            # Reward comes from the learned reward head, not the goal location.
            r = r_pred
            total_return = total_return + gamma_t * r
            gamma_t *= self.gamma
            s = F.one_hot(s_next_idx, num_classes=self.n_states).float()
        self.baseline = ((1 - self.baseline_alpha) * self.baseline
                         + self.baseline_alpha * total_return.mean().item())
        advantages = (total_return - self.baseline).detach()
        loss = -(total_logprob * advantages).mean() - self.entropy_coef * total_entropy.mean()
        self.opt_c.zero_grad()
        loss.backward()
        self.opt_c.step()


# ---------------------------------------------------------------------------
# Agent factory + Rollout: t_total real env steps, episodes resetting on goal
# ---------------------------------------------------------------------------

def make_agent(agent_name, config, seed):
    if agent_name == 'Schmidhuber 1990 (NN)':
        return Schmidhuber1990Agent(
            n_states=N_STATES, n_actions=N_ACTIONS,
            gamma=config['GAMMA'],
            lr_m=config['LR_M'], lr_c=config['LR_C'],
            hidden_dim=config['HIDDEN_DIM'],
            K_plan_interval=config['K_PLAN_INTERVAL'],
            H_plan=config['H_PLAN'],
            n_imagined=config['N_IMAGINED'],
            entropy_coef=config['ENTROPY_COEF'],
            goal_state_id=GOAL_STATE_ID,
            buffer_cap=config['BUFFER_CAP'],
            seed=seed,
        )
    return DynaAgent(
        n_states=N_STATES, n_actions=N_ACTIONS,
        alpha=config['ALPHA'], gamma=config['GAMMA'],
        epsilon=config['EPSILON'],
        K=config['K'], bonus=config['bonus'],
        bonus_kappa=config['BONUS_KAPPA'],
        seed=seed,
    )


def rollout(agent_name, config, seed):
    env = BlockingMaze(t_switch=config['t_switch'],
                       t_total=config['t_total'],
                       episode_cap=config['episode_cap'])
    env.reset_global()
    pos = env.reset()
    agent = make_agent(agent_name, config, seed)
    s = env.state_id(pos)
    cum_reward_curve = np.zeros(config['t_total'])
    cum_r = 0.0
    for t in range(config['t_total']):
        a = agent.act(s)
        pos_next, r, done, info = env.step(a)
        s_next = env.state_id(pos_next)
        agent.observe(s, a, r, s_next)
        cum_r += r
        cum_reward_curve[t] = cum_r
        s = s_next
        if done:
            pos = env.reset()
            s = env.state_id(pos)
    return cum_reward_curve


def compute_agent(config, agent_name):
    N = config['N_SEEDS']
    T = config['t_total']
    curves = np.zeros((N, T))
    for seed in range(N):
        curves[seed] = rollout(agent_name, config, seed=seed)
    return dict(
        curves=curves,
        mean=curves.mean(axis=0),
        se=curves.std(axis=0, ddof=1) / np.sqrt(N),
        final_mean=float(curves[:, -1].mean()),
        final_se=float(curves[:, -1].std(ddof=1) / np.sqrt(N)),
        phase1_final=float(curves[:, config['t_switch'] - 1].mean()),
        phase1_se=float(curves[:, config['t_switch'] - 1].std(ddof=1) / np.sqrt(N)),
        phase2_gain_mean=float((curves[:, -1] - curves[:, config['t_switch'] - 1]).mean()),
        phase2_gain_se=float((curves[:, -1] - curves[:, config['t_switch'] - 1]).std(ddof=1) / np.sqrt(N)),
    )


def compute_data(force=None):
    force = force or set()
    out = {}
    for name in AGENT_ORDER:
        cfg = AGENT_CONFIGS[name]
        cache_key = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', 'plus')
        out[name] = compute_or_load(
            CACHE_DIR, SCRIPT_NAME, cache_key, cfg,
            compute_agent, cfg, name,
            force=(name in force),
        )
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def generate_outputs(data):
    apply_style()
    T = SHARED_CONFIG['t_total']
    t_switch = SHARED_CONFIG['t_switch']
    t_axis = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in AGENT_ORDER:
        res = data[name]
        mean, se = res['mean'], res['se']
        ax.plot(t_axis, mean, label=name, color=AGENT_COLORS[name], linewidth=1.7)
        ax.fill_between(t_axis, mean - se, mean + se, color=AGENT_COLORS[name], alpha=0.18)
    ax.axvline(t_switch, **BENCH_STYLE)
    ax.text(t_switch + 25, 0.5, 'wall flip', fontsize=9, alpha=0.7)
    ax.set_xlabel('environment step $t$')
    ax.set_ylabel('cumulative reward')
    ax.set_title('Dyna planning amplification on the blocking maze '
                 '(30 seeds, mean $\\pm$ SE)')
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'dyna_maze.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    tbl_path = os.path.join(OUTPUT_DIR, 'dyna_maze_results.tex')
    with open(tbl_path, 'w') as f:
        f.write('% Cumulative reward at end of Phase 1 (t=1000), gain over Phase 2 (last 2000 steps), and total at t=3000.\n')
        f.write('% Mean +/- SE across 30 seeds.\n')
        f.write('\\begin{tabular}{lrrr}\n')
        f.write('\\toprule\n')
        f.write('Agent & End of Phase 1 & Phase 2 gain & Total ($t = 3000$) \\\\\n')
        f.write('\\midrule\n')
        for name in AGENT_ORDER:
            r = data[name]
            row = (f"{name} & {r['phase1_final']:.1f} $\\pm$ {r['phase1_se']:.1f} & "
                   f"{r['phase2_gain_mean']:.1f} $\\pm$ {r['phase2_gain_se']:.1f} & "
                   f"{r['final_mean']:.1f} $\\pm$ {r['final_se']:.1f} \\\\\n")
            f.write(row)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
    print(f"  Table saved: {tbl_path}")

    print("\n=== Cumulative reward (mean ± SE, n=30 seeds) ===\n")
    print(f"{'Agent':<22} {'Phase 1 end':>15} {'Phase 2 gain':>15} {'Total t=3000':>15}")
    print('-' * 70)
    for name in AGENT_ORDER:
        r = data[name]
        print(f"{name:<22} {r['phase1_final']:>10.1f} ± {r['phase1_se']:>3.1f}  "
              f"{r['phase2_gain_mean']:>10.1f} ± {r['phase2_gain_se']:>3.1f}  "
              f"{r['final_mean']:>10.1f} ± {r['final_se']:>3.1f}")


def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    print(f"=== {SCRIPT_NAME} ===")
    print(f"Agents: {AGENT_ORDER}")
    print(f"Phase switch at t={SHARED_CONFIG['t_switch']}, total t={SHARED_CONFIG['t_total']}")
    print(f"N_SEEDS = {SHARED_CONFIG['N_SEEDS']}\n")

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
