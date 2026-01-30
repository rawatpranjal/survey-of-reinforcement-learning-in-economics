"""
Data Center Cooling Benchmark: Scaling Analysis (Chapter 3)

Multi-zone data center cooling control. The agent sets cooling intensity per zone
to minimize energy cost while avoiding overheating. Systematic scaling sweep
across number of zones.
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic,
    compute_policy_entropy, compute_policy_agreement, state_coverage,
    compute_q_error, evaluate_with_decomposition, capture_stdout,
    make_scaling_table, DP_FEASIBLE_THRESHOLD
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEMP_BINS = 6
LOAD_BINS = 4
EXT_TEMP_BINS = 3
COOLING_LEVELS = 4
COOLING_VALUES = [0.25, 0.50, 0.75, 1.00]

# Scaling sweep configuration
# N=2: 1,728 states (DP feasible), N=4: ~1M states (DP infeasible)
COMPLEXITY_SWEEP = [2, 4]  # Number of zones: small (DP works) vs large (DP breaks)
SEEDS = [42, 123, 7]  # 3 seeds for faster runs
NUM_EPISODES = 500
EPISODE_HORIZON = 20
EVAL_FREQ = 50
EVAL_EPISODES = 10

ALPHA = 0.4
BETA = 0.5
NOISE_STD = 0.3
EXT_INFLUENCE = 0.15

ENERGY_COST_BASE = 1.0
ENERGY_COST_SCALE = 2.0
OVERHEAT_PENALTY = 20.0
OVERHEAT_THRESHOLD = 4
TARGET_TEMP = 2
COMFORT_PENALTY = 0.5

GAMMA = 0.95

OUTPUT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = 'datacenter'

LOAD_TRANSITION = np.array([
    [0.5, 0.3, 0.15, 0.05],
    [0.2, 0.4, 0.3, 0.1],
    [0.1, 0.2, 0.4, 0.3],
    [0.05, 0.15, 0.3, 0.5],
])

EXT_TRANSITION = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
])

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class DataCenterCooling(EconBenchmark):
    """Data center cooling MDP.

    State: (temp_1, ..., temp_N, load_1, ..., load_N, ext_temp)
    Action: single integer encoding cooling levels for all zones
    Reward: -(energy_cost + overheat_penalty + comfort_penalty)
    """

    def __init__(self, N_zones, gamma=GAMMA, horizon=EPISODE_HORIZON):
        self.N_zones = N_zones
        self.complexity_param = N_zones
        self.gamma = gamma
        self.horizon = horizon

        self._num_states = (TEMP_BINS ** N_zones) * (LOAD_BINS ** N_zones) * EXT_TEMP_BINS
        self._num_actions = COOLING_LEVELS ** N_zones

        self.dp_feasible = self._num_states <= DP_FEASIBLE_THRESHOLD

        self.state = None
        self.t = 0

        # Track last step for decomposition
        self._last_energy_cost = 0.0
        self._last_overheat_cost = 0.0
        self._last_comfort_cost = 0.0
        self._last_temps = None
        self._last_overheated = []

        self.reset()

    def reset(self):
        temps = tuple([TARGET_TEMP] * self.N_zones)
        loads = tuple([1] * self.N_zones)
        ext = 1
        self.state = temps + loads + (ext,)
        self.t = 0
        return self.state

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def _decode_action(self, action_idx):
        cooling = []
        idx = action_idx
        for _ in range(self.N_zones):
            cooling.append(idx % COOLING_LEVELS)
            idx //= COOLING_LEVELS
        return cooling

    def _encode_action(self, cooling_list):
        idx = 0
        mult = 1
        for c in cooling_list:
            idx += c * mult
            mult *= COOLING_LEVELS
        return idx

    def state_to_index(self, state):
        N = self.N_zones
        temps = state[:N]
        loads = state[N:2*N]
        ext = state[2*N]

        idx = 0
        mult = 1
        for i in range(N):
            idx += temps[i] * mult
            mult *= TEMP_BINS
        for i in range(N):
            idx += loads[i] * mult
            mult *= LOAD_BINS
        idx += ext * mult
        return idx

    def index_to_state(self, idx):
        N = self.N_zones
        temps = []
        for _ in range(N):
            temps.append(idx % TEMP_BINS)
            idx //= TEMP_BINS
        loads = []
        for _ in range(N):
            loads.append(idx % LOAD_BINS)
            idx //= LOAD_BINS
        ext = idx % EXT_TEMP_BINS
        return tuple(temps) + tuple(loads) + (ext,)

    def state_to_features(self, state):
        N = self.N_zones
        temps = state[:N]
        loads = state[N:2*N]
        ext = state[2*N]
        features = []
        for t in temps:
            features.append(t / max(TEMP_BINS - 1, 1))
        for l in loads:
            features.append(l / max(LOAD_BINS - 1, 1))
        features.append(ext / max(EXT_TEMP_BINS - 1, 1))
        return np.array(features, dtype=np.float32)

    def _compute_reward(self, temps, cooling_indices, ext):
        energy_cost = 0.0
        overheat_cost = 0.0
        comfort_cost = 0.0
        overheated = []

        for k in range(self.N_zones):
            c = COOLING_VALUES[cooling_indices[k]]
            energy = ENERGY_COST_BASE * c + ENERGY_COST_SCALE * c * c
            energy_cost += energy

            if temps[k] >= OVERHEAT_THRESHOLD:
                overheat_cost += OVERHEAT_PENALTY * (temps[k] - OVERHEAT_THRESHOLD + 1)
                overheated.append(k)

            comfort_cost += COMFORT_PENALTY * abs(temps[k] - TARGET_TEMP)

        # Store for decomposition
        self._last_energy_cost = energy_cost
        self._last_overheat_cost = overheat_cost
        self._last_comfort_cost = comfort_cost
        self._last_temps = list(temps)
        self._last_overheated = overheated

        total_cost = energy_cost + overheat_cost + comfort_cost
        return -total_cost

    def _next_temp(self, temp, load, cooling_idx, ext, noise=None):
        c = COOLING_VALUES[cooling_idx]
        temp_continuous = temp + ALPHA * load - BETA * c + EXT_INFLUENCE * ext
        if noise is not None:
            temp_continuous += noise
        else:
            temp_continuous += np.random.normal(0, NOISE_STD)
        new_temp = int(round(np.clip(temp_continuous, 0, TEMP_BINS - 1)))
        return new_temp

    def step(self, action):
        N = self.N_zones
        temps = list(self.state[:N])
        loads = list(self.state[N:2*N])
        ext = self.state[2*N]

        cooling = self._decode_action(action)
        reward = self._compute_reward(temps, cooling, ext)

        new_temps = []
        for k in range(N):
            new_t = self._next_temp(temps[k], loads[k], cooling[k], ext)
            new_temps.append(new_t)

        new_loads = []
        for k in range(N):
            new_l = np.random.choice(LOAD_BINS, p=LOAD_TRANSITION[loads[k]])
            new_loads.append(new_l)

        new_ext = np.random.choice(EXT_TEMP_BINS, p=EXT_TRANSITION[ext])

        self.state = tuple(new_temps) + tuple(new_loads) + (new_ext,)
        self.t += 1
        done = (self.t >= self.horizon)

        return self.state, reward, done

    def transition_distribution(self, state, action):
        N = self.N_zones
        temps = list(state[:N])
        loads = list(state[N:2*N])
        ext = state[2*N]

        cooling = self._decode_action(action)

        noise_vals = [-1, 0, 1]
        noise_probs = [0.2, 0.6, 0.2]

        transitions = {}

        def recurse_temps(zone, current_temps, current_prob):
            if zone == N:
                recurse_loads(0, current_temps, [], current_prob)
                return
            for ni, nv in enumerate(noise_vals):
                np_val = noise_probs[ni]
                c = COOLING_VALUES[cooling[zone]]
                temp_cont = temps[zone] + ALPHA * loads[zone] - BETA * c + EXT_INFLUENCE * ext + nv * NOISE_STD
                new_t = int(round(np.clip(temp_cont, 0, TEMP_BINS - 1)))
                recurse_temps(zone + 1, current_temps + [new_t], current_prob * np_val)

        def recurse_loads(zone, final_temps, current_loads, current_prob):
            if zone == N:
                recurse_ext(final_temps, current_loads, current_prob)
                return
            for nl in range(LOAD_BINS):
                p_load = LOAD_TRANSITION[loads[zone], nl]
                if p_load < 1e-10:
                    continue
                recurse_loads(zone + 1, final_temps, current_loads + [nl], current_prob * p_load)

        def recurse_ext(final_temps, final_loads, current_prob):
            for ne in range(EXT_TEMP_BINS):
                p_ext = EXT_TRANSITION[ext, ne]
                if p_ext < 1e-10:
                    continue
                ns = tuple(final_temps) + tuple(final_loads) + (ne,)
                prob = current_prob * p_ext
                if ns in transitions:
                    transitions[ns] += prob
                else:
                    transitions[ns] = prob

        recurse_temps(0, [], 1.0)

        return [(s, p) for s, p in transitions.items()]

    def expected_reward(self, state, action):
        N = self.N_zones
        temps = list(state[:N])
        ext = state[2*N]
        cooling = self._decode_action(action)
        return self._compute_reward(temps, cooling, ext)


# ---------------------------------------------------------------------------
# Decomposition function for datacenter
# ---------------------------------------------------------------------------
def datacenter_decompose(state, action, reward, next_state, env):
    """Decompose cost into components for datacenter environment."""
    N = env.N_zones

    components = {
        'total_cost': -reward,
        'energy_cost': env._last_energy_cost,
        'overheat_cost': env._last_overheat_cost,
        'comfort_cost': env._last_comfort_cost,
        'overheated': 1.0 if len(env._last_overheated) > 0 else 0.0,
        'num_zones_overheated': len(env._last_overheated),
    }

    # Per-zone temperatures
    if env._last_temps:
        for k in range(N):
            components[f'zone_{k}_temp'] = env._last_temps[k]

    return components


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def make_heuristic_fixed(N_zones):
    """Fixed setpoint: always use medium cooling (0.50) for all zones."""
    fixed_cooling = [1] * N_zones

    def heuristic(state):
        env_dummy = DataCenterCooling.__new__(DataCenterCooling)
        env_dummy.N_zones = N_zones
        return env_dummy._encode_action(fixed_cooling)

    return heuristic


def make_heuristic_load_proportional(N_zones):
    """Load-proportional: cooling intensity proportional to server load."""
    def heuristic(state):
        loads = list(state[N_zones:2*N_zones])
        cooling = []
        for k in range(N_zones):
            c_idx = min(loads[k], COOLING_LEVELS - 1)
            cooling.append(c_idx)
        env_dummy = DataCenterCooling.__new__(DataCenterCooling)
        env_dummy.N_zones = N_zones
        return env_dummy._encode_action(cooling)

    return heuristic


def make_heuristic_pid(N_zones):
    """PID controller: adjust cooling based on temperature error."""
    integral = [0.0] * N_zones
    Kp = 0.5
    Ki = 0.1

    def heuristic(state):
        temps = list(state[:N_zones])
        cooling = []
        for k in range(N_zones):
            error = temps[k] - TARGET_TEMP
            integral[k] = integral[k] * 0.9 + error
            control = Kp * error + Ki * integral[k]
            c_continuous = 0.25 + 0.25 * max(0, control)
            c_continuous = np.clip(c_continuous, 0.25, 1.0)
            best_idx = 0
            best_diff = float('inf')
            for i, cv in enumerate(COOLING_VALUES):
                d = abs(cv - c_continuous)
                if d < best_diff:
                    best_diff = d
                    best_idx = i
            cooling.append(best_idx)
        env_dummy = DataCenterCooling.__new__(DataCenterCooling)
        env_dummy.N_zones = N_zones
        return env_dummy._encode_action(cooling)

    return heuristic


# ---------------------------------------------------------------------------
# Run single complexity level
# ---------------------------------------------------------------------------
def run_single_complexity(N_zones, seeds):
    """Run benchmark for a single complexity level (N_zones)."""
    env = DataCenterCooling(N_zones=N_zones)
    result = {
        'complexity': N_zones,
        'states': env.num_states,
        'dp_feasible': env.dp_feasible,
    }

    print(f"\n  N={N_zones}: {env.num_states:,} states, DP feasible={env.dp_feasible}")

    # --- Value Iteration (if feasible) ---
    V, policy, vi_metrics = None, None, None
    if env.dp_feasible:
        print("    Running Value Iteration...")
        V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
        dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                       n_episodes=200, horizon=EPISODE_HORIZON)
        result['dp_reward'] = dp_reward
        result['dp_time'] = vi_metrics.wall_time
        result['dp_iterations'] = vi_metrics.iterations
        print(f"      VI: {vi_metrics.iterations} iter, time={vi_metrics.wall_time:.2f}s, "
              f"reward={dp_reward:.2f}")
    else:
        result['dp_reward'] = None
        result['dp_time'] = None
        result['dp_iterations'] = None
        print("    DP skipped (state space too large)")

    # --- DQN (multiple seeds) ---
    print(f"    Running DQN ({len(seeds)} seeds)...")
    dqn_rewards = []
    dqn_times = []
    dqn_q_errors = []
    dqn_agreements = []
    all_curves = []
    checkpoint_episodes = None

    for i, seed in enumerate(seeds):
        env_dqn = DataCenterCooling(N_zones=N_zones)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            desc=f"N={N_zones} seed {i+1}/{len(seeds)}"
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=EPISODE_HORIZON)
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)
        all_curves.append(metrics.eval_checkpoints)
        checkpoint_episodes = metrics.checkpoint_episodes

        if V is not None:
            q_err = compute_q_error(q_net, V, env, n_samples=500, gamma=GAMMA)
            dqn_q_errors.append(q_err)
            agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
            dqn_agreements.append(agreement)

        print(f"      seed={seed}: reward={reward:.2f}, time={metrics.wall_time:.2f}s")

    result['dqn_reward_mean'] = np.mean(dqn_rewards)
    result['dqn_reward_std'] = np.std(dqn_rewards)
    result['dqn_rewards'] = dqn_rewards
    result['dqn_time_mean'] = np.mean(dqn_times)
    result['dqn_curves'] = all_curves
    result['checkpoint_episodes'] = checkpoint_episodes

    if dqn_q_errors:
        result['q_error'] = np.mean(dqn_q_errors)
        result['agreement'] = np.mean(dqn_agreements)
    else:
        result['q_error'] = None
        result['agreement'] = None

    # --- Heuristics ---
    print("    Running Heuristics...")
    h_fixed = make_heuristic_fixed(N_zones)
    h_fixed_reward = evaluate_heuristic(env, h_fixed,
                                         n_episodes=200, horizon=EPISODE_HORIZON)

    h_load = make_heuristic_load_proportional(N_zones)
    h_load_reward = evaluate_heuristic(env, h_load,
                                        n_episodes=200, horizon=EPISODE_HORIZON)

    h_pid = make_heuristic_pid(N_zones)
    h_pid_reward = evaluate_heuristic(env, h_pid,
                                       n_episodes=200, horizon=EPISODE_HORIZON)

    result['heuristics'] = {
        'fixed': h_fixed_reward,
        'load_proportional': h_load_reward,
        'pid': h_pid_reward,
    }
    print(f"      Fixed={h_fixed_reward:.2f}, Load-prop={h_load_reward:.2f}, "
          f"PID={h_pid_reward:.2f}")

    # --- Decomposition (using DQN policy) ---
    if all_curves:
        print("    Running decomposition analysis...")
        env_decomp = DataCenterCooling(N_zones=N_zones)
        q_net_decomp, _ = run_dqn(
            env_decomp, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=EPISODE_HORIZON, seed=42, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            show_progress=False
        )

        def dqn_policy(s):
            s_feat = env_decomp.state_to_features(s)
            with torch.no_grad():
                q_vals = q_net_decomp(torch.FloatTensor(s_feat).unsqueeze(0))
                return q_vals.argmax(dim=1).item()

        decomp = evaluate_with_decomposition(
            env_decomp, dqn_policy, n_episodes=100, horizon=EPISODE_HORIZON,
            decompose_fn=datacenter_decompose
        )
        result['decomposition'] = decomp.component_rewards
        result['overheat_frequency'] = decomp.component_rewards.get('overheated', 0)

    return result


# ---------------------------------------------------------------------------
# Scaling sweep
# ---------------------------------------------------------------------------
def run_scaling_sweep():
    """Run benchmark across all complexity levels."""
    print("=" * 70)
    print("  Data Center Cooling: Scaling Analysis")
    print("=" * 70)
    print(f"  Complexity sweep: N in {COMPLEXITY_SWEEP}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {NUM_EPISODES}, Horizon: {EPISODE_HORIZON}")

    results = []
    for N in COMPLEXITY_SWEEP:
        result = run_single_complexity(N, SEEDS)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(results):
    """Generate scaling comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reward vs complexity
    ax = axes[0]
    complexities = [r['complexity'] for r in results]
    dp_rewards = [r['dp_reward'] if r['dp_reward'] else np.nan for r in results]
    dqn_means = [r['dqn_reward_mean'] for r in results]
    dqn_stds = [r['dqn_reward_std'] for r in results]

    x = np.arange(len(complexities))
    width = 0.35

    ax.bar(x - width/2, dp_rewards, width, label='DP', color=DP_COLOR, alpha=0.8)
    ax.bar(x + width/2, dqn_means, width, yerr=dqn_stds, label='DQN',
           color=DQN_COLOR, alpha=0.8, capsize=3)

    ax.set_xlabel('Number of Zones (N)', fontsize=11)
    ax.set_ylabel('Episode Reward (negative cost)', fontsize=11)
    ax.set_title('Datacenter: Reward vs Complexity', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(complexities)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Learning curves for each complexity
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    for i, r in enumerate(results):
        N = r['complexity']
        curves = r['dqn_curves']
        episodes = r['checkpoint_episodes']

        if curves and episodes:
            min_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:min_len] for c in curves])
            episodes = episodes[:min_len]

            mean_rewards = np.mean(curves_arr, axis=0)
            std_rewards = np.std(curves_arr, axis=0)

            ax.plot(episodes, mean_rewards, color=colors[i], linewidth=2, label=f'N={N}')
            ax.fill_between(episodes, mean_rewards - std_rewards,
                           mean_rewards + std_rewards, color=colors[i], alpha=0.15)

        if r['dp_reward']:
            ax.axhline(r['dp_reward'], color=colors[i], linestyle='--',
                      linewidth=1, alpha=0.7)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Reward (negative cost)', fontsize=11)
    ax.set_title('Datacenter: Learning Curves by Complexity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def make_latex_tables(results):
    """Generate LaTeX tables for scaling results."""
    tex = make_scaling_table(results, 'N', 'Datacenter')
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")

    # Decomposition table for largest complexity
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        lines = []
        lines.append(r'\begin{tabular}{lr}')
        lines.append(r'\toprule')
        lines.append(r'Cost Component & Mean per Episode \\')
        lines.append(r'\midrule')

        decomp = largest['decomposition']
        lines.append(f"Total Cost & ${decomp.get('total_cost', 0):.2f}$ \\\\")
        lines.append(f"Energy Cost & ${decomp.get('energy_cost', 0):.2f}$ \\\\")
        lines.append(f"Overheat Cost & ${decomp.get('overheat_cost', 0):.2f}$ \\\\")
        lines.append(f"Comfort Cost & ${decomp.get('comfort_cost', 0):.2f}$ \\\\")
        lines.append(f"Overheat Freq. & ${decomp.get('overheated', 0):.2%}$ \\\\")

        N = largest['complexity']
        for k in range(N):
            key = f'zone_{k}_temp'
            if key in decomp:
                lines.append(f"Zone {k} Avg Temp & ${decomp[key]:.2f}$ \\\\")

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        decomp_tex = '\n'.join(lines)
        decomp_path = OUTPUT_DIR / f'{SCRIPT_NAME}_decomposition.tex'
        with open(decomp_path, 'w') as f:
            f.write(decomp_tex)
        print(f"Decomposition table saved to {decomp_path}")


# ---------------------------------------------------------------------------
# Detailed stdout output
# ---------------------------------------------------------------------------
def print_detailed_results(results):
    """Print detailed results to stdout."""
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)

    # Scaling summary table
    print("\n  Scaling Summary:")
    print("  " + "-" * 70)
    print(f"  {'N':>3} | {'States':>10} | {'DP Reward':>10} | {'DQN Reward':>18} | "
          f"{'Q-Error':>8} | {'Agr.':>6}")
    print("  " + "-" * 70)

    for r in results:
        N = r['complexity']
        states = r['states']
        dp_r = f"{r['dp_reward']:.2f}" if r['dp_reward'] else "---"
        dqn_r = f"{r['dqn_reward_mean']:.2f} +/- {r['dqn_reward_std']:.2f}"
        q_err = f"{r['q_error']:.3f}" if r['q_error'] else "---"
        agr = f"{r['agreement']:.1%}" if r['agreement'] else "---"
        print(f"  {N:>3} | {states:>10,} | {dp_r:>10} | {dqn_r:>18} | {q_err:>8} | {agr:>6}")

    print("  " + "-" * 70)

    # Per-seed results
    print("\n  Per-Seed DQN Rewards:")
    for r in results:
        N = r['complexity']
        rewards = r['dqn_rewards']
        print(f"    N={N}: " + ", ".join(f"{rw:.2f}" for rw in rewards))

    # Heuristic comparison
    print("\n  Heuristic Comparison:")
    print("  " + "-" * 50)
    print(f"  {'N':>3} | {'Fixed':>10} | {'Load-prop':>10} | {'PID':>10}")
    print("  " + "-" * 50)
    for r in results:
        N = r['complexity']
        h = r['heuristics']
        print(f"  {N:>3} | {h['fixed']:>10.2f} | {h['load_proportional']:>10.2f} | "
              f"{h['pid']:>10.2f}")
    print("  " + "-" * 50)

    # Decomposition for largest N
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        print(f"\n  Cost Decomposition (N={largest['complexity']}, DQN policy):")
        decomp = largest['decomposition']
        print(f"    Total cost: {decomp.get('total_cost', 0):.2f}")
        print(f"    Energy cost: {decomp.get('energy_cost', 0):.2f}")
        print(f"    Overheat cost: {decomp.get('overheat_cost', 0):.2f}")
        print(f"    Comfort cost: {decomp.get('comfort_cost', 0):.2f}")
        print(f"    Overheat frequency: {decomp.get('overheated', 0):.2%}")

        N = largest['complexity']
        for k in range(N):
            key = f'zone_{k}_temp'
            if key in decomp:
                print(f"    Zone {k} avg temp: {decomp[key]:.2f}")

    # Output files
    print("\n  Output Files:")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    stdout_path = OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'
    with capture_stdout(stdout_path):
        print("Data Center Cooling Benchmark")
        print(f"Complexity sweep: N in {COMPLEXITY_SWEEP}")
        print(f"Seeds: {SEEDS}")

        results = run_scaling_sweep()
        make_figures(results)
        make_latex_tables(results)
        print_detailed_results(results)

        print("\nBenchmark complete.")
