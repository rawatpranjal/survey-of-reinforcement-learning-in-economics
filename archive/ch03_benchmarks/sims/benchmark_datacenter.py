"""
Data Center Cooling Benchmark (Chapter 3)

Multi-zone data center cooling control. The agent sets cooling intensity per zone
to minimize energy cost while avoiding overheating. State space grows exponentially
in the number of zones. DP feasible for N_zones <= 2. Compare DP, DQN, and
classical heuristics (fixed setpoint, load-proportional, PID controller).
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
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic
)

# =============================================================================
# Configuration
# =============================================================================
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

SEED = 42
DQN_SEEDS = [42, 123, 7]
OUTPUT_DIR = Path(__file__).resolve().parent

# Environment parameters
TEMP_BINS = 6          # Temperature bins per zone: 0=cold(18C), ..., 5=hot(28C)
LOAD_BINS = 4          # Server load bins per zone: 0=idle, ..., 3=full
EXT_TEMP_BINS = 3      # External temperature: 0=cool, 1=mild, 2=hot
COOLING_LEVELS = 4     # Action per zone: cooling intensity {0.25, 0.50, 0.75, 1.00}
COOLING_VALUES = [0.25, 0.50, 0.75, 1.00]

ZONE_VALUES = [1, 2, 3, 4]  # Complexity dial: number of zones
DP_CUTOFF = 2               # DP feasible for N_zones <= 2

# Thermal dynamics parameters
ALPHA = 0.4           # Heat gain from server load (per unit load)
BETA = 0.5            # Cooling effectiveness
NOISE_STD = 0.3       # Temperature noise std
EXT_INFLUENCE = 0.15  # External temperature influence

# Cost parameters
ENERGY_COST_BASE = 1.0       # Base energy cost per unit cooling
ENERGY_COST_SCALE = 2.0      # Quadratic energy cost scaling
OVERHEAT_PENALTY = 20.0      # Penalty per zone if temp >= threshold
OVERHEAT_THRESHOLD = 4       # Temperature bin threshold (index 4 out of 0..5)
TARGET_TEMP = 2              # Target temperature bin (comfortable)
COMFORT_PENALTY = 0.5        # Penalty for deviation from target

GAMMA = 0.95
EPISODE_HORIZON = 20
EVAL_EPISODES = 50

# Load transition probabilities (Markov)
LOAD_TRANSITION = np.array([
    [0.5, 0.3, 0.15, 0.05],
    [0.2, 0.4, 0.3, 0.1],
    [0.1, 0.2, 0.4, 0.3],
    [0.05, 0.15, 0.3, 0.5],
])

# External temp transition
EXT_TRANSITION = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
])

def get_dqn_episodes(N_zones):
    if N_zones <= 1:
        return 500
    elif N_zones == 2:
        return 1000
    elif N_zones == 3:
        return 2000
    else:
        return 3000


# =============================================================================
# Environment
# =============================================================================
class DataCenterCooling(EconBenchmark):
    """
    Data center cooling MDP.

    State: (temp_1, ..., temp_N, load_1, ..., load_N, ext_temp)
           where temp_k in {0,...,TEMP_BINS-1}, load_k in {0,...,LOAD_BINS-1},
           ext_temp in {0,...,EXT_TEMP_BINS-1}

    Action: single integer encoding cooling levels for all zones.
            For N zones with C cooling levels each, action in {0, ..., C^N - 1}.

    Reward: -(energy_cost + overheat_penalty + comfort_penalty)
    """

    dp_feasible = True

    def __init__(self, N_zones, gamma=GAMMA, horizon=EPISODE_HORIZON):
        self.N_zones = N_zones
        self.complexity_param = N_zones
        self.gamma = gamma
        self.horizon = horizon

        # State space: (TEMP_BINS^N * LOAD_BINS^N * EXT_TEMP_BINS)
        self._num_states = (TEMP_BINS ** N_zones) * (LOAD_BINS ** N_zones) * EXT_TEMP_BINS
        # Action space: COOLING_LEVELS^N (joint cooling for all zones)
        self._num_actions = COOLING_LEVELS ** N_zones

        self.state = None
        self.t = 0
        self.reset()

    def reset(self):
        # Start at comfortable temperature, medium load, mild external
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
        """Decode flat action index to per-zone cooling levels."""
        cooling = []
        idx = action_idx
        for _ in range(self.N_zones):
            cooling.append(idx % COOLING_LEVELS)
            idx //= COOLING_LEVELS
        return cooling  # list of cooling level indices

    def _encode_action(self, cooling_list):
        """Encode per-zone cooling level indices to flat action."""
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
        """Compute reward given current temperatures and cooling actions."""
        total_cost = 0.0
        for k in range(self.N_zones):
            c = COOLING_VALUES[cooling_indices[k]]
            # Energy cost: quadratic in cooling intensity
            energy = ENERGY_COST_BASE * c + ENERGY_COST_SCALE * c * c
            total_cost += energy

            # Overheat penalty
            if temps[k] >= OVERHEAT_THRESHOLD:
                total_cost += OVERHEAT_PENALTY * (temps[k] - OVERHEAT_THRESHOLD + 1)

            # Comfort penalty: deviation from target
            total_cost += COMFORT_PENALTY * abs(temps[k] - TARGET_TEMP)

        return -total_cost

    def _next_temp(self, temp, load, cooling_idx, ext, noise=None):
        """Compute next temperature bin for one zone."""
        c = COOLING_VALUES[cooling_idx]
        # Continuous temperature dynamics
        temp_continuous = temp + ALPHA * load - BETA * c + EXT_INFLUENCE * ext
        if noise is not None:
            temp_continuous += noise
        else:
            temp_continuous += np.random.normal(0, NOISE_STD)
        # Discretize back to bin
        new_temp = int(round(np.clip(temp_continuous, 0, TEMP_BINS - 1)))
        return new_temp

    def step(self, action):
        N = self.N_zones
        temps = list(self.state[:N])
        loads = list(self.state[N:2*N])
        ext = self.state[2*N]

        cooling = self._decode_action(action)

        # Compute reward based on current state and action
        reward = self._compute_reward(temps, cooling, ext)

        # Transition temperatures
        new_temps = []
        for k in range(N):
            new_t = self._next_temp(temps[k], loads[k], cooling[k], ext)
            new_temps.append(new_t)

        # Transition loads (independent Markov per zone)
        new_loads = []
        for k in range(N):
            new_l = np.random.choice(LOAD_BINS, p=LOAD_TRANSITION[loads[k]])
            new_loads.append(new_l)

        # Transition external temperature
        new_ext = np.random.choice(EXT_TEMP_BINS, p=EXT_TRANSITION[ext])

        self.state = tuple(new_temps) + tuple(new_loads) + (new_ext,)
        self.t += 1
        done = (self.t >= self.horizon)

        return self.state, reward, done

    def transition_distribution(self, state, action):
        """Enumerate all possible next states and their probabilities."""
        N = self.N_zones
        temps = list(state[:N])
        loads = list(state[N:2*N])
        ext = state[2*N]

        cooling = self._decode_action(action)

        # For DP, discretize noise: use {-1, 0, +1} with probabilities
        noise_vals = [-1, 0, 1]
        noise_probs = [0.2, 0.6, 0.2]

        # Build all combinations of (temp_noise per zone) x (load per zone) x ext
        # This is expensive but necessary for exact DP
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


# =============================================================================
# Heuristic Policies
# =============================================================================
def make_heuristic_fixed(N_zones):
    """Fixed setpoint: always use medium cooling (0.50) for all zones."""
    fixed_cooling = [1] * N_zones  # Index 1 = 0.50

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
            # Map load bin to cooling level index
            # load 0 -> cooling 0 (0.25), load 3 -> cooling 3 (1.00)
            c_idx = min(loads[k], COOLING_LEVELS - 1)
            cooling.append(c_idx)
        env_dummy = DataCenterCooling.__new__(DataCenterCooling)
        env_dummy.N_zones = N_zones
        return env_dummy._encode_action(cooling)

    return heuristic


def make_heuristic_pid(N_zones):
    """
    PID controller: adjust cooling based on temperature error from target.
    Uses proportional + integral (accumulated error) terms.
    """
    integral = [0.0] * N_zones
    Kp = 0.5
    Ki = 0.1

    def heuristic(state):
        temps = list(state[:N_zones])
        cooling = []
        for k in range(N_zones):
            error = temps[k] - TARGET_TEMP
            integral[k] = integral[k] * 0.9 + error  # Leaky integrator
            control = Kp * error + Ki * integral[k]
            # Map control signal to cooling level index
            # control ~ 0 -> low cooling, control large -> high cooling
            c_continuous = 0.25 + 0.25 * max(0, control)
            c_continuous = np.clip(c_continuous, 0.25, 1.0)
            # Find closest cooling level
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


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    results = {
        'N_zones': [],
        'states': [],
        'dp_time': [],
        'dp_reward': [],
        'dqn_time': [],
        'dqn_reward': [],
        'dqn_std': [],
        'fixed_reward': [],
        'load_prop_reward': [],
        'pid_reward': [],
    }

    for N_zones in ZONE_VALUES:
        print(f"\n{'='*60}")
        print(f"N_zones = {N_zones}")
        print(f"{'='*60}")

        env = DataCenterCooling(N_zones=N_zones)
        n_states = env.num_states
        n_actions = env.num_actions

        print(f"State space: {n_states}, Action space: {n_actions}")

        results['N_zones'].append(N_zones)
        results['states'].append(n_states)

        # Dynamic Programming
        if N_zones <= DP_CUTOFF:
            print("\nRunning Value Iteration (DP)...")
            V, policy, dp_time = run_value_iteration(env, gamma=GAMMA)
            results['dp_time'].append(dp_time)

            print("Evaluating DP policy...")
            dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                           n_episodes=EVAL_EPISODES,
                                           horizon=EPISODE_HORIZON)
            results['dp_reward'].append(dp_reward)

            print(f"  DP time: {dp_time:.2f}s")
            print(f"  DP reward: {dp_reward:.2f}")
        else:
            print("\nSkipping DP (state space too large)")
            results['dp_time'].append(np.nan)
            results['dp_reward'].append(np.nan)

        # DQN
        print("\nRunning DQN...")
        dqn_episodes = get_dqn_episodes(N_zones)
        dqn_times = []
        dqn_rewards = []

        for seed in DQN_SEEDS:
            env_dqn = DataCenterCooling(N_zones=N_zones)
            q_net, dqn_time = run_dqn(env_dqn, gamma=GAMMA,
                                       num_episodes=dqn_episodes,
                                       episode_horizon=EPISODE_HORIZON, seed=seed)
            dqn_reward = evaluate_dqn_policy(env, q_net,
                                              n_episodes=EVAL_EPISODES,
                                              horizon=EPISODE_HORIZON)
            dqn_times.append(dqn_time)
            dqn_rewards.append(dqn_reward)
            print(f"  seed={seed}: time={dqn_time:.2f}s, reward={dqn_reward:.2f}")

        results['dqn_time'].append(np.mean(dqn_times))
        results['dqn_reward'].append(np.mean(dqn_rewards))
        results['dqn_std'].append(np.std(dqn_rewards))

        # Heuristics
        print("\nEvaluating heuristics...")

        h_fixed = make_heuristic_fixed(N_zones)
        fixed_reward = evaluate_heuristic(env, h_fixed,
                                           n_episodes=EVAL_EPISODES,
                                           horizon=EPISODE_HORIZON)
        results['fixed_reward'].append(fixed_reward)
        print(f"  Fixed setpoint: {fixed_reward:.2f}")

        h_load = make_heuristic_load_proportional(N_zones)
        load_prop_reward = evaluate_heuristic(env, h_load,
                                               n_episodes=EVAL_EPISODES,
                                               horizon=EPISODE_HORIZON)
        results['load_prop_reward'].append(load_prop_reward)
        print(f"  Load-proportional: {load_prop_reward:.2f}")

        h_pid = make_heuristic_pid(N_zones)
        pid_reward = evaluate_heuristic(env, h_pid,
                                         n_episodes=EVAL_EPISODES,
                                         horizon=EPISODE_HORIZON)
        results['pid_reward'].append(pid_reward)
        print(f"  PID: {pid_reward:.2f}")

    # =========================================================================
    # Generate Figure
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    nz_array = np.array(results['N_zones'])

    # Left panel: Computation time (log scale)
    dp_times_plot = [t if not np.isnan(t) else None for t in results['dp_time']]

    dp_nz = [nz_array[i] for i in range(len(nz_array)) if dp_times_plot[i] is not None]
    dp_t = [dp_times_plot[i] for i in range(len(nz_array)) if dp_times_plot[i] is not None]
    if dp_nz:
        ax1.semilogy(dp_nz, dp_t, 'o-', label='DP', linewidth=2, markersize=8)

    ax1.semilogy(nz_array, results['dqn_time'], 's-', label='DQN', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Zones', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Scaling: Computation Time', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(nz_array)

    # Right panel: Reward comparison
    dp_nz_r = [nz_array[i] for i in range(len(nz_array)) if not np.isnan(results['dp_reward'][i])]
    dp_r = [results['dp_reward'][i] for i in range(len(nz_array)) if not np.isnan(results['dp_reward'][i])]
    if dp_nz_r:
        ax2.plot(dp_nz_r, dp_r, 'o-', label='DP', linewidth=2, markersize=8)

    ax2.errorbar(nz_array, results['dqn_reward'], yerr=results['dqn_std'],
                 fmt='s-', label='DQN', linewidth=2, markersize=8, capsize=5)
    ax2.plot(nz_array, results['fixed_reward'], '^--', label='Fixed', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(nz_array, results['load_prop_reward'], 'v--', label='Load-prop.', linewidth=2, markersize=8, alpha=0.7)
    ax2.plot(nz_array, results['pid_reward'], 'd--', label='PID', linewidth=2, markersize=8, alpha=0.7)
    ax2.set_xlabel('Number of Zones', fontsize=12)
    ax2.set_ylabel('Average Reward per Episode', fontsize=12)
    ax2.set_title('Performance Comparison', fontsize=13)
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(nz_array)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / "datacenter_scaling.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================
    table_lines = []
    table_lines.append(r"\begin{tabular}{lrrrrrrr}")
    table_lines.append(r"\hline")
    table_lines.append(r"Zones & States & DP Time & DP Reward & DQN Time & DQN Reward & Fixed & PID \\")
    table_lines.append(r"\hline")

    for i, nz in enumerate(results['N_zones']):
        states = results['states'][i]

        if np.isnan(results['dp_time'][i]):
            dp_time_str = "---"
            dp_reward_str = "---"
        else:
            dp_time_str = f"{results['dp_time'][i]:.2f}"
            dp_reward_str = f"{results['dp_reward'][i]:.2f}"

        dqn_time_str = f"{results['dqn_time'][i]:.2f}"
        dqn_reward_str = f"{results['dqn_reward'][i]:.2f}"
        fixed_str = f"{results['fixed_reward'][i]:.2f}"
        pid_str = f"{results['pid_reward'][i]:.2f}"

        table_lines.append(
            f"{nz} & {states} & {dp_time_str} & {dp_reward_str} & "
            f"{dqn_time_str} & {dqn_reward_str} & {fixed_str} & {pid_str} \\\\"
        )

    table_lines.append(r"\hline")
    table_lines.append(r"\end{tabular}")

    table_content = "\n".join(table_lines)
    table_path = OUTPUT_DIR / "datacenter_results.tex"
    with open(table_path, 'w') as f:
        f.write(table_content)

    print(f"Table saved: {table_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
