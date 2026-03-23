"""
DPO Diagnostic Study: Systematic Investigation of DPO Failure Modes
Chapter 8 - RLHF & Preference Learning

Tests six hypotheses for why DPO achieves negative returns (-7.7 at K=5000)
while the DP oracle returns 6.52 on the same 10x10 stochastic gridworld.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, CMAP_SEQ, FIG_DOUBLE
apply_style()

# Import from main RLHF script
sys.path.insert(0, os.path.dirname(__file__))
import gridworld_rlhf as rlhf

# Diagnostic hyperparameters
N_SEEDS = 5
MASTER_SEED = 42
EVAL_EPISODES = 200
OUTPUT_DIR = 'ch08_rlhf/sims'


# =============================================================================
# Helper: generate comparisons with variable segment length
# =============================================================================

def generate_segment_L(env, policy, length):
    """Generate a single segment of specified length."""
    while True:
        start = (np.random.randint(env.N), np.random.randint(env.N))
        if start != env.goal:
            break
    transitions = []
    total_reward = 0.0
    s = env.reset(start)
    for _ in range(length):
        if s == env.goal:
            break
        si = env.state_to_index(s)
        if isinstance(policy, str) and policy == 'random':
            a = np.random.randint(env.num_actions)
        elif callable(policy):
            a = policy(si)
        else:
            a = int(policy[si])
        ns, r, done = env.step(a)
        transitions.append((s, a, ns))
        total_reward += r
        s = ns
        if done:
            break
    return transitions, total_reward


def generate_comparisons_L(env, K, length, policy='random'):
    """Generate K pairwise comparisons with specified segment length."""
    comparisons = []
    for _ in range(K):
        t1, r1 = generate_segment_L(env, policy, length)
        t2, r2 = generate_segment_L(env, policy, length)
        if not t1 or not t2:
            continue
        diff = np.clip(r1 - r2, -500, 500)
        if np.random.random() < 1.0 / (1.0 + np.exp(-diff)):
            comparisons.append((t1, t2))
        else:
            comparisons.append((t2, t1))
    return comparisons


# =============================================================================
# Helper: DPO with variable segment length arrays
# =============================================================================

def precompute_dpo_arrays_varlen(comparisons, env, max_len):
    """Precompute DPO arrays with variable max segment length."""
    K = len(comparisons)
    L = max_len
    nA = env.num_actions
    pad_si = env.goal_si

    w_si = np.full((K, L), pad_si, dtype=np.intp)
    w_a = np.zeros((K, L), dtype=np.intp)
    w_mask = np.zeros((K, L), dtype=np.float64)

    l_si = np.full((K, L), pad_si, dtype=np.intp)
    l_a = np.zeros((K, L), dtype=np.intp)
    l_mask = np.zeros((K, L), dtype=np.float64)

    for i, (winner, loser) in enumerate(comparisons):
        for t, (s, a, ns) in enumerate(winner):
            if t >= L:
                break
            w_si[i, t] = env.state_to_index(s)
            w_a[i, t] = a
            w_mask[i, t] = 1.0
        for t, (s, a, ns) in enumerate(loser):
            if t >= L:
                break
            l_si[i, t] = env.state_to_index(s)
            l_a[i, t] = a
            l_mask[i, t] = 1.0

    return w_si, w_a, w_mask, l_si, l_a, l_mask


def train_dpo_custom(comparisons, env, lambda_kl, max_len, epochs, lr):
    """DPO training with custom segment length, epochs, and learning rate."""
    nS, nA = env.num_states, env.num_actions
    theta = np.zeros((nS, nA))
    log_nA = np.log(nA)

    dpo_arrays = precompute_dpo_arrays_varlen(comparisons, env, max_len)
    w_si, w_a, w_mask, l_si, l_a, l_mask = dpo_arrays

    K = w_si.shape[0]
    L = w_si.shape[1]
    w_len = w_mask.sum(axis=1)
    l_len = l_mask.sum(axis=1)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    final_loss = 0.0
    for epoch in range(epochs):
        log_pi_w = rlhf._batch_log_prob(theta, w_si, w_a, w_mask)
        log_pi_l = rlhf._batch_log_prob(theta, l_si, l_a, l_mask)
        log_ratio_w = log_pi_w + w_len * log_nA
        log_ratio_l = log_pi_l + l_len * log_nA

        h = np.clip(lambda_kl * (log_ratio_w - log_ratio_l), -500, 500)
        sigma_h = 1.0 / (1.0 + np.exp(-h))
        final_loss = -np.mean(np.log(sigma_h + 1e-15))

        coeff = -(1.0 - sigma_h) * lambda_kl / K

        grad = np.zeros_like(theta)

        w_logits = theta[w_si]
        w_probs = np.exp(w_logits - w_logits.max(axis=2, keepdims=True))
        w_probs /= w_probs.sum(axis=2, keepdims=True)

        l_logits = theta[l_si]
        l_probs = np.exp(l_logits - l_logits.max(axis=2, keepdims=True))
        l_probs /= l_probs.sum(axis=2, keepdims=True)

        w_indicator = np.zeros((K, L, nA))
        w_indicator[np.arange(K)[:, None], np.arange(L)[None, :], w_a] = 1.0
        w_delta = (w_indicator - w_probs) * w_mask[:, :, None] * coeff[:, None, None]

        l_indicator = np.zeros((K, L, nA))
        l_indicator[np.arange(K)[:, None], np.arange(L)[None, :], l_a] = 1.0
        l_delta = -(l_indicator - l_probs) * l_mask[:, :, None] * coeff[:, None, None]

        np.add.at(grad, w_si.ravel(), w_delta.reshape(-1, nA))
        np.add.at(grad, l_si.ravel(), l_delta.reshape(-1, nA))

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return theta, final_loss


def train_dpo_with_reference(comparisons, env, lambda_kl, ref_log_probs, epochs=300, lr=5e-3):
    """DPO training with a non-uniform reference policy.

    ref_log_probs: (nS, nA) array of log pi_ref(a|s)
    """
    nS, nA = env.num_states, env.num_actions
    theta = np.zeros((nS, nA))

    dpo_arrays = precompute_dpo_arrays_varlen(comparisons, env, rlhf.SEGMENT_LENGTH)
    w_si, w_a, w_mask, l_si, l_a, l_mask = dpo_arrays

    K = w_si.shape[0]
    L = w_si.shape[1]

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    final_loss = 0.0
    for epoch in range(epochs):
        # Log pi(a|s) for policy
        logits_all = theta  # (nS, nA)
        log_z_all = np.logaddexp.reduce(logits_all, axis=1, keepdims=True)  # (nS, 1)
        log_pi_all = logits_all - log_z_all  # (nS, nA)

        # Log-ratio: log pi(a|s) - log pi_ref(a|s) for each transition
        log_ratio_all = log_pi_all - ref_log_probs  # (nS, nA)

        # Sum log-ratios over segment transitions
        w_lr = log_ratio_all[w_si, w_a]  # (K, L)
        l_lr = log_ratio_all[l_si, l_a]  # (K, L)
        log_ratio_w = (w_lr * w_mask).sum(axis=1)  # (K,)
        log_ratio_l = (l_lr * l_mask).sum(axis=1)  # (K,)

        h = np.clip(lambda_kl * (log_ratio_w - log_ratio_l), -500, 500)
        sigma_h = 1.0 / (1.0 + np.exp(-h))
        final_loss = -np.mean(np.log(sigma_h + 1e-15))

        coeff = -(1.0 - sigma_h) * lambda_kl / K

        grad = np.zeros_like(theta)
        pi_all = np.exp(log_pi_all)  # (nS, nA)

        w_probs = pi_all[w_si]  # (K, L, nA)
        l_probs = pi_all[l_si]

        w_indicator = np.zeros((K, L, nA))
        w_indicator[np.arange(K)[:, None], np.arange(L)[None, :], w_a] = 1.0
        w_delta = (w_indicator - w_probs) * w_mask[:, :, None] * coeff[:, None, None]

        l_indicator = np.zeros((K, L, nA))
        l_indicator[np.arange(K)[:, None], np.arange(L)[None, :], l_a] = 1.0
        l_delta = -(l_indicator - l_probs) * l_mask[:, :, None] * coeff[:, None, None]

        np.add.at(grad, w_si.ravel(), w_delta.reshape(-1, nA))
        np.add.at(grad, l_si.ravel(), l_delta.reshape(-1, nA))

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return theta, final_loss


# =============================================================================
# Helper: greedy evaluation from DPO theta
# =============================================================================

def evaluate_greedy_from_theta(env, theta, n_episodes=EVAL_EPISODES):
    """Extract greedy (argmax) policy from softmax theta, evaluate deterministically."""
    policy = np.argmax(theta, axis=1)
    return rlhf.evaluate_deterministic(env, policy, n_episodes)


# =============================================================================
# Helper: 5x5 gridworld
# =============================================================================

class SmallGridworldEnv:
    """5x5 gridworld with analogous structure to the 10x10."""
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self):
        self.N = 5
        self.goal = (4, 4)
        self.hazards = {(2, 2)}
        self.traps = {(3, 1)}
        self.goal_si = self.state_to_index(self.goal)
        self.state = None
        self._trans_cache = {}
        self._precompute_transitions()
        self.reset()

    def _precompute_transitions(self):
        for si in range(self.num_states):
            s = self.index_to_state(si)
            for a in range(self.num_actions):
                self._trans_cache[(si, a)] = self._compute_transitions(s, a)

    def _compute_transitions(self, s, a):
        if s == self.goal:
            return [(s, 1.0)]
        transitions = {}
        ns_intended = self._next_state_det(s, a)
        transitions[ns_intended] = transitions.get(ns_intended, 0.0) + (1.0 - rlhf.SLIP_PROB)
        for a2 in range(self.num_actions):
            ns_slip = self._next_state_det(s, a2)
            transitions[ns_slip] = transitions.get(ns_slip, 0.0) + rlhf.SLIP_PROB / self.num_actions
        return list(transitions.items())

    @property
    def num_states(self):
        return self.N * self.N

    @property
    def num_actions(self):
        return len(self.ACTIONS)

    def reset(self, start=None):
        self.state = (0, 0) if start is None else start
        return self.state

    def state_to_index(self, s):
        return s[0] * self.N + s[1]

    def index_to_state(self, idx):
        return (idx // self.N, idx % self.N)

    def _next_state_det(self, state, action):
        dr, dc = self.ACTIONS[action]
        return (max(0, min(self.N - 1, state[0] + dr)),
                max(0, min(self.N - 1, state[1] + dc)))

    def transition_features(self, s, a, ns):
        return np.array([1.0, float(ns == self.goal),
                         float(ns in self.hazards), float(ns in self.traps)])

    def true_reward(self, s, a, ns):
        if s == self.goal:
            return 0.0
        return float(rlhf.PHI_TRUE @ self.transition_features(s, a, ns))

    def step(self, action):
        if self.state == self.goal:
            return self.state, 0.0, True
        if np.random.random() < rlhf.SLIP_PROB:
            actual = np.random.randint(self.num_actions)
        else:
            actual = action
        ns = self._next_state_det(self.state, actual)
        r = self.true_reward(self.state, action, ns)
        self.state = ns
        return ns, r, (ns == self.goal)


def value_iteration_small(env, tol=1e-10, max_iter=5000):
    """Value iteration for the small gridworld."""
    nS, nA = env.num_states, env.num_actions
    V = np.zeros(nS)
    for it in range(max_iter):
        V_new = np.zeros(nS)
        for si in range(nS):
            if si == env.goal_si:
                continue
            s = env.index_to_state(si)
            best = -1e30
            for a in range(nA):
                q = sum(p * (env.true_reward(s, a, ns) + rlhf.GAMMA * V[env.state_to_index(ns)])
                        for ns, p in env._trans_cache[(si, a)])
                if q > best:
                    best = q
            V_new[si] = best
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = np.zeros(nS, dtype=int)
    for si in range(nS):
        if si == env.goal_si:
            continue
        s = env.index_to_state(si)
        best_q, best_a = -1e30, 0
        for a in range(nA):
            q = sum(p * (env.true_reward(s, a, ns) + rlhf.GAMMA * V[env.state_to_index(ns)])
                    for ns, p in env._trans_cache[(si, a)])
            if q > best_q:
                best_q, best_a = q, a
        policy[si] = best_a
    return V, policy


# =============================================================================
# Experiment 1: Greedy vs Stochastic Evaluation (H1)
# =============================================================================

def experiment_1(env, dp_ret):
    print('\n' + '=' * 80)
    print('EXPERIMENT 1: GREEDY VS STOCHASTIC EVALUATION (H1)')
    print('=' * 80)
    print('Hypothesis: Stochastic evaluation inflates DPO failure.')
    print(f'Config: K=2000, segment_length=25, 3 lambda candidates, {N_SEEDS} seeds\n')

    stoch_rets = []
    greedy_rets = []
    for seed in range(N_SEEDS):
        rng_seed = MASTER_SEED + seed * 10000 + 2000
        np.random.seed(rng_seed)
        comps = rlhf.generate_comparisons(env, 2000)

        best_theta, best_loss, best_lam = rlhf.train_dpo_best_lambda(comps, env)

        np.random.seed(rng_seed + 80000)
        s_ret, _ = rlhf.evaluate_stochastic(env, best_theta)
        stoch_rets.append(s_ret)

        np.random.seed(rng_seed + 90000)
        g_ret, _ = evaluate_greedy_from_theta(env, best_theta)
        greedy_rets.append(g_ret)

        print(f'  Seed {seed:2d}: stochastic={s_ret:7.3f}, greedy={g_ret:7.3f}, '
              f'delta={g_ret - s_ret:+7.3f}, lambda={best_lam}')

    sm, ss = np.mean(stoch_rets), np.std(stoch_rets) / np.sqrt(N_SEEDS)
    gm, gs = np.mean(greedy_rets), np.std(greedy_rets) / np.sqrt(N_SEEDS)
    print(f'\n  Stochastic: {sm:7.3f} +/- {ss:.3f} ({100*sm/dp_ret:5.1f}% of DP)')
    print(f'  Greedy:     {gm:7.3f} +/- {gs:.3f} ({100*gm/dp_ret:5.1f}% of DP)')
    print(f'  Difference: {gm - sm:+7.3f} (greedy - stochastic)')

    return stoch_rets, greedy_rets


# =============================================================================
# Experiment 2: Segment Length Ablation (H2)
# =============================================================================

def experiment_2(env, dp_ret):
    print('\n' + '=' * 80)
    print('EXPERIMENT 2: SEGMENT LENGTH ABLATION (H2)')
    print('=' * 80)
    print('Hypothesis: Long segments dilute credit assignment signal.')
    print(f'Config: K=1000, lambda candidates=[0.001, 0.01, 0.1], {N_SEEDS} seeds\n')

    lengths = [1, 3, 5, 10, 25]
    lambda_candidates = [0.001, 0.01, 0.1]
    results = {}

    for L in lengths:
        rets = []
        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000
            np.random.seed(rng_seed)
            comps = generate_comparisons_L(env, 1000, length=L)

            best_theta, best_loss, best_lam = None, float('inf'), None
            for lam in lambda_candidates:
                theta, loss = train_dpo_custom(comps, env, lam, max_len=L,
                                               epochs=300, lr=5e-3)
                if loss < best_loss:
                    best_loss = loss
                    best_theta = theta.copy()
                    best_lam = lam

            np.random.seed(rng_seed + 80000)
            g_ret, _ = evaluate_greedy_from_theta(env, best_theta)
            rets.append(g_ret)

        results[L] = rets
        m, s = np.mean(rets), np.std(rets) / np.sqrt(N_SEEDS)
        print(f'  L={L:2d}: return={m:7.3f} +/- {s:.3f} ({100*m/dp_ret:5.1f}% of DP)')

    return results


# =============================================================================
# Experiment 3: Lambda KL Search (H3)
# =============================================================================

def experiment_3(env, dp_ret):
    print('\n' + '=' * 80)
    print('EXPERIMENT 3: LAMBDA KL SEARCH (H3)')
    print('=' * 80)
    print('Hypothesis: Current lambda range is too narrow.')
    print(f'Config: K=1000, segment_length=25, {N_SEEDS} seeds\n')

    lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    results = {}

    for lam in lambdas:
        rets = []
        losses = []
        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000
            np.random.seed(rng_seed)
            comps = rlhf.generate_comparisons(env, 1000)

            theta, loss = train_dpo_custom(comps, env, lam, max_len=25,
                                           epochs=300, lr=5e-3)
            losses.append(loss)

            np.random.seed(rng_seed + 80000)
            g_ret, _ = evaluate_greedy_from_theta(env, theta)
            rets.append(g_ret)

        results[lam] = rets
        m, s = np.mean(rets), np.std(rets) / np.sqrt(N_SEEDS)
        ml = np.mean(losses)
        print(f'  lambda={lam:6.3f}: return={m:7.3f} +/- {s:.3f} '
              f'({100*m/dp_ret:5.1f}% of DP), loss={ml:.4f}')

    return results


# =============================================================================
# Experiment 4: Training Budget (H4)
# =============================================================================

def experiment_4(env, dp_ret):
    print('\n' + '=' * 80)
    print('EXPERIMENT 4: TRAINING BUDGET (H4)')
    print('=' * 80)
    print('Hypothesis: 300 epochs with lr=5e-3 is insufficient.')
    print(f'Config: K=1000, segment_length=25, lambda=0.1, {N_SEEDS} seeds\n')

    epoch_vals = [300, 1000]
    lr_vals = [1e-3, 5e-3, 1e-2]
    results = {}

    print(f'  {"epochs":>6s} | {"lr":>6s} | {"return":>18s} | {"% DP":>6s} | {"loss":>8s}')
    print('  ' + '-' * 55)

    for epochs in epoch_vals:
        for lr in lr_vals:
            rets = []
            losses = []
            for seed in range(N_SEEDS):
                rng_seed = MASTER_SEED + seed * 10000
                np.random.seed(rng_seed)
                comps = rlhf.generate_comparisons(env, 1000)

                theta, loss = train_dpo_custom(comps, env, 0.1, max_len=25,
                                               epochs=epochs, lr=lr)
                losses.append(loss)

                np.random.seed(rng_seed + 80000)
                g_ret, _ = evaluate_greedy_from_theta(env, theta)
                rets.append(g_ret)

            results[(epochs, lr)] = rets
            m, s = np.mean(rets), np.std(rets) / np.sqrt(N_SEEDS)
            ml = np.mean(losses)
            print(f'  {epochs:6d} | {lr:6.4f} | {m:7.3f} +/- {s:.3f} | '
                  f'{100*m/dp_ret:5.1f}% | {ml:8.4f}')

    return results


# =============================================================================
# Experiment 5: Grid Size (H5)
# =============================================================================

def experiment_5():
    print('\n' + '=' * 80)
    print('EXPERIMENT 5: 5x5 GRID (H5)')
    print('=' * 80)
    print('Hypothesis: 400 parameters with trajectory comparisons is too many.')
    print(f'Config: 5x5 grid (100 params), segment_length=25, {N_SEEDS} seeds\n')

    env5 = SmallGridworldEnv()
    V5, pi5 = value_iteration_small(env5)

    np.random.seed(MASTER_SEED)
    dp5_ret, dp5_se = rlhf.evaluate_deterministic(env5, pi5)
    print(f'  5x5 DP Oracle: V*(start)={V5[0]:.4f}, return={dp5_ret:.3f} +/- {dp5_se:.3f}')

    K_vals = [100, 500, 2000]
    lambda_candidates = [0.001, 0.01, 0.1, 1.0]
    results = {}

    for K in K_vals:
        rets = []
        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000 + K
            np.random.seed(rng_seed)
            comps = generate_comparisons_L(env5, K, length=25)

            best_theta, best_loss = None, float('inf')
            for lam in lambda_candidates:
                theta, loss = train_dpo_custom(comps, env5, lam, max_len=25,
                                               epochs=300, lr=5e-3)
                if loss < best_loss:
                    best_loss = loss
                    best_theta = theta.copy()

            np.random.seed(rng_seed + 80000)
            policy = np.argmax(best_theta, axis=1)
            g_ret, _ = rlhf.evaluate_deterministic(env5, policy)
            rets.append(g_ret)

        results[K] = rets
        m, s = np.mean(rets), np.std(rets) / np.sqrt(N_SEEDS)
        print(f'  K={K:4d}: return={m:7.3f} +/- {s:.3f} ({100*m/dp5_ret:5.1f}% of DP)')

    return results, dp5_ret


# =============================================================================
# Experiment 6: Informed Reference Policy (H6)
# =============================================================================

def experiment_6(env, dp_ret):
    print('\n' + '=' * 80)
    print('EXPERIMENT 6: INFORMED REFERENCE POLICY (H6)')
    print('=' * 80)
    print('Hypothesis: Uniform reference provides no useful regularization.')
    print(f'Config: K=1000, segment_length=25, {N_SEEDS} seeds\n')

    nS, nA = env.num_states, env.num_actions

    # Train a Q-learning policy as reference (imperfect but informed)
    np.random.seed(MASTER_SEED + 99999)
    Q_ref = np.zeros((nS, nA))
    for _ in range(1000):
        s = env.reset()
        for _ in range(rlhf.MAX_STEPS):
            if s == env.goal:
                break
            si = env.state_to_index(s)
            a = np.random.randint(nA) if np.random.random() < 0.3 else np.argmax(Q_ref[si])
            ns, r, done = env.step(a)
            Q_ref[si, a] += 0.1 * (r + rlhf.GAMMA * np.max(Q_ref[env.state_to_index(ns)]) - Q_ref[si, a])
            s = ns
            if done:
                break

    # Convert Q to softmax log-probabilities (temperature=1)
    ref_log_probs = Q_ref - np.logaddexp.reduce(Q_ref, axis=1, keepdims=True)
    uniform_log_probs = np.full((nS, nA), -np.log(nA))

    # Evaluate reference policy quality
    ref_policy = np.argmax(Q_ref, axis=1)
    np.random.seed(MASTER_SEED + 99998)
    ref_ret, _ = rlhf.evaluate_deterministic(env, ref_policy)
    print(f'  Reference policy quality: return={ref_ret:.3f} ({100*ref_ret/dp_ret:.1f}% of DP)')

    lambda_candidates = [0.001, 0.01, 0.1, 1.0]
    uniform_rets = []
    informed_rets = []

    for seed in range(N_SEEDS):
        rng_seed = MASTER_SEED + seed * 10000
        np.random.seed(rng_seed)
        comps = rlhf.generate_comparisons(env, 1000)

        # Uniform reference (standard DPO)
        best_theta_u, best_loss_u = None, float('inf')
        for lam in lambda_candidates:
            theta, loss = train_dpo_with_reference(comps, env, lam, uniform_log_probs)
            if loss < best_loss_u:
                best_loss_u = loss
                best_theta_u = theta.copy()

        np.random.seed(rng_seed + 80000)
        u_ret, _ = evaluate_greedy_from_theta(env, best_theta_u)
        uniform_rets.append(u_ret)

        # Informed reference
        best_theta_i, best_loss_i = None, float('inf')
        for lam in lambda_candidates:
            theta, loss = train_dpo_with_reference(comps, env, lam, ref_log_probs)
            if loss < best_loss_i:
                best_loss_i = loss
                best_theta_i = theta.copy()

        np.random.seed(rng_seed + 80000)
        i_ret, _ = evaluate_greedy_from_theta(env, best_theta_i)
        informed_rets.append(i_ret)

        print(f'  Seed {seed:2d}: uniform={u_ret:7.3f}, informed={i_ret:7.3f}, '
              f'delta={i_ret - u_ret:+7.3f}')

    um, us = np.mean(uniform_rets), np.std(uniform_rets) / np.sqrt(N_SEEDS)
    im, is_ = np.mean(informed_rets), np.std(informed_rets) / np.sqrt(N_SEEDS)
    print(f'\n  Uniform ref:  {um:7.3f} +/- {us:.3f} ({100*um/dp_ret:5.1f}% of DP)')
    print(f'  Informed ref: {im:7.3f} +/- {is_:.3f} ({100*im/dp_ret:5.1f}% of DP)')

    return uniform_rets, informed_rets


# =============================================================================
# Main
# =============================================================================

def run_diagnostics():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('=' * 80)
    print('DPO DIAGNOSTIC STUDY')
    print('=' * 80)
    print(f'\nTimestamp: {timestamp}')
    print(f'Python: {sys.version.split()[0]}, NumPy: {np.__version__}')
    print(f'\nBASELINE PARAMETERS:')
    print(f'  Grid: {rlhf.GRID_SIZE}x{rlhf.GRID_SIZE}, slip={rlhf.SLIP_PROB}, gamma={rlhf.GAMMA}')
    print(f'  Goal: {rlhf.GOAL}, Hazards: {rlhf.HAZARDS}, Traps: {rlhf.TRAPS}')
    print(f'  DPO defaults: lambda={rlhf.DPO_LAMBDA_CANDIDATES}, lr={rlhf.DPO_LR}, '
          f'epochs={rlhf.DPO_EPOCHS}')
    print(f'  Diagnostic seeds: {N_SEEDS}')

    env = rlhf.GridworldEnv()

    # DP Oracle baseline
    V_star, pi_star, _ = rlhf.value_iteration(env)
    np.random.seed(MASTER_SEED)
    dp_ret, dp_se = rlhf.evaluate_deterministic(env, pi_star)
    print(f'\n  DP Oracle: V*(start)={V_star[0]:.4f}, return={dp_ret:.3f} +/- {dp_se:.3f}')

    # Run all experiments
    e1_stoch, e1_greedy = experiment_1(env, dp_ret)
    e2_results = experiment_2(env, dp_ret)
    e3_results = experiment_3(env, dp_ret)
    e4_results = experiment_4(env, dp_ret)
    e5_results, e5_dp = experiment_5()
    e6_uniform, e6_informed = experiment_6(env, dp_ret)

    # --- Summary ---
    print('\n' + '=' * 80)
    print('SUMMARY OF FINDINGS')
    print('=' * 80)

    # H1
    s1, g1 = np.mean(e1_stoch), np.mean(e1_greedy)
    print(f'\n  H1 (Evaluation artifact):')
    print(f'    Stochastic: {s1:.3f}, Greedy: {g1:.3f}, Delta: {g1 - s1:+.3f}')
    if g1 - s1 > 2.0:
        print(f'    CONFIRMED: Greedy evaluation improves return by {g1 - s1:.1f}')
    elif g1 > s1:
        print(f'    PARTIAL: Greedy helps but does not explain full failure')
    else:
        print(f'    REJECTED: Evaluation method does not explain failure')

    # H2
    print(f'\n  H2 (Segment length):')
    for L in sorted(e2_results.keys()):
        m = np.mean(e2_results[L])
        print(f'    L={L:2d}: {m:7.3f} ({100*m/dp_ret:5.1f}% of DP)')
    best_L = max(e2_results.keys(), key=lambda L: np.mean(e2_results[L]))
    worst_L = min(e2_results.keys(), key=lambda L: np.mean(e2_results[L]))
    if np.mean(e2_results[best_L]) - np.mean(e2_results[worst_L]) > 3.0:
        print(f'    CONFIRMED: Segment length matters (best L={best_L})')
    else:
        print(f'    WEAK/REJECTED: Segment length has limited effect')

    # H3
    print(f'\n  H3 (Lambda range):')
    for lam in sorted(e3_results.keys()):
        m = np.mean(e3_results[lam])
        print(f'    lambda={lam:6.3f}: {m:7.3f} ({100*m/dp_ret:5.1f}% of DP)')
    best_lam = max(e3_results.keys(), key=lambda l: np.mean(e3_results[l]))
    if best_lam > 0.1:
        print(f'    CONFIRMED: Best lambda={best_lam} is outside original range [0.001, 0.1]')
    else:
        print(f'    REJECTED: Original range covers the best lambda')

    # H4
    print(f'\n  H4 (Training budget):')
    best_config = max(e4_results.keys(), key=lambda k: np.mean(e4_results[k]))
    worst_config = min(e4_results.keys(), key=lambda k: np.mean(e4_results[k]))
    print(f'    Best: epochs={best_config[0]}, lr={best_config[1]}: '
          f'{np.mean(e4_results[best_config]):.3f}')
    print(f'    Worst: epochs={worst_config[0]}, lr={worst_config[1]}: '
          f'{np.mean(e4_results[worst_config]):.3f}')
    default_key = (300, 5e-3)
    if default_key in e4_results and np.mean(e4_results[best_config]) - np.mean(e4_results[default_key]) > 2.0:
        print(f'    CONFIRMED: More training helps')
    else:
        print(f'    REJECTED: Training budget is not the bottleneck')

    # H5
    print(f'\n  H5 (Grid size):')
    for K in sorted(e5_results.keys()):
        m = np.mean(e5_results[K])
        print(f'    5x5, K={K:4d}: {m:7.3f} ({100*m/e5_dp:5.1f}% of DP)')
    best_5x5 = np.mean(e5_results[max(e5_results.keys())])
    if best_5x5 / e5_dp > 0.5:
        print(f'    CONFIRMED: DPO works better on smaller grid')
    else:
        print(f'    REJECTED: DPO still fails on 5x5 grid')

    # H6
    u6, i6 = np.mean(e6_uniform), np.mean(e6_informed)
    print(f'\n  H6 (Reference policy):')
    print(f'    Uniform ref:  {u6:.3f} ({100*u6/dp_ret:5.1f}% of DP)')
    print(f'    Informed ref: {i6:.3f} ({100*i6/dp_ret:5.1f}% of DP)')
    if i6 - u6 > 2.0:
        print(f'    CONFIRMED: Informed reference helps by {i6 - u6:.1f}')
    else:
        print(f'    REJECTED: Reference policy quality does not matter much')

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Panel 1: Greedy vs Stochastic
    ax = axes[0, 0]
    ax.bar([0, 1], [np.mean(e1_stoch), np.mean(e1_greedy)],
           yerr=[np.std(e1_stoch)/np.sqrt(N_SEEDS), np.std(e1_greedy)/np.sqrt(N_SEEDS)],
           color=[COLORS['orange'], COLORS['blue']], capsize=5)
    ax.axhline(dp_ret, color=COLORS['black'], ls='--', alpha=0.7, label='DP Oracle')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stochastic', 'Greedy'])
    ax.set_ylabel('Return')
    ax.set_title('H1: Evaluation Method')
    ax.legend(fontsize=7)

    # Panel 2: Segment Length
    ax = axes[0, 1]
    lengths = sorted(e2_results.keys())
    means = [np.mean(e2_results[L]) for L in lengths]
    ses = [np.std(e2_results[L])/np.sqrt(N_SEEDS) for L in lengths]
    ax.errorbar(lengths, means, yerr=ses, marker='o', color=COLORS['green'], capsize=3)
    ax.axhline(dp_ret, color=COLORS['black'], ls='--', alpha=0.7)
    ax.set_xlabel('Segment Length $L$')
    ax.set_ylabel('Return')
    ax.set_title('H2: Segment Length')

    # Panel 3: Lambda
    ax = axes[0, 2]
    lams = sorted(e3_results.keys())
    means = [np.mean(e3_results[l]) for l in lams]
    ses = [np.std(e3_results[l])/np.sqrt(N_SEEDS) for l in lams]
    ax.errorbar(lams, means, yerr=ses, marker='s', color=COLORS['purple'], capsize=3)
    ax.axhline(dp_ret, color=COLORS['black'], ls='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda_{\mathrm{KL}}$')
    ax.set_ylabel('Return')
    ax.set_title('H3: Lambda Range')

    # Panel 4: Training Budget (heatmap-style)
    ax = axes[1, 0]
    epoch_vals = [300, 1000]
    lr_vals = [1e-3, 5e-3, 1e-2]
    grid = np.zeros((len(epoch_vals), len(lr_vals)))
    for i, ep in enumerate(epoch_vals):
        for j, lr in enumerate(lr_vals):
            grid[i, j] = np.mean(e4_results[(ep, lr)])
    im = ax.imshow(grid, cmap=CMAP_SEQ, aspect='auto')
    ax.set_xticks(range(len(lr_vals)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in lr_vals], fontsize=7)
    ax.set_yticks(range(len(epoch_vals)))
    ax.set_yticklabels([str(e) for e in epoch_vals])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Epochs')
    ax.set_title('H4: Training Budget')
    for i in range(len(epoch_vals)):
        for j in range(len(lr_vals)):
            ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center', fontsize=8)

    # Panel 5: Grid Size
    ax = axes[1, 1]
    K_vals = sorted(e5_results.keys())
    means = [np.mean(e5_results[K]) for K in K_vals]
    ses = [np.std(e5_results[K])/np.sqrt(N_SEEDS) for K in K_vals]
    ax.errorbar(K_vals, means, yerr=ses, marker='^', color=COLORS['red'], capsize=3)
    ax.axhline(e5_dp, color=COLORS['black'], ls='--', alpha=0.7, label='5x5 DP')
    ax.set_xlabel('Comparisons $K$')
    ax.set_ylabel('Return')
    ax.set_title('H5: 5x5 Grid')
    ax.legend(fontsize=7)

    # Panel 6: Reference Policy
    ax = axes[1, 2]
    ax.bar([0, 1], [np.mean(e6_uniform), np.mean(e6_informed)],
           yerr=[np.std(e6_uniform)/np.sqrt(N_SEEDS), np.std(e6_informed)/np.sqrt(N_SEEDS)],
           color=[COLORS['orange'], COLORS['cyan']], capsize=5)
    ax.axhline(dp_ret, color=COLORS['black'], ls='--', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Uniform Ref', 'Informed Ref'])
    ax.set_ylabel('Return')
    ax.set_title('H6: Reference Policy')

    fig.suptitle('DPO Diagnostic Study: Six Hypotheses', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(OUTPUT_DIR, 'dpo_diagnosis.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {fig_path}')

    print(f'\nOUTPUT FILES:')
    print(f'  {fig_path}')


if __name__ == '__main__':
    run_diagnostics()
