"""
DPO Simulation Study: Direct Preference Optimization in a Job Search Model
Chapter 8 - RLHF & Preference Learning

McCall-style job search with compensating differentials.
Compares four methods:
  1. DP Oracle (value iteration with true utility)
  2. RLHF - Neural Net reward model (MLP, Bradley-Terry MLE)
  3. DPO - Tabular softmax policy trained via DPO loss
  4. RLHF - Correct structural model (alpha*log(w) + (1-alpha)*z)

Key experiment: segment length ablation showing DPO degrades with horizon.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE, FIG_DOUBLE
apply_style()

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

WAGE_LEVELS = 8
AMENITY_LEVELS = 7
W = WAGE_LEVELS
Z = AMENITY_LEVELS
ALPHA_TRUE = 0.6
GAMMA = 0.95
MAX_STEPS = 200

WAGES = np.array([20, 28, 38, 50, 65, 82, 100, 125], dtype=float)
AMENITIES = np.linspace(0, 6, AMENITY_LEVELS)
BENEFIT_WAGE = WAGES[1]

P_LAYOFF = 0.05

COMPARISON_COUNTS = [25, 50, 100, 200, 500, 1000, 2000, 5000]
SEGMENT_LENGTH = 15

N_SEEDS = 30
N_SEEDS_ABLATION = 20

NN_INPUT_DIM = 4
NN_HIDDEN = 32
NN_LR = 1e-3
NN_EPOCHS = 100
NN_BATCH = 64

DPO_LR = 1e-3
DPO_EPOCHS = 500
DPO_LAMBDAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

SEGMENT_LENGTHS = [1, 3, 5, 10, 15, 30]
K_HORIZON = 2000

MASTER_SEED = 42
OUTPUT_DIR = 'ch08_rlhf/sims'

# =============================================================================
# State space: 112 states
#   States 0..55:  Searching with offer (w_i, z_j), index = i*Z + j
#   States 56..111: Employed at (w_i, z_j), index = 56 + i*Z + j
# Actions: 0 = Accept/Stay, 1 = Reject/Quit
# =============================================================================

NUM_WZ = W * Z  # 56
NUM_STATES = 2 * NUM_WZ  # 112


def _searching_state(wi, zi):
    return wi * Z + zi


def _employed_state(wi, zi):
    return NUM_WZ + wi * Z + zi


def _decode_state(si):
    """Returns (is_employed, wi, zi)."""
    if si < NUM_WZ:
        wi, zi = divmod(si, Z)
        return False, wi, zi
    else:
        wi, zi = divmod(si - NUM_WZ, Z)
        return True, wi, zi


# Precompute per-state features
LOG_WAGES_BY_STATE = np.zeros(NUM_STATES)
AMENITIES_BY_STATE = np.zeros(NUM_STATES)
IS_EMPLOYED = np.zeros(NUM_STATES)

for _si in range(NUM_STATES):
    _emp, _wi, _zi = _decode_state(_si)
    LOG_WAGES_BY_STATE[_si] = np.log(WAGES[_wi])
    AMENITIES_BY_STATE[_si] = AMENITIES[_zi]
    IS_EMPLOYED[_si] = float(_emp)

REWARD_LOGW = np.zeros(NUM_STATES)
REWARD_Z = np.zeros(NUM_STATES)
ZBAR = np.mean(AMENITIES)

for _si in range(NUM_STATES):
    _emp, _wi, _zi = _decode_state(_si)
    if _emp:
        REWARD_LOGW[_si] = np.log(WAGES[_wi])
        REWARD_Z[_si] = AMENITIES[_zi]
    else:
        REWARD_LOGW[_si] = np.log(BENEFIT_WAGE)
        REWARD_Z[_si] = 0.0

BENEFIT_UTILITY = ALPHA_TRUE * np.log(BENEFIT_WAGE) + (1 - ALPHA_TRUE) * 0.0
TRUE_REWARD_VEC = np.zeros(NUM_STATES)
for _si in range(NUM_STATES):
    _emp, _wi, _zi = _decode_state(_si)
    if _emp:
        TRUE_REWARD_VEC[_si] = ALPHA_TRUE * np.log(WAGES[_wi]) + (1 - ALPHA_TRUE) * AMENITIES[_zi]
    else:
        TRUE_REWARD_VEC[_si] = BENEFIT_UTILITY

STATE_FEATURES = np.zeros((NUM_STATES, 2, NN_INPUT_DIM))
_max_log_w = np.log(WAGES[-1])
_max_z = AMENITIES[-1]
for _si in range(NUM_STATES):
    for _a in range(2):
        STATE_FEATURES[_si, _a] = np.array([
            LOG_WAGES_BY_STATE[_si] / _max_log_w,
            AMENITIES_BY_STATE[_si] / _max_z,
            IS_EMPLOYED[_si],
            float(_a)])


# =============================================================================
# Environment
# =============================================================================

class JobSearchEnv:
    def __init__(self):
        self.W = W
        self.Z = Z
        self.num_states = NUM_STATES
        self.num_actions = 2
        self.state = 0
        self._build_offer_distribution()
        self._build_transition_matrices()

    def _build_offer_distribution(self):
        self.offer_probs = np.zeros((self.W, self.Z))
        for i in range(self.W):
            for j in range(self.Z):
                target_j = (self.Z - 1) * (1 - i / (self.W - 1))
                self.offer_probs[i, j] = np.exp(-0.5 * ((j - target_j) / 1.5)**2)
        self.offer_probs /= self.offer_probs.sum()
        self._offer_flat = self.offer_probs.ravel()

    def _build_transition_matrices(self):
        nS, nA = self.num_states, self.num_actions
        self.T = np.zeros((nA, nS, nS))
        for si in range(nS):
            emp, wi, zi = _decode_state(si)
            if not emp:
                self.T[0, si, _employed_state(wi, zi)] = 1.0
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[1, si, _searching_state(wi2, zi2)] += self.offer_probs[wi2, zi2]
            else:
                self.T[0, si, si] = 1.0 - P_LAYOFF
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[0, si, _searching_state(wi2, zi2)] += P_LAYOFF * self.offer_probs[wi2, zi2]
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[1, si, _searching_state(wi2, zi2)] += self.offer_probs[wi2, zi2]

    def reset(self):
        flat_idx = np.random.choice(NUM_WZ, p=self._offer_flat)
        self.state = flat_idx
        return self.state

    def _draw_searching_state(self):
        return np.random.choice(NUM_WZ, p=self._offer_flat)

    def step(self, action):
        r = TRUE_REWARD_VEC[self.state]
        emp, wi, zi = _decode_state(self.state)
        if not emp:
            if action == 0:
                self.state = _employed_state(wi, zi)
            else:
                self.state = self._draw_searching_state()
        else:
            if action == 1:
                self.state = self._draw_searching_state()
            elif np.random.random() < P_LAYOFF:
                self.state = self._draw_searching_state()
        return self.state, r, False


# =============================================================================
# DP
# =============================================================================

def value_iteration_vec(env, R, tol=1e-10, max_iter=5000):
    nS = env.num_states
    V = np.zeros(nS)
    for it in range(max_iter):
        Q = R[None, :] + GAMMA * (env.T @ V)[:]
        V_new = Q.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = Q.argmax(axis=0)
    return V, policy, it + 1


def policy_eval_vec(env, policy, R, tol=1e-10, max_iter=5000):
    nS = env.num_states
    V = np.zeros(nS)
    T_pi = env.T[policy, np.arange(nS)]
    for it in range(max_iter):
        V_new = R + GAMMA * T_pi @ V
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V_new


# =============================================================================
# Segment Generation
# =============================================================================

def generate_segment(env, policy, length):
    gamma_powers = GAMMA ** np.arange(length)
    states = np.zeros(length, dtype=int)
    actions = np.zeros(length, dtype=int)
    s = env.reset()
    for t in range(length):
        if policy == 'random':
            a = np.random.randint(2)
        else:
            a = int(policy[s])
        ns, r, _ = env.step(a)
        states[t] = s
        actions[t] = a
        s = ns
    total_reward = np.dot(gamma_powers, TRUE_REWARD_VEC[states])
    return states, actions, total_reward


def generate_segment_from_state(env, s0, policy, length):
    gamma_powers = GAMMA ** np.arange(length)
    env.state = s0
    states = np.zeros(length, dtype=int)
    actions = np.zeros(length, dtype=int)
    for t in range(length):
        s = env.state
        if policy == 'random':
            a = np.random.randint(2)
        else:
            a = int(policy[s])
        ns, r, _ = env.step(a)
        states[t] = s
        actions[t] = a
    total_reward = np.dot(gamma_powers, TRUE_REWARD_VEC[states])
    return states, actions, total_reward


def generate_comparisons(env, K, length, policy='random'):
    """Generate K comparisons (cross-state pairing for RLHF/structural)."""
    w_states = np.zeros((K, length), dtype=int)
    l_states = np.zeros((K, length), dtype=int)
    w_actions = np.zeros((K, length), dtype=int)
    l_actions = np.zeros((K, length), dtype=int)
    for count in range(K):
        s1, a1, r1 = generate_segment(env, policy, length)
        s2, a2, r2 = generate_segment(env, policy, length)
        diff = np.clip(r1 - r2, -500, 500)
        if np.random.random() < 1.0 / (1.0 + np.exp(-diff)):
            w_states[count] = s1; l_states[count] = s2
            w_actions[count] = a1; l_actions[count] = a2
        else:
            w_states[count] = s2; l_states[count] = s1
            w_actions[count] = a2; l_actions[count] = a1
    return w_states, l_states, w_actions, l_actions


def generate_comparisons_same_state(env, K, length, policy='random'):
    """Generate K comparisons where each pair starts from the same state (for DPO)."""
    w_states = np.zeros((K, length), dtype=int)
    l_states = np.zeros((K, length), dtype=int)
    w_actions = np.zeros((K, length), dtype=int)
    l_actions = np.zeros((K, length), dtype=int)
    for count in range(K):
        s0 = env._draw_searching_state()
        s1, a1, r1 = generate_segment_from_state(env, s0, policy, length)
        s2, a2, r2 = generate_segment_from_state(env, s0, policy, length)
        diff = np.clip(r1 - r2, -500, 500)
        if np.random.random() < 1.0 / (1.0 + np.exp(-diff)):
            w_states[count] = s1; l_states[count] = s2
            w_actions[count] = a1; l_actions[count] = a2
        else:
            w_states[count] = s2; l_states[count] = s1
            w_actions[count] = a2; l_actions[count] = a1
    return w_states, l_states, w_actions, l_actions


# =============================================================================
# Neural Net Reward Model
# =============================================================================

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NN_INPUT_DIM, NN_HIDDEN), nn.ReLU(),
            nn.Linear(NN_HIDDEN, NN_HIDDEN), nn.ReLU(),
            nn.Linear(NN_HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_net(w_states, l_states, w_actions, l_actions, length):
    K = w_states.shape[0]
    gamma_powers = GAMMA ** np.arange(length)
    w_feats = STATE_FEATURES[w_states, w_actions]
    l_feats = STATE_FEATURES[l_states, l_actions]
    mask = np.broadcast_to(gamma_powers[None, :], (K, length)).copy()
    w_t = torch.tensor(w_feats, dtype=torch.float32)
    l_t = torch.tensor(l_feats, dtype=torch.float32)
    m_t = torch.tensor(mask, dtype=torch.float32)

    model = RewardNet()
    optimizer = optim.Adam(model.parameters(), lr=NN_LR)
    indices = np.arange(K)

    for epoch in range(NN_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, K, NN_BATCH):
            batch = indices[start:start + NN_BATCH]
            bw, bl, bm = w_t[batch], l_t[batch], m_t[batch]
            B, L, D = bw.shape
            r_w = (model(bw.reshape(B*L, D)).reshape(B, L) * bm).sum(1)
            r_l = (model(bl.reshape(B*L, D)).reshape(B, L) * bm).sum(1)
            loss = -torch.nn.functional.logsigmoid(r_w - r_l).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def extract_reward_table(model):
    all_feats = STATE_FEATURES.reshape(NUM_STATES * 2, NN_INPUT_DIM)
    model.eval()
    with torch.no_grad():
        rewards = model(torch.tensor(all_feats, dtype=torch.float32)).numpy()
    return rewards.reshape(NUM_STATES, 2).mean(axis=1)


# =============================================================================
# Structural Model
# =============================================================================

def fit_structural(w_states, l_states, length):
    K = w_states.shape[0]
    gp = GAMMA ** np.arange(length)
    w_logw = (REWARD_LOGW[w_states] * gp[None, :]).sum(axis=1)
    w_z = (REWARD_Z[w_states] * gp[None, :]).sum(axis=1)
    l_logw = (REWARD_LOGW[l_states] * gp[None, :]).sum(axis=1)
    l_z = (REWARD_Z[l_states] * gp[None, :]).sum(axis=1)
    d_logw = w_logw - l_logw
    d_z = w_z - l_z

    def nll(alpha):
        diff = np.clip(alpha * d_logw + (1 - alpha) * d_z, -500, 500)
        return -np.sum(np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-15))

    result = minimize_scalar(nll, bounds=(0.01, 0.99), method='bounded')
    return result.x


def make_reward_vec(alpha):
    R = np.zeros(NUM_STATES)
    for si in range(NUM_STATES):
        emp, wi, zi = _decode_state(si)
        if emp:
            R[si] = alpha * np.log(WAGES[wi]) + (1 - alpha) * AMENITIES[zi]
        else:
            R[si] = alpha * np.log(BENEFIT_WAGE) + (1 - alpha) * 0.0
    return R


# =============================================================================
# DPO Implementation
# =============================================================================

def _log_sigmoid(x):
    """Numerically stable log sigmoid."""
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def train_dpo(w_states, l_states, w_actions, l_actions, lam,
              lr=DPO_LR, epochs=DPO_EPOCHS):
    """Train DPO tabular softmax policy.

    Policy: pi(a=0|s) = sigmoid(phi[s]), pi(a=1|s) = 1 - sigmoid(phi[s]).
    Reference: pi_ref(a|s) = 0.5 (cancels in log-ratio difference).
    Returns: (phi, final_loss).
    """
    K, L = w_states.shape
    phi = np.zeros(NUM_STATES)

    # Precompute static arrays
    w_accept = (w_actions == 0).astype(np.float64)  # (K, L)
    l_accept = (l_actions == 0).astype(np.float64)
    signs_w = 1 - 2 * w_actions  # +1 for a=0, -1 for a=1
    signs_l = 1 - 2 * l_actions
    w_flat = w_states.ravel()
    l_flat = l_states.ravel()

    # Adam state
    m = np.zeros(NUM_STATES)
    v = np.zeros(NUM_STATES)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    final_loss = np.log(2.0)

    for epoch in range(epochs):
        probs = _sigmoid(phi)

        # Log-probs along trajectories
        log_pi_w = _log_sigmoid(signs_w * phi[w_states])  # (K, L)
        log_pi_l = _log_sigmoid(signs_l * phi[l_states])
        logr_w = log_pi_w.sum(axis=1)  # (K,)
        logr_l = log_pi_l.sum(axis=1)

        diff = lam * (logr_w - logr_l)
        final_loss = -_log_sigmoid(diff).mean()

        # Gradient via bincount
        weights = _sigmoid(-diff) * lam  # (K,)
        grad = np.zeros(NUM_STATES)

        weighted_w = ((w_accept - probs[w_states]) * weights[:, None]).ravel()
        grad += np.bincount(w_flat, weights=weighted_w, minlength=NUM_STATES)

        weighted_l = ((l_accept - probs[l_states]) * weights[:, None]).ravel()
        grad -= np.bincount(l_flat, weights=weighted_l, minlength=NUM_STATES)

        grad = -grad / K  # loss gradient (we minimize)

        # Adam
        t_step = epoch + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_step)
        v_hat = v / (1 - beta2**t_step)
        phi -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return phi, final_loss


def dpo_greedy_policy(phi):
    """Greedy policy: a=0 (accept/stay) if sigmoid(phi) > 0.5, else a=1."""
    return (phi <= 0).astype(int)


def run_dpo(w_states, l_states, w_actions, l_actions):
    """Train DPO across lambda sweep, return best greedy policy."""
    best_loss = np.inf
    best_phi = None
    for lam in DPO_LAMBDAS:
        phi, loss = train_dpo(w_states, l_states, w_actions, l_actions, lam)
        if loss < best_loss:
            best_loss = loss
            best_phi = phi.copy()
    return dpo_greedy_policy(best_phi), best_loss


# =============================================================================
# Diagnostics helpers
# =============================================================================

def mean_accepted_amenity(policy):
    vals = [AMENITIES[zi] for wi in range(W) for zi in range(Z)
            if policy[_searching_state(wi, zi)] == 0]
    return np.mean(vals) if vals else 0.0


def mean_accepted_wage(policy):
    vals = [WAGES[wi] for wi in range(W) for zi in range(Z)
            if policy[_searching_state(wi, zi)] == 0]
    return np.mean(vals) if vals else 0.0


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('=' * 80)
    print('DPO SIMULATION STUDY: JOB SEARCH WITH COMPENSATING DIFFERENTIALS')
    print('=' * 80)
    print(f'\nTimestamp: {timestamp}')
    print(f'Python: {sys.version.split()[0]}, NumPy: {np.__version__}, '
          f'PyTorch: {torch.__version__}')
    print(f'\nPARAMETERS:')
    print(f'  Wage levels: {W}, Amenity levels: {Z}')
    print(f'  States: {NUM_STATES} ({NUM_WZ} searching + {NUM_WZ} employed)')
    print(f'  True alpha: {ALPHA_TRUE}, gamma: {GAMMA}')
    print(f'  Wages: {WAGES}')
    print(f'  Amenities: {AMENITIES}')
    print(f'  Benefit wage: {BENEFIT_WAGE}, Layoff prob: {P_LAYOFF}')
    print(f'  Segment length: {SEGMENT_LENGTH}, K values: {COMPARISON_COUNTS}')
    print(f'  Seeds: {N_SEEDS} (main), {N_SEEDS_ABLATION} (horizon)')
    print(f'  NN: {NN_INPUT_DIM}->{NN_HIDDEN}->{NN_HIDDEN}->1, lr={NN_LR}, '
          f'epochs={NN_EPOCHS}, batch={NN_BATCH}')
    print(f'  DPO: {NUM_STATES} logits, lr={DPO_LR}, epochs={DPO_EPOCHS}')
    print(f'  DPO lambdas: {DPO_LAMBDAS}')
    print(f'  Horizon sweep: L={SEGMENT_LENGTHS}, K={K_HORIZON}')

    env = JobSearchEnv()
    R_true = TRUE_REWARD_VEC.copy()

    # --- DP Oracle ---
    print('\n' + '=' * 80)
    print('1. DP ORACLE')
    print('=' * 80)
    V_star, pi_star, vi_iters = value_iteration_vec(env, R_true)
    dp_v0 = np.dot(env._offer_flat, V_star[:NUM_WZ])
    maz_opt = mean_accepted_amenity(pi_star)
    mwz_opt = mean_accepted_wage(pi_star)
    print(f'  Converged in {vi_iters} iterations')
    print(f'  V*(s0) = {dp_v0:.4f}')
    print(f'  Mean accepted amenity: {maz_opt:.2f}, wage: {mwz_opt:.0f}')

    # --- Experiment 1: Sample Complexity ---
    print('\n' + '=' * 80)
    print(f'EXPERIMENT 1: SAMPLE COMPLEXITY (L={SEGMENT_LENGTH}, {N_SEEDS} seeds)')
    print('=' * 80)

    nn_res = {K: [] for K in COMPARISON_COUNTS}
    dpo_res = {K: [] for K in COMPARISON_COUNTS}
    co_res = {K: [] for K in COMPARISON_COUNTS}
    co_alphas = {K: [] for K in COMPARISON_COUNTS}

    for K in COMPARISON_COUNTS:
        print(f'\n  K = {K}', end='', flush=True)
        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000 + K

            # Cross-state data for NN RLHF and structural
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            ws, ls, wa, la = generate_comparisons(env, K, SEGMENT_LENGTH)

            # Same-state data for DPO
            np.random.seed(rng_seed + 500000)
            dws, dls, dwa, dla = generate_comparisons_same_state(
                env, K, SEGMENT_LENGTH)

            # NN RLHF
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            model = train_reward_net(ws, ls, wa, la, SEGMENT_LENGTH)
            R_nn = extract_reward_table(model)
            _, nn_pol, _ = value_iteration_vec(env, R_nn)
            V_nn = policy_eval_vec(env, nn_pol, R_true)
            nn_res[K].append(np.dot(env._offer_flat, V_nn[:NUM_WZ]))

            # Structural
            alpha_hat = fit_structural(ws, ls, SEGMENT_LENGTH)
            R_co = make_reward_vec(alpha_hat)
            _, pi_co, _ = value_iteration_vec(env, R_co)
            V_co = policy_eval_vec(env, pi_co, R_true)
            co_res[K].append(np.dot(env._offer_flat, V_co[:NUM_WZ]))
            co_alphas[K].append(alpha_hat)

            # DPO
            dpo_pol, _ = run_dpo(dws, dls, dwa, dla)
            V_dpo = policy_eval_vec(env, dpo_pol, R_true)
            dpo_res[K].append(np.dot(env._offer_flat, V_dpo[:NUM_WZ]))

            if seed % 10 == 9:
                print('.', end='', flush=True)

        nn_m = np.mean(nn_res[K]); nn_s = np.std(nn_res[K]) / np.sqrt(N_SEEDS)
        dpo_m = np.mean(dpo_res[K]); dpo_s = np.std(dpo_res[K]) / np.sqrt(N_SEEDS)
        co_m = np.mean(co_res[K]); co_s = np.std(co_res[K]) / np.sqrt(N_SEEDS)
        print(f'\n    NN RLHF:    {nn_m:7.3f} +/- {nn_s:.3f}  ({100*nn_m/dp_v0:.1f}%)')
        print(f'    DPO:        {dpo_m:7.3f} +/- {dpo_s:.3f}  ({100*dpo_m/dp_v0:.1f}%)')
        print(f'    Structural: {co_m:7.3f} +/- {co_s:.3f}  '
              f'({100*co_m/dp_v0:.1f}%, alpha={np.mean(co_alphas[K]):.3f})')

    # --- Experiment 2: Segment Length Sweep ---
    print('\n' + '=' * 80)
    print(f'EXPERIMENT 2: SEGMENT LENGTH SWEEP (K={K_HORIZON}, '
          f'{N_SEEDS_ABLATION} seeds)')
    print('=' * 80)

    nn_L_res = {L: [] for L in SEGMENT_LENGTHS}
    dpo_L_res = {L: [] for L in SEGMENT_LENGTHS}

    for L in SEGMENT_LENGTHS:
        print(f'\n  L = {L}', end='', flush=True)
        for seed in range(N_SEEDS_ABLATION):
            rng_seed = MASTER_SEED + seed * 10000 + L * 100

            # NN RLHF
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            ws, ls, wa, la = generate_comparisons(env, K_HORIZON, L)
            model = train_reward_net(ws, ls, wa, la, L)
            R_nn = extract_reward_table(model)
            _, nn_pol, _ = value_iteration_vec(env, R_nn)
            V_nn = policy_eval_vec(env, nn_pol, R_true)
            nn_L_res[L].append(np.dot(env._offer_flat, V_nn[:NUM_WZ]))

            # DPO
            np.random.seed(rng_seed + 500000)
            dws, dls, dwa, dla = generate_comparisons_same_state(
                env, K_HORIZON, L)
            dpo_pol, _ = run_dpo(dws, dls, dwa, dla)
            V_dpo = policy_eval_vec(env, dpo_pol, R_true)
            dpo_L_res[L].append(np.dot(env._offer_flat, V_dpo[:NUM_WZ]))

            if seed % 5 == 4:
                print('.', end='', flush=True)

        nn_m = np.mean(nn_L_res[L])
        nn_s = np.std(nn_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
        dpo_m = np.mean(dpo_L_res[L])
        dpo_s = np.std(dpo_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
        print(f'\n    NN RLHF: {nn_m:7.3f} +/- {nn_s:.3f}  ({100*nn_m/dp_v0:.1f}%)')
        print(f'    DPO:     {dpo_m:7.3f} +/- {dpo_s:.3f}  ({100*dpo_m/dp_v0:.1f}%)')

    # --- Experiment 3: Diagnostics ---
    print('\n' + '=' * 80)
    print('EXPERIMENT 3: DIAGNOSTICS (K=5000, single seed)')
    print('=' * 80)
    K_diag = 5000
    rng_seed = MASTER_SEED + 99999
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    ws, ls, wa, la = generate_comparisons(env, K_diag, SEGMENT_LENGTH)

    np.random.seed(rng_seed + 500000)
    dws, dls, dwa, dla = generate_comparisons_same_state(
        env, K_diag, SEGMENT_LENGTH)

    diag_results = []

    # NN RLHF
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    model = train_reward_net(ws, ls, wa, la, SEGMENT_LENGTH)
    R_nn = extract_reward_table(model)
    _, pi_nn, _ = value_iteration_vec(env, R_nn)
    V_nn = policy_eval_vec(env, pi_nn, R_true)
    diag_results.append(('NN RLHF',
                         np.mean(pi_nn == pi_star) * 100,
                         np.corrcoef(V_nn, V_star)[0, 1],
                         mean_accepted_amenity(pi_nn),
                         mean_accepted_wage(pi_nn), None))

    # DPO
    dpo_pol, dpo_loss = run_dpo(dws, dls, dwa, dla)
    V_dpo = policy_eval_vec(env, dpo_pol, R_true)
    diag_results.append(('DPO',
                         np.mean(dpo_pol == pi_star) * 100,
                         np.corrcoef(V_dpo, V_star)[0, 1],
                         mean_accepted_amenity(dpo_pol),
                         mean_accepted_wage(dpo_pol), None))

    # Structural
    alpha_hat = fit_structural(ws, ls, SEGMENT_LENGTH)
    R_co = make_reward_vec(alpha_hat)
    _, pi_co, _ = value_iteration_vec(env, R_co)
    V_co = policy_eval_vec(env, pi_co, R_true)
    diag_results.append(('Structural',
                         np.mean(pi_co == pi_star) * 100,
                         np.corrcoef(V_co, V_star)[0, 1],
                         mean_accepted_amenity(pi_co),
                         mean_accepted_wage(pi_co), alpha_hat))

    hdr = (f'  {"Method":<15s} | {"Agree%":>7s} | {"V corr":>7s} | '
           f'{"Mean z":>7s} | {"Mean w":>7s} | {"alpha":>7s}')
    print(hdr)
    print('-' * len(hdr))
    for name, agree, vcorr, maz, mwz, alpha_est in diag_results:
        alpha_str = f'{alpha_est:.3f}' if alpha_est is not None else '    ---'
        print(f'  {name:<15s} | {agree:7.1f} | {vcorr:7.4f} | '
              f'{maz:7.2f} | {mwz:7.1f} | {alpha_str:>7s}')
    print(f'  {"Optimal":<15s} | {100.0:7.1f} | {1.0000:7.4f} | '
          f'{maz_opt:7.2f} | {mwz_opt:7.1f} |     ---')

    # --- Summary Tables ---
    print('\n' + '=' * 80)
    print('MAIN RESULTS TABLE')
    print('=' * 80)
    hdr = f'{"Method":<20s} | {"K":>5s} | {"V(s0)":>10s} | {"% DP":>6s}'
    print(hdr)
    print('-' * len(hdr))
    print(f'{"DP Oracle":<20s} | {"--":>5s} | {dp_v0:10.3f} | {100.0:6.1f}')
    print('-' * len(hdr))
    for K in COMPARISON_COUNTS:
        for name, res in [('NN RLHF', nn_res), ('DPO', dpo_res),
                          ('Structural', co_res)]:
            m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
            print(f'{name:<20s} | {K:5d} | {m:10.3f} | {100*m/dp_v0:6.1f}')
        if K != COMPARISON_COUNTS[-1]:
            print('-' * len(hdr))

    print('\n' + '=' * 80)
    print(f'SEGMENT LENGTH ABLATION (K={K_HORIZON})')
    print('=' * 80)
    hdr2 = (f'{"L":>5s} | {"NN RLHF":>15s} | {"% DP":>6s} | '
            f'{"DPO":>15s} | {"% DP":>6s}')
    print(hdr2)
    print('-' * len(hdr2))
    for L in SEGMENT_LENGTHS:
        nn_m = np.mean(nn_L_res[L])
        nn_s = np.std(nn_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
        dpo_m = np.mean(dpo_L_res[L])
        dpo_s = np.std(dpo_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
        print(f'{L:5d} | {nn_m:7.3f}+/-{nn_s:.3f} | {100*nn_m/dp_v0:6.1f} | '
              f'{dpo_m:7.3f}+/-{dpo_s:.3f} | {100*dpo_m/dp_v0:6.1f}')

    # --- Verification ---
    print('\n' + '=' * 80)
    print('VERIFICATION')
    print('=' * 80)
    print(f'  1. DP V*(s0) = {dp_v0:.3f} -> '
          f'{"PASS" if abs(dp_v0 - 74.13) < 0.5 else "CHECK"}')
    co_hi = np.mean(co_res[COMPARISON_COUNTS[-1]])
    print(f'  2. Structural at K={COMPARISON_COUNTS[-1]}: '
          f'{100*co_hi/dp_v0:.1f}% -> '
          f'{"PASS" if co_hi/dp_v0 > 0.99 else "CHECK"}')
    nn_hi = np.mean(nn_res[COMPARISON_COUNTS[-1]])
    print(f'  3. NN RLHF at K={COMPARISON_COUNTS[-1]}: '
          f'{100*nn_hi/dp_v0:.1f}% -> '
          f'{"PASS" if nn_hi/dp_v0 > 0.99 else "CHECK"}')
    dpo_hi = np.mean(dpo_res[COMPARISON_COUNTS[-1]])
    print(f'  4. DPO V > 0% of DP: {100*dpo_hi/dp_v0:.1f}% -> '
          f'{"PASS" if dpo_hi > 0 else "FAIL"}')
    dpo_lo = np.mean(dpo_res[COMPARISON_COUNTS[0]])
    print(f'  5. DPO improves with K: {100*dpo_lo/dp_v0:.1f}% -> '
          f'{100*dpo_hi/dp_v0:.1f}% -> '
          f'{"PASS" if dpo_hi > dpo_lo else "CHECK"}')
    dpo_L1 = np.mean(dpo_L_res[1]) if 1 in dpo_L_res else 0
    dpo_L30 = np.mean(dpo_L_res[30]) if 30 in dpo_L_res else 0
    print(f'  6. DPO at L=1 >= L=30: {100*dpo_L1/dp_v0:.1f}% vs '
          f'{100*dpo_L30/dp_v0:.1f}% -> '
          f'{"PASS" if dpo_L1 >= dpo_L30 else "CHECK"}')

    # --- LaTeX Tables ---
    tex1 = os.path.join(OUTPUT_DIR, 'job_search_dpo_results.tex')
    with open(tex1, 'w') as f:
        f.write('\\begin{tabular}{llrr}\n\\hline\n')
        f.write('Method & $K$ & $V^\\pi(s_0)$ & \\% of DP \\\\\n\\hline\n')
        f.write(f'DP Oracle & --- & ${dp_v0:.2f}$ & $100.0$ \\\\\n\\hline\n')
        for K in COMPARISON_COUNTS:
            for name, res in [('NN RLHF', nn_res), ('DPO', dpo_res),
                              ('Structural', co_res)]:
                m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
                f.write(f'{name} & ${K}$ & ${m:.2f} \\pm {s:.2f}$ & '
                        f'${100*m/dp_v0:.1f}$ \\\\\n')
            if K != COMPARISON_COUNTS[-1]:
                f.write('\\hline\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'\nSaved: {tex1}')

    tex2 = os.path.join(OUTPUT_DIR, 'job_search_dpo_horizon.tex')
    with open(tex2, 'w') as f:
        f.write('\\begin{tabular}{rrrrr}\n\\hline\n')
        f.write('$L$ & NN RLHF $V^\\pi(s_0)$ & \\% of DP '
                '& DPO $V^\\pi(s_0)$ & \\% of DP \\\\\n\\hline\n')
        for L in SEGMENT_LENGTHS:
            nn_m = np.mean(nn_L_res[L])
            nn_s = np.std(nn_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
            dpo_m = np.mean(dpo_L_res[L])
            dpo_s = np.std(dpo_L_res[L]) / np.sqrt(N_SEEDS_ABLATION)
            f.write(f'${L}$ & ${nn_m:.2f} \\pm {nn_s:.2f}$ & '
                    f'${100*nn_m/dp_v0:.1f}$ & '
                    f'${dpo_m:.2f} \\pm {dpo_s:.2f}$ & '
                    f'${100*dpo_m/dp_v0:.1f}$ \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex2}')

    tex3 = os.path.join(OUTPUT_DIR, 'job_search_dpo_diagnostics.tex')
    with open(tex3, 'w') as f:
        f.write('\\begin{tabular}{lrrrrr}\n\\hline\n')
        f.write('Method & Policy agree.~(\\%) & $V^\\pi$ corr.'
                ' & Mean amenity & Mean wage & $\\hat{\\alpha}$ '
                '\\\\\n\\hline\n')
        for name, agree, vcorr, maz, mwz, alpha_est in diag_results:
            alpha_str = f'${alpha_est:.3f}$' if alpha_est is not None else '---'
            f.write(f'{name} & ${agree:.1f}$ & ${vcorr:.3f}$ & ${maz:.2f}$ '
                    f'& ${mwz:.0f}$ & {alpha_str} \\\\\n')
        f.write('\\hline\n')
        f.write(f'Optimal & $100.0$ & $1.000$ & ${maz_opt:.2f}$ '
                f'& ${mwz_opt:.0f}$ & --- \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex3}')

    # --- Figures ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    K_arr = np.array(COMPARISON_COUNTS)
    ax.axhline(dp_v0, color=COLORS['black'], ls='--', label='DP Oracle',
               alpha=0.8)
    for name, res, color, marker in [
        ('NN RLHF', nn_res, COLORS['green'], 'o'),
        ('DPO', dpo_res, COLORS['orange'], 'D'),
        ('Structural', co_res, COLORS['purple'], 's'),
    ]:
        means = [np.mean(res[K]) for K in COMPARISON_COUNTS]
        ses = [np.std(res[K]) / np.sqrt(N_SEEDS) for K in COMPARISON_COUNTS]
        ax.errorbar(K_arr, means, yerr=ses, marker=marker, color=color,
                    label=name, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Preference Comparisons ($K$)')
    ax.set_ylabel('Policy Value $V^\\pi(s_0)$')
    ax.legend(loc='lower right', fontsize=8)
    fig_path1 = os.path.join(OUTPUT_DIR, 'job_search_dpo_sample_complexity.png')
    fig.savefig(fig_path1, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fig_path1}')

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    L_arr = np.array(SEGMENT_LENGTHS)
    ax.axhline(dp_v0, color=COLORS['black'], ls='--', label='DP Oracle',
               alpha=0.8)
    for name, res, color, marker in [
        ('NN RLHF', nn_L_res, COLORS['green'], 'o'),
        ('DPO', dpo_L_res, COLORS['orange'], 'D'),
    ]:
        means = [np.mean(res[L]) for L in SEGMENT_LENGTHS]
        ses = [np.std(res[L]) / np.sqrt(N_SEEDS_ABLATION)
               for L in SEGMENT_LENGTHS]
        ax.errorbar(L_arr, means, yerr=ses, marker=marker, color=color,
                    label=name, capsize=3)
    ax.set_xlabel('Segment Length ($L$)')
    ax.set_ylabel('Policy Value $V^\\pi(s_0)$')
    ax.legend(loc='lower right', fontsize=8)
    fig_path2 = os.path.join(OUTPUT_DIR, 'job_search_dpo_horizon.png')
    fig.savefig(fig_path2, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fig_path2}')

    print(f'\nOUTPUT FILES:')
    for p in [tex1, tex2, tex3, fig_path1, fig_path2]:
        print(f'  {p}')


if __name__ == '__main__':
    run_experiment()
