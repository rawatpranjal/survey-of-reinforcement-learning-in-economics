"""
RLHF Simulation Study: Preference-Based Utility Recovery in a Job Search Model
Chapter 8 - RLHF & Preference Learning

McCall-style job search with offer-visible states and compensating differentials.
Compares five methods:
  1. DP Oracle (value iteration with true utility)
  2. Q-Learning (true scalar reward)
  3. RLHF - Neural Net reward model (MLP, Bradley-Terry MLE)
  4. RLHF - Correct structural model (alpha*log(w) + (1-alpha)*z)
  5. RLHF - Misspecified structural model (alpha*log(w) + (1-alpha)*z_bar, ignores amenity variation)

Ablation: Online vs offline data collection for neural net RLHF at K=1000.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
from scipy import stats
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
from sims.plot_style import apply_style, COLORS, CMAP_SEQ, FIG_SINGLE, FIG_DOUBLE
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

Q_LEARNING_EPISODES = 10000
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_EPSILON = 0.15

COMPARISON_COUNTS = [25, 50, 100, 200, 500, 1000, 2000, 5000]
SEGMENT_LENGTH = 15

N_SEEDS = 30
N_SEEDS_ABLATION = 20
EVAL_EPISODES = 300

NN_INPUT_DIM = 4  # (norm_log_wage, norm_amenity, employed_flag, action)
NN_HIDDEN = 32
NN_LR = 1e-3
NN_EPOCHS = 100
NN_BATCH = 64

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

# Reward-relevant attributes (for structural model fitting):
# Searching states use benefit attributes, not offer attributes
REWARD_LOGW = np.zeros(NUM_STATES)
REWARD_Z = np.zeros(NUM_STATES)
ZBAR = np.mean(AMENITIES)  # 3.0

for _si in range(NUM_STATES):
    _emp, _wi, _zi = _decode_state(_si)
    if _emp:
        REWARD_LOGW[_si] = np.log(WAGES[_wi])
        REWARD_Z[_si] = AMENITIES[_zi]
    else:
        REWARD_LOGW[_si] = np.log(BENEFIT_WAGE)
        REWARD_Z[_si] = 0.0

# True reward: searching states get unemployment benefit, employed get utility
BENEFIT_UTILITY = ALPHA_TRUE * np.log(BENEFIT_WAGE) + (1 - ALPHA_TRUE) * 0.0
TRUE_REWARD_VEC = np.zeros(NUM_STATES)
for _si in range(NUM_STATES):
    _emp, _wi, _zi = _decode_state(_si)
    if _emp:
        TRUE_REWARD_VEC[_si] = ALPHA_TRUE * np.log(WAGES[_wi]) + (1 - ALPHA_TRUE) * AMENITIES[_zi]
    else:
        TRUE_REWARD_VEC[_si] = BENEFIT_UTILITY

# Discount factors for segment
GAMMA_POWERS = GAMMA ** np.arange(SEGMENT_LENGTH)

# Precompute state features lookup: STATE_FEATURES[si, a] = feature vector
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
    """
    McCall-style job search MDP with compensating differentials.
    States 0..55: Searching with offer (w_i, z_j).
    States 56..111: Employed at (w_i, z_j).
    Actions: 0=Accept/Stay, 1=Reject/Quit.
    """

    def __init__(self):
        self.W = W
        self.Z = Z
        self.num_states = NUM_STATES
        self.num_actions = 2
        self.state = 0
        self._build_offer_distribution()
        self._build_transition_matrices()

    def _build_offer_distribution(self):
        """Negatively correlated offers: high wage -> low amenity."""
        self.offer_probs = np.zeros((self.W, self.Z))
        for i in range(self.W):
            for j in range(self.Z):
                # Center amenity distribution on anti-diagonal
                target_j = (self.Z - 1) * (1 - i / (self.W - 1))
                self.offer_probs[i, j] = np.exp(-0.5 * ((j - target_j) / 1.5)**2)
        self.offer_probs /= self.offer_probs.sum()
        self._offer_flat = self.offer_probs.ravel()

    def _build_transition_matrices(self):
        """Dense transition matrices T[a, s, s'] = P(s'|s,a)."""
        nS, nA = self.num_states, self.num_actions
        self.T = np.zeros((nA, nS, nS))

        for si in range(nS):
            emp, wi, zi = _decode_state(si)

            if not emp:
                # Searching with offer (wi, zi)
                # Action 0: Accept -> Employed(wi, zi)
                self.T[0, si, _employed_state(wi, zi)] = 1.0
                # Action 1: Reject -> new random offer (searching state)
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[1, si, _searching_state(wi2, zi2)] += self.offer_probs[wi2, zi2]
            else:
                # Employed at (wi, zi)
                # Action 0: Stay -> Employed(wi, zi) w.p. (1-p_layoff),
                #                   Searching(random offer) w.p. p_layoff
                self.T[0, si, si] = 1.0 - P_LAYOFF
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[0, si, _searching_state(wi2, zi2)] += P_LAYOFF * self.offer_probs[wi2, zi2]
                # Action 1: Quit -> Searching(random offer)
                for wi2 in range(self.W):
                    for zi2 in range(self.Z):
                        self.T[1, si, _searching_state(wi2, zi2)] += self.offer_probs[wi2, zi2]

    def true_reward(self, si):
        return TRUE_REWARD_VEC[si]

    def reset(self):
        flat_idx = np.random.choice(NUM_WZ, p=self._offer_flat)
        self.state = flat_idx  # searching state
        return self.state

    def _draw_searching_state(self):
        flat_idx = np.random.choice(NUM_WZ, p=self._offer_flat)
        return flat_idx  # searching states are 0..55

    def step(self, action):
        r = TRUE_REWARD_VEC[self.state]
        emp, wi, zi = _decode_state(self.state)

        if not emp:
            # Searching with offer
            if action == 0:
                # Accept
                self.state = _employed_state(wi, zi)
            else:
                # Reject -> new offer
                self.state = self._draw_searching_state()
        else:
            # Employed
            if action == 1:
                # Quit
                self.state = self._draw_searching_state()
            elif np.random.random() < P_LAYOFF:
                # Laid off
                self.state = self._draw_searching_state()
            # else: stay employed
        return self.state, r, False


# =============================================================================
# DP (vectorized with dense transition matrices)
# =============================================================================

def value_iteration_vec(env, R, tol=1e-10, max_iter=5000):
    """Vectorized value iteration. R is (nS,) reward vector."""
    nS = env.num_states
    V = np.zeros(nS)
    for it in range(max_iter):
        Q = R[None, :] + GAMMA * (env.T @ V)[:]  # (nA, nS)
        V_new = Q.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = Q.argmax(axis=0)
    return V, policy, it + 1


def policy_eval_vec(env, policy, R, tol=1e-10, max_iter=5000):
    """Vectorized policy evaluation. R is (nS,) reward vector."""
    nS = env.num_states
    V = np.zeros(nS)
    T_pi = env.T[policy, np.arange(nS)]  # (nS, nS)
    for it in range(max_iter):
        V_new = R + GAMMA * T_pi @ V
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V_new


def evaluate_policy_mc(env, policy, n_episodes=EVAL_EPISODES):
    """Monte Carlo evaluation for verification."""
    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for t in range(MAX_STEPS):
            ns, r, _ = env.step(int(policy[s]))
            total += (GAMMA ** t) * r
            s = ns
        returns.append(total)
    return np.mean(returns), np.std(returns) / np.sqrt(len(returns))


# =============================================================================
# Q-Learning
# =============================================================================

def q_learning(env):
    nS, nA = env.num_states, env.num_actions
    Q = np.zeros((nS, nA))
    total_steps = 0
    for _ in range(Q_LEARNING_EPISODES):
        s = env.reset()
        for t in range(MAX_STEPS):
            if np.random.random() < Q_LEARNING_EPSILON:
                a = np.random.randint(nA)
            else:
                a = int(np.argmax(Q[s]))
            ns, r, _ = env.step(a)
            Q[s, a] += Q_LEARNING_ALPHA * (r + GAMMA * np.max(Q[ns]) - Q[s, a])
            total_steps += 1
            s = ns
    return np.argmax(Q, axis=1), total_steps


# =============================================================================
# Segment Generation
# =============================================================================

def generate_segment(env, policy, length=SEGMENT_LENGTH):
    states = np.zeros(length, dtype=int)
    actions = np.zeros(length, dtype=int)
    s = env.reset()
    for t in range(length):
        if isinstance(policy, str) and policy == 'random':
            a = np.random.randint(env.num_actions)
        elif isinstance(policy, np.ndarray):
            a = int(policy[s])
        elif callable(policy):
            a = policy(s)
        else:
            a = int(policy[s])
        ns, r, _ = env.step(a)
        states[t] = s
        actions[t] = a
        s = ns
    true_rewards = TRUE_REWARD_VEC[states]
    total_reward = np.dot(GAMMA_POWERS[:length], true_rewards)
    return states, actions, total_reward


def generate_comparisons(env, K, policy='random'):
    """Generate K comparisons. Returns (w_states, l_states, w_actions, l_actions)."""
    w_states = np.zeros((K, SEGMENT_LENGTH), dtype=int)
    l_states = np.zeros((K, SEGMENT_LENGTH), dtype=int)
    w_actions = np.zeros((K, SEGMENT_LENGTH), dtype=int)
    l_actions = np.zeros((K, SEGMENT_LENGTH), dtype=int)
    count = 0
    while count < K:
        s1, a1, r1 = generate_segment(env, policy)
        s2, a2, r2 = generate_segment(env, policy)
        diff = np.clip(r1 - r2, -500, 500)
        if np.random.random() < 1.0 / (1.0 + np.exp(-diff)):
            w_states[count] = s1; l_states[count] = s2
            w_actions[count] = a1; l_actions[count] = a2
        else:
            w_states[count] = s2; l_states[count] = s1
            w_actions[count] = a2; l_actions[count] = a1
        count += 1
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


def _precompute_nn_tensors(w_states, l_states, w_actions, l_actions, env):
    K, L = w_states.shape
    w_feats = STATE_FEATURES[w_states, w_actions]
    l_feats = STATE_FEATURES[l_states, l_actions]
    mask = np.broadcast_to(GAMMA_POWERS[:L][None, :], (K, L)).copy()
    return (torch.tensor(w_feats, dtype=torch.float32),
            torch.tensor(l_feats, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32))


def train_reward_net(w_states, l_states, w_actions, l_actions, env):
    model = RewardNet()
    optimizer = optim.Adam(model.parameters(), lr=NN_LR)
    w_feats, l_feats, mask = _precompute_nn_tensors(
        w_states, l_states, w_actions, l_actions, env)
    K = w_feats.shape[0]
    indices = np.arange(K)

    for epoch in range(NN_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, K, NN_BATCH):
            batch = indices[start:start + NN_BATCH]
            bw, bl, bm = w_feats[batch], l_feats[batch], mask[batch]
            B, L, D = bw.shape
            r_w = (model(bw.reshape(B*L, D)).reshape(B, L) * bm).sum(1)
            r_l = (model(bl.reshape(B*L, D)).reshape(B, L) * bm).sum(1)
            loss = -torch.nn.functional.logsigmoid(r_w - r_l).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def extract_reward_table(model, env):
    nS, nA = env.num_states, env.num_actions
    all_feats = STATE_FEATURES.reshape(nS * nA, NN_INPUT_DIM)
    model.eval()
    with torch.no_grad():
        rewards = model(torch.tensor(all_feats, dtype=torch.float32)).numpy()
    return rewards.reshape(nS, nA).mean(axis=1)


# =============================================================================
# Structural Models (VECTORIZED Bradley-Terry MLE)
# =============================================================================

def _precompute_structural_sums(w_states, l_states):
    """Precompute discounted sums using reward-relevant attributes.
    Searching states contribute log(benefit_wage) and z=0, not offer attributes."""
    K, L = w_states.shape
    gp = GAMMA_POWERS[:L]
    w_logw = (REWARD_LOGW[w_states] * gp[None, :]).sum(axis=1)
    w_z = (REWARD_Z[w_states] * gp[None, :]).sum(axis=1)
    l_logw = (REWARD_LOGW[l_states] * gp[None, :]).sum(axis=1)
    l_z = (REWARD_Z[l_states] * gp[None, :]).sum(axis=1)
    return w_logw, w_z, l_logw, l_z


def fit_structural(w_states, l_states):
    """Estimate alpha by BT MLE: u = alpha*log(w) + (1-alpha)*z."""
    w_logw, w_z, l_logw, l_z = _precompute_structural_sums(w_states, l_states)
    d_logw = w_logw - l_logw
    d_z = w_z - l_z

    def nll(alpha):
        diff = np.clip(alpha * d_logw + (1 - alpha) * d_z, -500, 500)
        return -np.sum(np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-15))

    result = minimize_scalar(nll, bounds=(0.01, 0.99), method='bounded')
    return result.x


def fit_misspecified(w_states, l_states):
    """Estimate alpha by BT MLE: u = alpha*log(w) + (1-alpha)*z_bar.
    Ignores amenity variation (treats all amenities as constant z_bar)."""
    K, L = w_states.shape
    gp = GAMMA_POWERS[:L]
    # Log-wage component (same as correct model)
    w_logw = (REWARD_LOGW[w_states] * gp[None, :]).sum(axis=1)
    l_logw = (REWARD_LOGW[l_states] * gp[None, :]).sum(axis=1)
    # Constant amenity: z_bar for employed, 0 for searching
    w_zbar = (IS_EMPLOYED[w_states] * ZBAR * gp[None, :]).sum(axis=1)
    l_zbar = (IS_EMPLOYED[l_states] * ZBAR * gp[None, :]).sum(axis=1)

    d_logw = w_logw - l_logw
    d_zbar = w_zbar - l_zbar

    def nll(alpha):
        diff = np.clip(alpha * d_logw + (1 - alpha) * d_zbar, -500, 500)
        return -np.sum(np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-15))

    result = minimize_scalar(nll, bounds=(0.01, 0.99), method='bounded')
    return result.x


def make_reward_vec(alpha):
    """Reward vector for correct structural model."""
    R = np.zeros(NUM_STATES)
    for si in range(NUM_STATES):
        emp, wi, zi = _decode_state(si)
        if emp:
            R[si] = alpha * np.log(WAGES[wi]) + (1 - alpha) * AMENITIES[zi]
        else:
            R[si] = alpha * np.log(BENEFIT_WAGE) + (1 - alpha) * 0.0
    return R


def make_wage_only_reward_vec(alpha):
    """Misspecified reward: alpha*log(w) + (1-alpha)*z_bar. Ignores amenity variation."""
    R = np.zeros(NUM_STATES)
    for si in range(NUM_STATES):
        emp, wi, zi = _decode_state(si)
        if emp:
            R[si] = alpha * np.log(WAGES[wi]) + (1 - alpha) * ZBAR
        else:
            R[si] = alpha * np.log(BENEFIT_WAGE)
    return R


# =============================================================================
# Online vs Offline
# =============================================================================

def run_nn_online(env, K, n_rounds=4):
    k_per = K // n_rounds
    all_ws, all_ls, all_wa, all_la = [], [], [], []
    policy = 'random'
    for _ in range(n_rounds):
        ws, ls, wa, la = generate_comparisons(env, k_per, policy=policy)
        all_ws.append(ws); all_ls.append(ls); all_wa.append(wa); all_la.append(la)
        cws = np.concatenate(all_ws); cls = np.concatenate(all_ls)
        cwa = np.concatenate(all_wa); cla = np.concatenate(all_la)
        model = train_reward_net(cws, cls, cwa, cla, env)
        R = extract_reward_table(model, env)
        _, policy, _ = value_iteration_vec(env, R)
    return policy


def run_nn_offline(env, K):
    ws, ls, wa, la = generate_comparisons(env, K)
    model = train_reward_net(ws, ls, wa, la, env)
    R = extract_reward_table(model, env)
    _, policy, _ = value_iteration_vec(env, R)
    return policy


# =============================================================================
# Diagnostics
# =============================================================================

def search_accept_mask(env, policy):
    """Which offers does the policy accept when searching?"""
    mask = np.zeros((env.W, env.Z), dtype=bool)
    for wi in range(env.W):
        for zi in range(env.Z):
            si = _searching_state(wi, zi)
            if policy[si] == 0:  # Accept
                mask[wi, zi] = True
    return mask


def employed_stay_mask(env, policy):
    """Which jobs does the policy stay at when employed?"""
    mask = np.zeros((env.W, env.Z), dtype=bool)
    for wi in range(env.W):
        for zi in range(env.Z):
            si = _employed_state(wi, zi)
            if policy[si] == 0:  # Stay
                mask[wi, zi] = True
    return mask


def mean_accepted_amenity(env, policy):
    """Mean amenity of offers the policy accepts (searching states)."""
    accept = search_accept_mask(env, policy)
    amenities = []
    for wi in range(env.W):
        for zi in range(env.Z):
            if accept[wi, zi]:
                amenities.append(AMENITIES[zi])
    return np.mean(amenities) if amenities else 0.0


def mean_accepted_wage(env, policy):
    """Mean wage of offers the policy accepts (searching states)."""
    accept = search_accept_mask(env, policy)
    wages = []
    for wi in range(env.W):
        for zi in range(env.Z):
            if accept[wi, zi]:
                wages.append(WAGES[wi])
    return np.mean(wages) if wages else 0.0


# =============================================================================
# Output: Environment Figure
# =============================================================================

def generate_env_figure(env, V_star, pi_star, pi_misspec=None):
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Panel (a): Accept/Reject boundary for searching states
    accept = search_accept_mask(env, pi_star)
    axes[0].imshow(accept.astype(float), cmap=CMAP_SEQ, aspect='auto',
                   origin='lower', vmin=0, vmax=1)
    for wi in range(env.W):
        for zi in range(env.Z):
            label = 'Acc' if accept[wi, zi] else 'Rej'
            color = 'black' if accept[wi, zi] else 'white'
            axes[0].text(zi, wi, label, ha='center', va='center',
                        fontsize=7, color=color)
    axes[0].set_xticks(range(env.Z))
    axes[0].set_xticklabels([f'{AMENITIES[j]:.1f}' for j in range(env.Z)], fontsize=8)
    axes[0].set_yticks(range(env.W))
    axes[0].set_yticklabels([f'{WAGES[i]:.0f}' for i in range(env.W)], fontsize=8)
    axes[0].set_xlabel('Amenity Level $z$')
    axes[0].set_ylabel('Wage $w$ (thousands)')
    axes[0].set_title('Optimal: Accept vs. Reject Offers')

    # Panel (b): Stay/Quit boundary for employed states
    stay = employed_stay_mask(env, pi_star)
    axes[1].imshow(stay.astype(float), cmap=CMAP_SEQ, aspect='auto',
                   origin='lower', vmin=0, vmax=1)
    for wi in range(env.W):
        for zi in range(env.Z):
            label = 'Stay' if stay[wi, zi] else 'Quit'
            color = 'black' if stay[wi, zi] else 'white'
            axes[1].text(zi, wi, label, ha='center', va='center',
                        fontsize=7, color=color)
    axes[1].set_xticks(range(env.Z))
    axes[1].set_xticklabels([f'{AMENITIES[j]:.1f}' for j in range(env.Z)], fontsize=8)
    axes[1].set_yticks(range(env.W))
    axes[1].set_yticklabels([f'{WAGES[i]:.0f}' for i in range(env.W)], fontsize=8)
    axes[1].set_xlabel('Amenity Level $z$')
    axes[1].set_ylabel('Wage $w$ (thousands)')
    axes[1].set_title('Optimal: Stay vs. Quit Jobs')

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'job_search_env.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fig_path}')


# =============================================================================
# Diagnostics Table
# =============================================================================

def generate_diagnostics(env, V_star, pi_star):
    K = 5000
    rng_seed = MASTER_SEED + 99999
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    ws, ls, wa, la = generate_comparisons(env, K)
    R_true = TRUE_REWARD_VEC.copy()
    results = []

    # NN RLHF
    model = train_reward_net(ws, ls, wa, la, env)
    R_nn = extract_reward_table(model, env)
    _, pi_nn, _ = value_iteration_vec(env, R_nn)
    V_nn = policy_eval_vec(env, pi_nn, R_true)
    results.append(('NN RLHF',
                     np.mean(pi_nn == pi_star) * 100,
                     np.corrcoef(V_nn, V_star)[0, 1],
                     mean_accepted_amenity(env, pi_nn),
                     mean_accepted_wage(env, pi_nn), None))

    # Correct structural
    alpha_hat = fit_structural(ws, ls)
    R_co = make_reward_vec(alpha_hat)
    _, pi_co, _ = value_iteration_vec(env, R_co)
    V_co = policy_eval_vec(env, pi_co, R_true)
    results.append(('Correct',
                     np.mean(pi_co == pi_star) * 100,
                     np.corrcoef(V_co, V_star)[0, 1],
                     mean_accepted_amenity(env, pi_co),
                     mean_accepted_wage(env, pi_co), alpha_hat))

    # Misspecified
    alpha_mis = fit_misspecified(ws, ls)
    R_mi = make_wage_only_reward_vec(alpha_mis)
    _, pi_mi, _ = value_iteration_vec(env, R_mi)
    V_mi = policy_eval_vec(env, pi_mi, R_true)
    results.append(('Misspecified',
                     np.mean(pi_mi == pi_star) * 100,
                     np.corrcoef(V_mi, V_star)[0, 1],
                     mean_accepted_amenity(env, pi_mi),
                     mean_accepted_wage(env, pi_mi), None))

    maz_opt = mean_accepted_amenity(env, pi_star)
    mwz_opt = mean_accepted_wage(env, pi_star)

    print('\n' + '=' * 80)
    print('DIAGNOSTICS (K=5000, single seed)')
    print('=' * 80)
    hdr = f'  {"Method":<15s} | {"Agree%":>7s} | {"V corr":>7s} | {"Mean z":>7s} | {"Mean w":>7s} | {"alpha":>7s}'
    print(hdr)
    print('-' * len(hdr))
    for name, agree, vcorr, maz, mwz, alpha_est in results:
        alpha_str = f'{alpha_est:.3f}' if alpha_est is not None else '    ---'
        print(f'  {name:<15s} | {agree:7.1f} | {vcorr:7.4f} | {maz:7.2f} | {mwz:7.1f} | {alpha_str:>7s}')
    print(f'  {"Optimal":<15s} | {100.0:7.1f} | {1.0000:7.4f} | {maz_opt:7.2f} | {mwz_opt:7.1f} |     ---')

    tex_path = os.path.join(OUTPUT_DIR, 'job_search_rlhf_diagnostics.tex')
    with open(tex_path, 'w') as f:
        f.write('\\begin{tabular}{lrrrrr}\n\\hline\n')
        f.write('Method & Policy agree.~(\\%) & $V^\\pi$ corr.'
                ' & Mean amenity & Mean wage & $\\hat{\\alpha}$ \\\\\n\\hline\n')
        for name, agree, vcorr, maz, mwz, alpha_est in results:
            alpha_str = f'${alpha_est:.3f}$' if alpha_est is not None else '---'
            f.write(f'{name} & ${agree:.1f}$ & ${vcorr:.3f}$ & ${maz:.2f}$ '
                    f'& ${mwz:.0f}$ & {alpha_str} \\\\\n')
        f.write('\\hline\n')
        f.write(f'Optimal & $100.0$ & $1.000$ & ${maz_opt:.2f}$ & ${mwz_opt:.0f}$ & --- \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex_path}')
    return results


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('=' * 80)
    print('RLHF SIMULATION STUDY: JOB SEARCH WITH COMPENSATING DIFFERENTIALS')
    print('=' * 80)
    print(f'\nTimestamp: {timestamp}')
    print(f'Python: {sys.version.split()[0]}, NumPy: {np.__version__}, PyTorch: {torch.__version__}')
    print(f'\nPARAMETERS:')
    print(f'  Wage levels: {W}, Amenity levels: {Z}')
    print(f'  States: {NUM_STATES} ({NUM_WZ} searching + {NUM_WZ} employed)')
    print(f'  True alpha: {ALPHA_TRUE}, gamma: {GAMMA}')
    print(f'  Wages: {WAGES}')
    print(f'  Amenities: {AMENITIES}')
    print(f'  Benefit wage: {BENEFIT_WAGE}')
    print(f'  Layoff prob: {P_LAYOFF}, z_bar (misspec): {ZBAR:.1f}')
    print(f'  Segment length: {SEGMENT_LENGTH}, K values: {COMPARISON_COUNTS}')
    print(f'  Seeds: {N_SEEDS} (main), {N_SEEDS_ABLATION} (ablation)')
    print(f'  NN: {NN_INPUT_DIM}->{NN_HIDDEN}->{NN_HIDDEN}->1, lr={NN_LR}, '
          f'epochs={NN_EPOCHS}, batch={NN_BATCH}')

    env = JobSearchEnv()

    # Print offer distribution correlation
    print(f'\nOFFER DISTRIBUTION:')
    wi_vals, zi_vals = [], []
    for wi in range(env.W):
        for zi in range(env.Z):
            count = int(env.offer_probs[wi, zi] * 10000)
            wi_vals.extend([wi] * count)
            zi_vals.extend([zi] * count)
    corr = np.corrcoef(wi_vals, zi_vals)[0, 1]
    print(f'  Correlation between wage rank and amenity rank: {corr:.3f}')
    print(f'  Offer distribution (probabilities):')
    print(f'  {"":>6s}', end='')
    for zi in range(env.Z):
        print(f'  z={AMENITIES[zi]:.1f}', end='')
    print()
    for wi in range(env.W):
        print(f'  w={WAGES[wi]:3.0f}', end='')
        for zi in range(env.Z):
            print(f'  {env.offer_probs[wi, zi]:.4f}', end='')
        print()

    # --- 1. DP Oracle ---
    print('\n' + '=' * 80)
    print('1. DP ORACLE')
    print('=' * 80)
    R_true = TRUE_REWARD_VEC.copy()
    V_star, pi_star, vi_iters = value_iteration_vec(env, R_true)
    np.random.seed(MASTER_SEED)
    dp_ret, dp_se = evaluate_policy_mc(env, pi_star)
    print(f'  Converged in {vi_iters} iterations')
    print(f'  V*(mean searching): {np.mean(V_star[:NUM_WZ]):.4f}')
    print(f'  V*(mean employed): {np.mean(V_star[NUM_WZ:]):.4f}')
    print(f'  MC return: {dp_ret:.3f} +/- {dp_se:.3f}')

    accept = search_accept_mask(env, pi_star)
    stay = employed_stay_mask(env, pi_star)
    n_accept = accept.sum()
    n_stay = stay.sum()
    maz_opt = mean_accepted_amenity(env, pi_star)
    mwz_opt = mean_accepted_wage(env, pi_star)
    print(f'  Accepts {n_accept} of {NUM_WZ} offer types')
    print(f'  Stays at {n_stay} of {NUM_WZ} job types')
    print(f'  Mean accepted amenity: {maz_opt:.2f}, Mean accepted wage: {mwz_opt:.0f}')

    generate_env_figure(env, V_star, pi_star)

    # Use a representative searching state for V(s0)
    # Average over all searching states weighted by offer probs
    V_s0 = np.dot(env._offer_flat, V_star[:NUM_WZ])
    dp_v0 = V_s0

    # --- 2. Q-Learning ---
    print('\n' + '=' * 80)
    print('2. Q-LEARNING')
    print('=' * 80)
    ql_returns, ql_steps_list = [], []
    for seed in range(N_SEEDS):
        np.random.seed(MASTER_SEED + seed)
        pol, steps = q_learning(env)
        V_ql = policy_eval_vec(env, pol, R_true)
        ql_val = np.dot(env._offer_flat, V_ql[:NUM_WZ])
        ql_returns.append(ql_val)
        ql_steps_list.append(steps)
        if seed < 3 or seed == N_SEEDS - 1:
            print(f'  Seed {seed:2d}: queries={steps}, V(s0)={ql_val:.3f}')
            if seed == 2 and N_SEEDS > 4:
                print('  ...')
    ql_mean = np.mean(ql_returns)
    ql_se = np.std(ql_returns) / np.sqrt(N_SEEDS)
    print(f'\n  Summary: V(s0)={ql_mean:.3f} +/- {ql_se:.3f}, '
          f'queries={int(np.mean(ql_steps_list))}')

    # --- 3-5. Preference methods ---
    nn_res = {K: [] for K in COMPARISON_COUNTS}
    co_res = {K: [] for K in COMPARISON_COUNTS}
    mi_res = {K: [] for K in COMPARISON_COUNTS}
    co_alphas = {K: [] for K in COMPARISON_COUNTS}
    mi_alphas = {K: [] for K in COMPARISON_COUNTS}
    mi_amenities = {K: [] for K in COMPARISON_COUNTS}

    for K in COMPARISON_COUNTS:
        print(f'\n{"=" * 80}')
        print(f'K = {K}')
        print('=' * 80)

        for seed in range(N_SEEDS):
            rng_seed = MASTER_SEED + seed * 10000 + K
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)

            ws, ls, wa, la = generate_comparisons(env, K)

            # Neural net RLHF
            model = train_reward_net(ws, ls, wa, la, env)
            R_nn = extract_reward_table(model, env)
            _, nn_pol, _ = value_iteration_vec(env, R_nn)
            V_nn = policy_eval_vec(env, nn_pol, R_true)
            nn_val = np.dot(env._offer_flat, V_nn[:NUM_WZ])
            nn_res[K].append(nn_val)

            # Correct structural
            alpha_hat = fit_structural(ws, ls)
            R_co = make_reward_vec(alpha_hat)
            _, pi_co, _ = value_iteration_vec(env, R_co)
            V_co = policy_eval_vec(env, pi_co, R_true)
            co_val = np.dot(env._offer_flat, V_co[:NUM_WZ])
            co_res[K].append(co_val)
            co_alphas[K].append(alpha_hat)

            # Misspecified structural (constant amenity z_bar)
            alpha_mis = fit_misspecified(ws, ls)
            R_mi = make_wage_only_reward_vec(alpha_mis)
            _, pi_mi, _ = value_iteration_vec(env, R_mi)
            V_mi = policy_eval_vec(env, pi_mi, R_true)
            mi_val = np.dot(env._offer_flat, V_mi[:NUM_WZ])
            mi_res[K].append(mi_val)
            mi_alphas[K].append(alpha_mis)
            mi_amenities[K].append(mean_accepted_amenity(env, pi_mi))

        nn_m = np.mean(nn_res[K]); nn_s = np.std(nn_res[K]) / np.sqrt(N_SEEDS)
        co_m = np.mean(co_res[K]); co_s = np.std(co_res[K]) / np.sqrt(N_SEEDS)
        mi_m = np.mean(mi_res[K]); mi_s = np.std(mi_res[K]) / np.sqrt(N_SEEDS)
        print(f'  NN RLHF:     {nn_m:7.3f} +/- {nn_s:.3f}')
        print(f'  Correct:     {co_m:7.3f} +/- {co_s:.3f}  (alpha={np.mean(co_alphas[K]):.3f})')
        print(f'  Misspecified:{mi_m:7.3f} +/- {mi_s:.3f}  '
              f'(mean amenity={np.mean(mi_amenities[K]):.2f})')

    # --- Ablation ---
    print('\n' + '=' * 80)
    print('ABLATION: ONLINE VS OFFLINE (NN RLHF, K=1000)')
    print('=' * 80)
    K_abl = 1000
    on_rets, off_rets = [], []
    for seed in range(N_SEEDS_ABLATION):
        np.random.seed(MASTER_SEED + 5000 + seed)
        torch.manual_seed(MASTER_SEED + 5000 + seed)
        pol_on = run_nn_online(env, K_abl)
        V_on = policy_eval_vec(env, pol_on, R_true)
        on_val = np.dot(env._offer_flat, V_on[:NUM_WZ])
        on_rets.append(on_val)

        np.random.seed(MASTER_SEED + 6000 + seed)
        torch.manual_seed(MASTER_SEED + 6000 + seed)
        pol_off = run_nn_offline(env, K_abl)
        V_off = policy_eval_vec(env, pol_off, R_true)
        off_val = np.dot(env._offer_flat, V_off[:NUM_WZ])
        off_rets.append(off_val)

        if seed < 3 or seed == N_SEEDS_ABLATION - 1:
            print(f'  Seed {seed:2d}: online={on_rets[-1]:.3f}, '
                  f'offline={off_rets[-1]:.3f}')
            if seed == 2 and N_SEEDS_ABLATION > 4:
                print('  ...')

    on_mean = np.mean(on_rets); on_se = np.std(on_rets) / np.sqrt(N_SEEDS_ABLATION)
    off_mean = np.mean(off_rets); off_se = np.std(off_rets) / np.sqrt(N_SEEDS_ABLATION)
    _, p_val = stats.ttest_ind(off_rets, on_rets, alternative='greater')
    print(f'\n  Online:  {on_mean:.3f} +/- {on_se:.3f}')
    print(f'  Offline: {off_mean:.3f} +/- {off_se:.3f}')
    print(f'  p-value (offline > online): {p_val:.4f}')

    # --- Summary ---
    print('\n' + '=' * 80)
    print('MAIN RESULTS TABLE')
    print('=' * 80)
    hdr = f'{"Method":<28s} | {"K":>5s} | {"V(s0)":>10s} | {"% DP":>6s}'
    print(hdr)
    print('-' * len(hdr))
    print(f'{"DP Oracle":<28s} | {"--":>5s} | {dp_v0:10.3f} | {100.0:6.1f}')
    print(f'{"Q-Learning":<28s} | {"--":>5s} | {ql_mean:10.3f} | '
          f'{100*ql_mean/dp_v0:6.1f}')
    print('-' * len(hdr))
    for K in COMPARISON_COUNTS:
        for name, res in [('NN RLHF', nn_res), ('Correct Structural', co_res),
                          ('Misspecified', mi_res)]:
            m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
            print(f'{name:<28s} | {K:5d} | {m:10.3f} | {100*m/dp_v0:6.1f}')
        if K != COMPARISON_COUNTS[-1]:
            print('-' * len(hdr))

    # --- Verification ---
    print('\n' + '=' * 80)
    print('VERIFICATION')
    print('=' * 80)
    print(f'  1. DP V(s0) positive: {dp_v0:.3f} -> '
          f'{"PASS" if dp_v0 > 0 else "FAIL"}')
    print(f'  2. Q-learning near DP: {100*ql_mean/dp_v0:.1f}% -> '
          f'{"PASS" if ql_mean/dp_v0 > 0.85 else "FAIL"}')
    nn_hi = np.mean(nn_res[COMPARISON_COUNTS[-1]])
    print(f'  3. NN at K={COMPARISON_COUNTS[-1]}: {100*nn_hi/dp_v0:.1f}% -> '
          f'{"PASS" if nn_hi/dp_v0 > 0.85 else "FAIL"}')
    co_hi = np.mean(co_res[COMPARISON_COUNTS[-1]])
    print(f'  4. Correct at K={COMPARISON_COUNTS[-1]}: {100*co_hi/dp_v0:.1f}% -> '
          f'{"PASS" if co_hi/dp_v0 > 0.85 else "FAIL"}')
    mi_hi = np.mean(mi_res[COMPARISON_COUNTS[-1]])
    mi_az_hi = np.mean(mi_amenities[COMPARISON_COUNTS[-1]])
    print(f'  5. Misspec mean amenity < optimal: {mi_az_hi:.2f} vs {maz_opt:.2f} -> '
          f'{"PASS" if mi_az_hi < maz_opt else "CHECK"}')
    nn_lo = np.mean(nn_res[COMPARISON_COUNTS[0]])
    print(f'  6. NN at K={COMPARISON_COUNTS[0]} < 95% DP: {100*nn_lo/dp_v0:.1f}% -> '
          f'{"PASS" if nn_lo/dp_v0 < 0.95 else "CHECK (too easy)"}')
    mi_var = np.std(mi_res[COMPARISON_COUNTS[-1]])
    print(f'  7. Misspec has nonzero variance: std={mi_var:.4f} -> '
          f'{"PASS" if mi_var > 0.001 else "CHECK (zero variance)"}')
    print(f'  8. Offer correlation < -0.3: {corr:.3f} -> '
          f'{"PASS" if corr < -0.3 else "CHECK"}')

    # --- Parameter Recovery ---
    print('\n' + '=' * 80)
    print('PARAMETER RECOVERY')
    print('=' * 80)
    print(f'  True alpha = {ALPHA_TRUE}')
    print(f'     K | {"alpha_hat":>10s} | {"alpha_mis":>10s}')
    print('-' * 35)
    for K in COMPARISON_COUNTS:
        print(f'{K:6d} | {np.mean(co_alphas[K]):>10.4f} | '
              f'{np.mean(mi_alphas[K]):>10.4f}')

    print('\n' + '=' * 80)
    print('MISSPECIFICATION ANALYSIS: MEAN ACCEPTED AMENITY')
    print('=' * 80)
    print(f'  Optimal mean amenity: {maz_opt:.2f}')
    for K in COMPARISON_COUNTS:
        maz = np.mean(mi_amenities[K])
        mm = np.mean(mi_res[K])
        print(f'  K={K:5d}: mean amenity={maz:.2f}, V(s0)={mm:.3f} '
              f'({100*mm/dp_v0:.1f}% DP)')

    # --- Diagnostics ---
    generate_diagnostics(env, V_star, pi_star)

    # --- Sample Complexity Figure ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    K_arr = np.array(COMPARISON_COUNTS)
    ax.axhline(dp_v0, color=COLORS['black'], ls='--', label='DP Oracle', alpha=0.8)
    ax.axhline(ql_mean, color=COLORS['blue'], ls=':', label='Q-Learning', alpha=0.8)
    for name, res, color, marker in [
        ('RLHF (Neural Net)', nn_res, COLORS['green'], 'o'),
        ('RLHF (Correct)', co_res, COLORS['purple'], 's'),
        ('RLHF (Misspecified)', mi_res, COLORS['red'], '^'),
    ]:
        means = [np.mean(res[K]) for K in COMPARISON_COUNTS]
        ses = [np.std(res[K]) / np.sqrt(N_SEEDS) for K in COMPARISON_COUNTS]
        ax.errorbar(K_arr, means, yerr=ses, marker=marker, color=color,
                     label=name, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Preference Comparisons ($K$)')
    ax.set_ylabel('Policy Value $V^\\pi(s_0)$')
    ax.set_title('Sample Complexity: Job Search Preference Learning')
    ax.legend(loc='lower right', fontsize=8)
    fig_path = os.path.join(OUTPUT_DIR, 'job_search_sample_complexity.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {fig_path}')

    # --- LaTeX Tables ---
    tex1 = os.path.join(OUTPUT_DIR, 'job_search_rlhf_results.tex')
    with open(tex1, 'w') as f:
        f.write('\\begin{tabular}{llrr}\n\\hline\n')
        f.write('Method & $K$ & $V^\\pi(s_0)$ & \\% of DP \\\\\n\\hline\n')
        f.write(f'DP Oracle & --- & ${dp_v0:.2f}$ & $100.0$ \\\\\n')
        f.write(f'Q-Learning & --- & ${ql_mean:.2f} \\pm {ql_se:.2f}$ & '
                f'${100*ql_mean/dp_v0:.1f}$ \\\\\n\\hline\n')
        for K in COMPARISON_COUNTS:
            for name, res in [('NN RLHF', nn_res), ('Correct', co_res),
                              ('Misspecified', mi_res)]:
                m = np.mean(res[K]); s = np.std(res[K]) / np.sqrt(N_SEEDS)
                f.write(f'{name} & ${K}$ & ${m:.2f} \\pm {s:.2f}$ & '
                        f'${100*m/dp_v0:.1f}$ \\\\\n')
            if K != COMPARISON_COUNTS[-1]:
                f.write('\\hline\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex1}')

    tex2 = os.path.join(OUTPUT_DIR, 'job_search_online_offline.tex')
    with open(tex2, 'w') as f:
        f.write('\\begin{tabular}{lrr}\n\\hline\n')
        f.write('Collection & $V^\\pi(s_0)$ & $p$-value \\\\\n\\hline\n')
        f.write(f'Online (4 rounds) & ${on_mean:.2f} \\pm {on_se:.2f}$ & \\\\\n')
        f.write(f'Offline & ${off_mean:.2f} \\pm {off_se:.2f}$ & '
                f'${p_val:.4f}$ \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f'Saved: {tex2}')

    print(f'\nOUTPUT FILES:')
    for p in [os.path.join(OUTPUT_DIR, 'job_search_env.png'),
              fig_path, tex1, tex2,
              os.path.join(OUTPUT_DIR, 'job_search_rlhf_diagnostics.tex')]:
        print(f'  {p}')


if __name__ == '__main__':
    run_experiment()
