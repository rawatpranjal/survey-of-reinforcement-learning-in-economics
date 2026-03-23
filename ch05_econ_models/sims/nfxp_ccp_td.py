"""Structural estimation comparison: NFXP vs CCP vs TD methods on multi-component bus engine.
Chapter 5 — Solving Economic Models with RL.
Compares four estimation methods at increasing state-space scale (K=1..4 components).
"""

import argparse
import itertools
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
apply_style()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'version': 14,
    'M': 20,
    'gamma': 0.95,
    'trans_probs': [0.4, 0.4, 0.2],
    'true_RC': 5.0,
    'true_theta1': 2.0,
    'true_theta2': 4.0,
    'N_agents': 500,
    'T_periods': 100,
    'exp2_seeds': 5,
    'exp2_K_values': [1, 2, 3, 4],
    'timeout': 300,
    'vi_tol': 1e-8,
    'vi_max_iter': 5000,
    'nn_hidden': 64,
    'nn_avi_iters': 20,
    'nn_epochs_per_iter': 30,
    'nn_lr': 1e-3,
    'poly_degree': 2,
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_DIR = os.path.dirname(__file__)

METHOD_COLORS = {
    'NFXP':           COLORS['red'],
    'CCP':            COLORS['orange'],
    'TD-CCP Linear':  COLORS['blue'],
    'TD-CCP Neural':  COLORS['green'],
}
METHOD_ORDER = ['NFXP', 'CCP', 'TD-CCP Linear', 'TD-CCP Neural']


# ============================================================================
# Multi-Component Bus Engine (sparse transition matrices)
# ============================================================================

class MultiComponentBusEngine:
    """Rust (1987) bus engine with K independent wear components."""

    def __init__(self, K, M, RC, theta1, theta2, gamma, trans_probs):
        self.K = K
        self.M = M
        self.RC = RC
        self.theta1 = theta1
        self.theta2 = theta2
        self.gamma = gamma
        self.trans_probs = np.array(trans_probs)
        self.n_increments = len(trans_probs)

        # State space: K-tuples of ints in [0, M-1]
        # Represent states as flat indices using mixed-radix encoding
        self.n_states = M ** K
        # x(s) = sum_k(m_k / M) for each state
        self.x_vec = self._compute_x_vec()

    def _state_to_idx(self, state_tuple):
        """Convert K-tuple to flat index (big-endian mixed radix)."""
        idx = 0
        for k in range(self.K):
            idx = idx * self.M + state_tuple[k]
        return idx

    def _idx_to_state(self, idx):
        """Convert flat index to K-tuple."""
        state = []
        for k in range(self.K - 1, -1, -1):
            state.append(idx % self.M)
            idx //= self.M
        return tuple(reversed(state))

    def _compute_x_vec(self):
        """Compute x(s) = sum_k(m_k/M) for all states."""
        x = np.zeros(self.n_states)
        for i in range(self.n_states):
            s = self._idx_to_state(i)
            x[i] = sum(m / self.M for m in s)
        return x

    def build_transition_matrix(self):
        """Build sparse |S| x |S| transition matrix for action=keep."""
        rows, cols, vals = [], [], []
        inc_combos = list(itertools.product(range(1, self.n_increments + 1), repeat=self.K))
        # Precompute probabilities for each combo
        combo_probs = []
        for deltas in inc_combos:
            prob = 1.0
            for k in range(self.K):
                prob *= self.trans_probs[deltas[k] - 1]
            combo_probs.append(prob)

        for i in range(self.n_states):
            s = self._idx_to_state(i)
            for ci, deltas in enumerate(inc_combos):
                new_s = tuple(min(s[k] + deltas[k], self.M - 1) for k in range(self.K))
                j = self._state_to_idx(new_s)
                rows.append(i)
                cols.append(j)
                vals.append(combo_probs[ci])

        P = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_states, self.n_states))
        # Some (i,j) pairs may appear multiple times (when capping merges targets);
        # csr_matrix sums duplicates automatically.
        return P

    def solve_vi(self, RC=None, theta1=None, theta2=None, tol=None, max_iter=None):
        """Value iteration. Returns (EV, v0, v1_vec, P_keep_sparse)."""
        RC = RC if RC is not None else self.RC
        theta1 = theta1 if theta1 is not None else self.theta1
        theta2 = theta2 if theta2 is not None else self.theta2
        tol = tol or CONFIG['vi_tol']
        max_iter = max_iter or CONFIG['vi_max_iter']

        cost = theta1 * self.x_vec + theta2 * self.x_vec ** 2
        P_keep = self.build_transition_matrix()
        idx_zero = 0  # state (0,...,0) is always index 0

        EV = np.zeros(self.n_states)
        for _ in range(max_iter):
            v0 = -cost + self.gamma * P_keep.dot(EV)
            v1_val = -RC + self.gamma * EV[idx_zero]
            EV_new = np.logaddexp(v0, v1_val)
            if np.max(np.abs(EV_new - EV)) < tol:
                EV = EV_new
                break
            EV = EV_new

        v0 = -cost + self.gamma * P_keep.dot(EV)
        v1_vec = np.full(self.n_states, -RC + self.gamma * EV[idx_zero])
        return EV, v0, v1_vec, P_keep

    def simulate_panel(self, v0, v1_vec, N, T, seed=42):
        """Simulate panel data. Returns (states[N,T], actions[N,T]) as flat indices."""
        rng = np.random.RandomState(seed)
        p_replace = 1.0 / (1.0 + np.exp(-(v1_vec - v0)))

        P_keep = self.build_transition_matrix()
        # Convert to dense CDF row-by-row for simulation (only practical for moderate n_states)
        # For large state spaces, use per-component simulation instead.
        if self.n_states <= 10000:
            P_dense = P_keep.toarray()
            P_cdf = np.cumsum(P_dense, axis=1)
        else:
            P_cdf = None  # use component-wise simulation

        idx_zero = 0
        states = np.zeros((N, T), dtype=np.int32)
        actions = np.zeros((N, T), dtype=np.int32)
        states[:, 0] = rng.randint(0, self.n_states, size=N)

        for t in range(T):
            s = states[:, t]
            u = rng.random(N)
            a = (u < p_replace[s]).astype(np.int32)
            actions[:, t] = a

            if t < T - 1:
                next_s = np.empty(N, dtype=np.int32)
                keep_mask = (a == 0)
                replace_mask = (a == 1)

                if keep_mask.any():
                    keep_idx = np.where(keep_mask)[0]
                    if P_cdf is not None:
                        u2 = rng.random(len(keep_idx))
                        for ii, agent_idx in enumerate(keep_idx):
                            next_s[agent_idx] = np.searchsorted(P_cdf[s[agent_idx]], u2[ii])
                    else:
                        # Component-wise simulation for large state spaces
                        for agent_idx in keep_idx:
                            curr = self._idx_to_state(s[agent_idx])
                            new_s = []
                            for k in range(self.K):
                                delta = rng.choice(self.n_increments, p=self.trans_probs) + 1
                                new_s.append(min(curr[k] + delta, self.M - 1))
                            next_s[agent_idx] = self._state_to_idx(tuple(new_s))

                next_s[replace_mask] = idx_zero
                states[:, t + 1] = next_s

        return states, actions


# ============================================================================
# Shared utilities
# ============================================================================

def _precompute_env(K, M, gamma, trans_probs):
    """Precompute environment structures (sparse transition matrix, etc.)."""
    env = MultiComponentBusEngine(K, M, 1.0, 1.0, 1.0, gamma, trans_probs)
    P_keep = env.build_transition_matrix()
    return {
        'n_states': env.n_states,
        'x_vec': env.x_vec,
        'P_keep': P_keep,
        'idx_zero': 0,
    }


def _estimate_ccps(s_flat, a_flat, n_states):
    """Estimate CCPs from data frequencies."""
    count_s = np.bincount(s_flat, minlength=n_states).astype(float)
    count_sa1 = np.bincount(s_flat[a_flat == 1], minlength=n_states).astype(float)
    p1_hat = np.clip(count_sa1 / np.maximum(count_s, 1), 0.01, 0.99)
    return p1_hat, count_s


def _vi_solve(cost, RC, gamma, P_keep, idx_zero, n_states):
    """Inner VI loop with sparse P_keep. Returns (v0, v1_vec)."""
    EV = np.zeros(n_states)
    for _ in range(CONFIG['vi_max_iter']):
        v0 = -cost + gamma * P_keep.dot(EV)
        v1_val = -RC + gamma * EV[idx_zero]
        EV_new = np.logaddexp(v0, v1_val)
        if np.max(np.abs(EV_new - EV)) < CONFIG['vi_tol']:
            EV = EV_new
            break
        EV = EV_new
    v0 = -cost + gamma * P_keep.dot(EV)
    v1_vec = np.full(n_states, -RC + gamma * EV[idx_zero])
    return v0, v1_vec


def _neg_loglik_from_vdiff(v_diff, s_flat, a_flat):
    """Negative log-likelihood from value differences v0 - v1."""
    vd = v_diff[s_flat]
    log_p0 = -np.logaddexp(0, -vd)
    log_p1 = -np.logaddexp(0, vd)
    return -np.where(a_flat == 0, log_p0, log_p1).sum()


# ============================================================================
# Estimators
# ============================================================================

def estimate_nfxp(states, actions, env_config, timeout=300, precomp=None):
    """NFXP: nested fixed-point MLE (Rust 1987)."""
    t0 = time.time()
    K, M, gamma = env_config['K'], env_config['M'], env_config['gamma']
    if precomp is None:
        precomp = _precompute_env(K, M, gamma, env_config['trans_probs'])
    n_states = precomp['n_states']
    x_vec, P_keep, idx_zero = precomp['x_vec'], precomp['P_keep'], precomp['idx_zero']
    s_flat, a_flat = states.ravel(), actions.ravel()

    class Timeout(Exception):
        pass

    def neg_ll(theta_vec):
        if time.time() - t0 > timeout:
            raise Timeout()
        RC, th1, th2 = theta_vec
        cost = th1 * x_vec + th2 * x_vec ** 2
        v0, v1_vec = _vi_solve(cost, RC, gamma, P_keep, idx_zero, n_states)
        return _neg_loglik_from_vdiff(v0 - v1_vec, s_flat, a_flat)

    try:
        res = minimize(neg_ll, [4.0, 1.5, 3.0], method='L-BFGS-B',
                       bounds=[(0.1, 20.0)]*3, options={'maxiter': 200, 'ftol': 1e-8})
        return {'converged': res.success, 'theta': res.x.tolist(), 'time': time.time()-t0}
    except Timeout:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0, 'timeout': True}
    except Exception:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0}


def estimate_ccp(states, actions, env_config, precomp=None):
    """CCP / Hotz-Miller: invert CCPs to recover EV, then PMLE."""
    t0 = time.time()
    K, M, gamma = env_config['K'], env_config['M'], env_config['gamma']
    if precomp is None:
        precomp = _precompute_env(K, M, gamma, env_config['trans_probs'])
    n_states = precomp['n_states']
    x_vec, P_keep, idx_zero = precomp['x_vec'], precomp['P_keep'], precomp['idx_zero']
    s_flat, a_flat = states.ravel(), actions.ravel()
    p1_hat, count_s = _estimate_ccps(s_flat, a_flat, n_states)

    coverage = (count_s >= 5).sum() / n_states
    if coverage < 0.1:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0,
                'reason': 'sparse', 'coverage': coverage}

    p0_hat = 1 - p1_hat
    H = -p0_hat * np.log(p0_hat) - p1_hat * np.log(p1_hat)

    # Conditional transition F = diag(p0) @ P_keep + p1 * e_0^T
    # F is sparse: same sparsity as P_keep plus one dense column (idx_zero).
    # Build (I - gamma*F) as sparse and solve iteratively or factor.
    # For n_states <= 10000, use direct solve. For larger, use iterative.
    # (I - gamma*F) @ EV = r_bar + H  where r_bar depends on theta.

    # Precompute (I - gamma*F) factorization once.
    # F = diag(p0) @ P_keep + outer(p1, e_{idx_zero})
    # I - gamma*F = I - gamma*diag(p0)@P_keep - gamma*outer(p1, e_{idx_zero})
    IgF = sparse.eye(n_states) - gamma * sparse.diags(p0_hat).dot(P_keep)
    # Subtract the rank-1 correction: gamma * p1 @ e_0^T
    # This adds -gamma*p1[i] to column idx_zero for each row i.
    e0_col = sparse.csc_matrix((gamma * p1_hat, (np.arange(n_states),
                                np.zeros(n_states, dtype=int))),
                               shape=(n_states, n_states))
    IgF = IgF - e0_col
    IgF = IgF.tocsc()

    if n_states <= 10000:
        # Factorize once, reuse for each theta eval
        from scipy.sparse.linalg import splu
        IgF_lu = splu(IgF)
        def solve_ev(rhs):
            return IgF_lu.solve(rhs)
    else:
        def solve_ev(rhs):
            return spsolve(IgF, rhs)

    def neg_ll_ccp(theta_vec):
        RC, th1, th2 = theta_vec
        cost = th1 * x_vec + th2 * x_vec ** 2
        r_no_ent = p0_hat * (-cost) + p1_hat * (-RC)
        EV = solve_ev(r_no_ent + H)
        v0 = -cost + gamma * P_keep.dot(EV)
        v1 = -RC + gamma * EV[idx_zero]
        return _neg_loglik_from_vdiff(v0 - v1, s_flat, a_flat)

    try:
        res = minimize(neg_ll_ccp, [4.0, 1.5, 3.0], method='L-BFGS-B',
                       bounds=[(0.1, 20.0)]*3, options={'maxiter': 200})
        return {'converged': res.success, 'theta': res.x.tolist(),
                'time': time.time()-t0, 'coverage': coverage}
    except Exception:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0}


def _polynomial_features(n_states, M, K, x_vec, degree=2):
    """Polynomial basis features. Returns (n_states, n_feat) array."""
    feats = [np.ones(n_states)]
    feats.append(x_vec)
    if degree >= 2:
        feats.append(x_vec ** 2)
    # Per-component features (only useful for K > 1)
    if K > 1:
        # Decode per-component values
        comp_vals = np.zeros((n_states, K))
        for i in range(n_states):
            idx = i
            for k in range(K - 1, -1, -1):
                comp_vals[i, k] = (idx % M) / M
                idx //= M
        for k in range(K):
            feats.append(comp_vals[:, k])
            if degree >= 2:
                feats.append(comp_vals[:, k] ** 2)
    return np.column_stack(feats)


def estimate_td_ccp_linear(states, actions, env_config, precomp=None):
    """TD-CCP with linear basis: decompose EV into theta-linear components, solve via TD."""
    t0 = time.time()
    K, M, gamma = env_config['K'], env_config['M'], env_config['gamma']
    if precomp is None:
        precomp = _precompute_env(K, M, gamma, env_config['trans_probs'])
    n_states = precomp['n_states']
    x_vec, P_keep, idx_zero = precomp['x_vec'], precomp['P_keep'], precomp['idx_zero']

    N_ag, T = states.shape
    s_flat, a_flat = states.ravel(), actions.ravel()
    p1_hat, _ = _estimate_ccps(s_flat, a_flat, n_states)
    p0_hat = 1 - p1_hat
    H = -p0_hat * np.log(p0_hat) - p1_hat * np.log(p1_hat)

    phi_all = _polynomial_features(n_states, M, K, x_vec, CONFIG['poly_degree'])
    n_feat = phi_all.shape[1]

    # EV(s; theta) = theta1*g1(s) + theta2*g2(s) + RC*g_rc(s) + g_H(s)
    # where each g_k satisfies: g_k = flow_k + gamma * F @ g_k
    # Approximate g_k ~ phi @ w_k via semi-gradient TD (projected Bellman).
    flow_1 = -p0_hat * x_vec
    flow_2 = -p0_hat * x_vec ** 2
    flow_rc = -p1_hat
    flow_H = H

    # Build A and b from panel transitions
    s_t_arr = states[:, :-1].ravel()
    s_t1_arr = states[:, 1:].ravel()
    phi_t = phi_all[s_t_arr]
    phi_t1 = phi_all[s_t1_arr]
    n_trans = len(s_t_arr)

    A_mat = (phi_t.T @ (phi_t - gamma * phi_t1)) / n_trans

    def solve_w(flow_vec):
        b = (phi_t.T @ flow_vec[s_t_arr]) / n_trans
        return np.linalg.lstsq(A_mat, b, rcond=None)[0]

    try:
        w_1 = solve_w(flow_1)
        w_2 = solve_w(flow_2)
        w_rc = solve_w(flow_rc)
        w_H = solve_w(flow_H)
    except np.linalg.LinAlgError:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0, 'reason': 'singular'}

    ev_1 = phi_all @ w_1
    ev_2 = phi_all @ w_2
    ev_rc = phi_all @ w_rc
    ev_H = phi_all @ w_H

    Pev_1 = P_keep.dot(ev_1)
    Pev_2 = P_keep.dot(ev_2)
    Pev_rc = P_keep.dot(ev_rc)
    Pev_H = P_keep.dot(ev_H)

    def neg_ll_tdccp(theta_vec):
        RC, th1, th2 = theta_vec
        cost = th1 * x_vec + th2 * x_vec ** 2
        v0 = -cost + gamma * (th1 * Pev_1 + th2 * Pev_2 + RC * Pev_rc + Pev_H)
        ev_zero = th1*ev_1[idx_zero] + th2*ev_2[idx_zero] + RC*ev_rc[idx_zero] + ev_H[idx_zero]
        v1 = -RC + gamma * ev_zero
        return _neg_loglik_from_vdiff(v0 - v1, s_flat, a_flat)

    try:
        res = minimize(neg_ll_tdccp, [4.0, 1.5, 3.0], method='L-BFGS-B',
                       bounds=[(0.1, 20.0)]*3, options={'maxiter': 200})
        return {'converged': res.success, 'theta': res.x.tolist(), 'time': time.time()-t0}
    except Exception:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0}


def estimate_td_ccp_nn(states, actions, env_config, precomp=None):
    """TD-CCP with neural net: learn EV components via AVI with MLP, then PMLE."""
    if not HAS_TORCH:
        return {'converged': False, 'theta': [np.nan]*3, 'time': 0, 'reason': 'no_torch'}

    t0 = time.time()
    K, M, gamma = env_config['K'], env_config['M'], env_config['gamma']
    if precomp is None:
        precomp = _precompute_env(K, M, gamma, env_config['trans_probs'])
    n_states = precomp['n_states']
    x_vec, P_keep, idx_zero = precomp['x_vec'], precomp['P_keep'], precomp['idx_zero']

    N_ag, T = states.shape
    s_flat, a_flat = states.ravel(), actions.ravel()
    p1_hat, _ = _estimate_ccps(s_flat, a_flat, n_states)
    p0_hat = 1 - p1_hat
    H = -p0_hat * np.log(p0_hat) - p1_hat * np.log(p1_hat)

    # State features for NN input: per-component normalized values
    raw_feats = np.zeros((n_states, K), dtype=np.float32)
    for i in range(n_states):
        idx = i
        for k in range(K - 1, -1, -1):
            raw_feats[i, k] = (idx % M) / M
            idx //= M

    flow_funcs = {
        '1': (-p0_hat * x_vec).astype(np.float32),
        '2': (-p0_hat * x_vec ** 2).astype(np.float32),
        'rc': (-p1_hat).astype(np.float32),
        'H': H.astype(np.float32),
    }

    s_t_arr = states[:, :-1].ravel()
    s_t1_arr = states[:, 1:].ravel()
    n_trans = len(s_t_arr)

    feat_t = torch.tensor(raw_feats[s_t_arr])
    feat_t1 = torch.tensor(raw_feats[s_t1_arr])
    all_feats_t = torch.tensor(raw_feats)

    hidden = CONFIG['nn_hidden']
    batch_size = min(8192, n_trans)

    class HNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    h_vals_dict = {}

    for name, flow_vec in flow_funcs.items():
        flow_t = torch.tensor(flow_vec[s_t_arr])
        net = HNet(K)
        opt_nn = optim.Adam(net.parameters(), lr=CONFIG['nn_lr'])

        for avi_iter in range(CONFIG['nn_avi_iters']):
            with torch.no_grad():
                h_next = net(feat_t1)
            Y = flow_t + gamma * h_next

            for epoch in range(CONFIG['nn_epochs_per_iter']):
                perm = torch.randperm(n_trans)
                for start in range(0, n_trans, batch_size):
                    idx = perm[start:start+batch_size]
                    pred = net(feat_t[idx])
                    loss = ((pred - Y[idx]) ** 2).mean()
                    opt_nn.zero_grad()
                    loss.backward()
                    opt_nn.step()

        with torch.no_grad():
            h_vals_dict[name] = net(all_feats_t).numpy()

    ev_1, ev_2, ev_rc, ev_H = h_vals_dict['1'], h_vals_dict['2'], h_vals_dict['rc'], h_vals_dict['H']

    Pev_1 = P_keep.dot(ev_1)
    Pev_2 = P_keep.dot(ev_2)
    Pev_rc = P_keep.dot(ev_rc)
    Pev_H = P_keep.dot(ev_H)

    def neg_ll_nn(theta_vec):
        RC, th1, th2 = theta_vec
        cost = th1 * x_vec + th2 * x_vec ** 2
        v0 = -cost + gamma * (th1 * Pev_1 + th2 * Pev_2 + RC * Pev_rc + Pev_H)
        ev_zero = th1*ev_1[idx_zero] + th2*ev_2[idx_zero] + RC*ev_rc[idx_zero] + ev_H[idx_zero]
        v1 = -RC + gamma * ev_zero
        return _neg_loglik_from_vdiff(v0 - v1, s_flat, a_flat)

    try:
        res = minimize(neg_ll_nn, [4.0, 1.5, 3.0], method='L-BFGS-B',
                       bounds=[(0.1, 20.0)]*3, options={'maxiter': 200})
        return {'converged': res.success, 'theta': res.x.tolist(), 'time': time.time()-t0}
    except Exception:
        return {'converged': False, 'theta': [np.nan]*3, 'time': time.time()-t0}


# ============================================================================
# Experiments
# ============================================================================

def run_single_estimation(method_name, states, actions, env_config, timeout=300, precomp=None):
    dispatch = {
        'NFXP': lambda: estimate_nfxp(states, actions, env_config, timeout, precomp),
        'CCP': lambda: estimate_ccp(states, actions, env_config, precomp),
        'TD-CCP Linear': lambda: estimate_td_ccp_linear(states, actions, env_config, precomp),
        'TD-CCP Neural': lambda: estimate_td_ccp_nn(states, actions, env_config, precomp),
    }
    return dispatch[method_name]()


def run_experiment_2():
    """Experiment 2: Scaling across K=1..4."""
    K_values = CONFIG['exp2_K_values']
    M = CONFIG['M']
    gamma = CONFIG['gamma']
    RC_true, th1_true, th2_true = CONFIG['true_RC'], CONFIG['true_theta1'], CONFIG['true_theta2']
    n_seeds = CONFIG['exp2_seeds']

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: Scaling (K={K_values})")
    print(f"  Seeds per K: {n_seeds}")
    print(f"{'='*70}")

    results = {}
    for K in K_values:
        n_states = M ** K
        print(f"\n  K={K}, |S|={n_states}")

        t_env = time.time()
        env = MultiComponentBusEngine(K, M, RC_true, th1_true, th2_true, gamma, CONFIG['trans_probs'])
        EV, v0, v1_vec, _ = env.solve_vi()
        print(f"  VI solve: {time.time()-t_env:.1f}s")

        env_config = {'K': K, 'M': M, 'gamma': gamma, 'trans_probs': CONFIG['trans_probs']}
        results[K] = {m: [] for m in METHOD_ORDER}

        # Precompute environment structures once per K (avoids redundant sparse
        # matrix builds across methods and seeds)
        t_pre = time.time()
        precomp = _precompute_env(K, M, gamma, CONFIG['trans_probs'])
        print(f"  Precompute env: {time.time()-t_pre:.1f}s")

        for seed in range(n_seeds):
            print(f"\n    Seed {seed+1}/{n_seeds}")
            sim_states, sim_actions = env.simulate_panel(v0, v1_vec, CONFIG['N_agents'],
                                                          CONFIG['T_periods'], seed=seed)
            for method in METHOD_ORDER:
                print(f"      {method:20s} ... ", end='', flush=True)
                res = run_single_estimation(method, sim_states, sim_actions, env_config,
                                             timeout=CONFIG['timeout'], precomp=precomp)
                theta = res['theta']
                status = ''
                if res.get('timeout'):
                    status = ' TIMEOUT'
                elif res.get('reason') == 'sparse':
                    status = f' SPARSE (cov={res.get("coverage", 0):.1%})'
                print(f"RC={theta[0]:6.3f}  th1={theta[1]:6.3f}  th2={theta[2]:6.3f}  "
                      f"({res['time']:.1f}s){status}")
                results[K][method].append(res)

    return results


# ============================================================================
# Compute / Cache
# ============================================================================

def compute_data():
    cached = load_results(CACHE_DIR, 'nfxp_ccp_td', CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    exp2 = run_experiment_2()

    data = {'exp2': exp2}
    save_results(CACHE_DIR, 'nfxp_ccp_td', CONFIG, data)
    return data


# ============================================================================
# Output Generation
# ============================================================================

def generate_outputs(data):
    exp2 = data['exp2']
    true_params = [CONFIG['true_RC'], CONFIG['true_theta1'], CONFIG['true_theta2']]

    # --- Figure: Scaling — wall-clock time vs K ---
    K_values = CONFIG['exp2_K_values']
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        times, k_plot = [], []
        for K in K_values:
            if K not in exp2:
                continue
            valid_times = [r['time'] for r in exp2[K][method]
                           if not r.get('skipped', False)
                           and not any(np.isnan(r['theta']))
                           and not np.isnan(r.get('time', np.nan))]
            if valid_times:
                times.append(np.mean(valid_times))
                k_plot.append(K)
        if k_plot:
            ax.plot(k_plot, times, 'o-', color=METHOD_COLORS[method], label=method, markersize=6)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Components (K)')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_xticks(K_values)
    ax.set_xticklabels([f'K={k}\n|S|={CONFIG["M"]**k:,}' for k in K_values])
    ax.legend(loc='upper left')
    ax.set_title('Estimation Time vs State-Space Scale')
    fig.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, 'nfxp_ccp_td_scaling_time.png')
    fig.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Table: Consolidated results ---
    lines = []
    lines.append(r'\begin{tabular}{ll rrr rrr}')
    lines.append(r'\toprule')
    lines.append(r'Method & K & $|S|$ & Time (s) & RC Bias & RC RMSE & '
                 r'$\theta_1$ RMSE & $\theta_2$ RMSE \\')
    lines.append(r'\midrule')

    for K in K_values:
        if K not in exp2:
            continue
        n_states = CONFIG['M'] ** K
        for method in METHOD_ORDER:
            valid = [r for r in exp2[K][method]
                     if not r.get('skipped', False) and not any(np.isnan(r['theta']))]
            if not valid:
                lines.append(f'{method} & {K} & {n_states:,} & --- & --- & --- & --- & --- \\\\')
                continue
            times = [r['time'] for r in valid]
            rc_vals = [r['theta'][0] for r in valid]
            th1_vals = [r['theta'][1] for r in valid]
            th2_vals = [r['theta'][2] for r in valid]
            rc_bias = np.mean(rc_vals) - true_params[0]
            rc_rmse = np.sqrt(np.mean([(v - true_params[0])**2 for v in rc_vals]))
            th1_rmse = np.sqrt(np.mean([(v - true_params[1])**2 for v in th1_vals]))
            th2_rmse = np.sqrt(np.mean([(v - true_params[2])**2 for v in th2_vals]))
            mean_time = np.mean(times)
            lines.append(f'{method} & {K} & {n_states:,} & {mean_time:.1f} & '
                         f'{rc_bias:+.3f} & {rc_rmse:.3f} & {th1_rmse:.3f} & '
                         f'{th2_rmse:.3f} \\\\')
        if K < K_values[-1]:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    table_path = os.path.join(SCRIPT_DIR, 'nfxp_ccp_td_results.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {table_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NFXP vs CCP vs TD structural estimation')
    add_cache_args(parser)
    args = parser.parse_args()

    print("="*70)
    print("STRUCTURAL ESTIMATION: NFXP vs CCP vs TD-CCP")
    print(f"  Model: Multi-component bus engine (M={CONFIG['M']})")
    print(f"  True params: RC={CONFIG['true_RC']}, theta1={CONFIG['true_theta1']}, "
          f"theta2={CONFIG['true_theta2']}")
    print(f"  gamma={CONFIG['gamma']}")
    print("="*70)

    if args.plots_only:
        cached = load_results(CACHE_DIR, 'nfxp_ccp_td', CONFIG)
        if cached is None:
            print("ERROR: No cached data found. Run without --plots-only first.")
            sys.exit(1)
        generate_outputs(cached)
    elif args.data_only:
        compute_data()
    else:
        data = compute_data()
        generate_outputs(data)

    print("\nOutput files:")
    for f in ['nfxp_ccp_td_scaling_time.png', 'nfxp_ccp_td_results.tex']:
        p = os.path.join(SCRIPT_DIR, f)
        status = "EXISTS" if os.path.exists(p) else "MISSING"
        print(f"  [{status}] {p}")


if __name__ == '__main__':
    main()
