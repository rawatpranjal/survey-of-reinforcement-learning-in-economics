"""
Baird's Counterexample: Divergence and Three Fixes (Verbose Edition)
Chapter 3a (Illustrated Example) — personal understanding, maximum logging.
Demonstrates off-policy semi-gradient TD divergence and three stabilization mechanisms.
Every computation step is printed to stdout for deep understanding.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS
apply_style()

import matplotlib.pyplot as plt

# ======================================================================
# Common setup: Sutton & Barto (2018) Example 11.2
# ======================================================================

N_STATES = 7
N_WEIGHTS = 8
GAMMA = 0.99
ALPHA = 0.01
N_EPOCHS = 1000


def make_features():
    """Feature vectors for 7-state Baird's star MDP."""
    X = np.zeros((N_STATES, N_WEIGHTS))
    for i in range(6):          # states 1-6 (0-indexed: 0-5)
        X[i, i] = 2.0          # own weight with coefficient 2
        X[i, 7] = 1.0          # shared weight w_8 with coefficient 1
    X[6, 6] = 1.0              # state 7: own weight w_7
    X[6, 7] = 2.0              # state 7: shared weight w_8 with coefficient 2
    return X


def make_w0():
    """Initial weights: w_1..w_7 = 1, w_8 = 10."""
    w = np.ones(N_WEIGHTS)
    w[7] = 10.0
    return w


def make_A(X):
    """Expected update matrix: A = (1/7) sum_s x(s)(gamma*x(7) - x(s))^T."""
    x7 = X[6]
    A = np.zeros((N_WEIGHTS, N_WEIGHTS))
    for s in range(N_STATES):
        A += np.outer(X[s], GAMMA * x7 - X[s])
    A /= N_STATES
    return A


def compute_V(X, w):
    """Compute V(s) for all states."""
    return X @ w


def compute_max_V(X, w_history):
    """Compute max|V(s)| = max_s |x(s)^T w| at each epoch."""
    return np.array([np.max(np.abs(X @ w)) for w in w_history])


def print_state_table(X, w, header=""):
    """Print V(s) for all 7 states."""
    V = compute_V(X, w)
    if header:
        print(header)
    print(f"  {'State':<8} {'V(s)':>10}")
    print(f"  {'-'*8} {'-'*10}")
    for s in range(N_STATES):
        print(f"  s={s+1:<5} {V[s]:>10.4f}")


def should_log_semigradient(epoch):
    """Log every epoch for first 10, then every 50."""
    return epoch <= 10 or epoch % 50 == 0


def should_log_fix(epoch):
    """Log at epochs 0, 1, 5, 10, 50, 100, 200, 500, 1000."""
    return epoch in {0, 1, 5, 10, 50, 100, 200, 500, 1000}


# ======================================================================
# Panel 1: Semi-gradient off-policy TD (diverges)
# ======================================================================

def run_semigradient(X, n_epochs):
    """Standard semi-gradient TD with IS correction. Expected updates."""
    w = make_w0()
    A = make_A(X)
    x7 = X[6]
    history = [w.copy()]

    print("\n" + "=" * 70)
    print("PANEL 1: Semi-gradient off-policy TD (diverges)")
    print("=" * 70)
    print(f"Update rule: w <- w + alpha * A @ w")
    print(f"  alpha = {ALPHA}, gamma = {GAMMA}")

    for epoch in range(1, n_epochs + 1):
        w_old = w.copy()
        V_old = compute_V(X, w_old)

        # Compute per-state TD errors and contributions to Δw_8
        if should_log_semigradient(epoch):
            print(f"\n--- Epoch {epoch} ---")
            print(f"  w = [{', '.join(f'{wi:.4f}' for wi in w_old)}]")
            print(f"  V(s): [{', '.join(f'{v:.4f}' for v in V_old)}]")

            # Per-state breakdown
            pos_dw8 = 0.0
            neg_dw8 = 0.0
            print(f"  {'State':<8} {'delta(s)':>10} {'x_8(s)':>8} {'Contrib dw_8':>14}")
            print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*14}")
            for s in range(N_STATES):
                delta_s = GAMMA * (x7 @ w_old) - (X[s] @ w_old)
                x8_s = X[s, 7]
                contrib = (1.0 / N_STATES) * delta_s * x8_s
                print(f"  s={s+1:<5} {delta_s:>+10.4f} {x8_s:>8.1f} {contrib:>+14.6f}")
                if contrib > 0:
                    pos_dw8 += contrib
                else:
                    neg_dw8 += contrib
            net_dw8 = pos_dw8 + neg_dw8
            print(f"  Positive push (s1-s6): {pos_dw8:+.6f}")
            print(f"  Negative push (s7):    {neg_dw8:+.6f}")
            print(f"  Net Δw_8 (before alpha): {net_dw8:+.6f}")
            print(f"  Net Δw_8 (after alpha):  {ALPHA * net_dw8:+.6f}")

        w = w + ALPHA * A @ w
        history.append(w.copy())

        if should_log_semigradient(epoch):
            print(f"  w_8: {w_old[7]:.4f} -> {w[7]:.4f}")
            print(f"  max|V(s)|: {np.max(np.abs(V_old)):.4f} -> {np.max(np.abs(compute_V(X, w))):.4f}")

    return history


# ======================================================================
# Panel 2: Fitted value iteration (weaken bootstrapping)
# ======================================================================

def run_fitted_vi(X, n_epochs):
    """Fitted value iteration: exact projection at each step."""
    w = make_w0()
    x7 = X[6]
    X_pinv = np.linalg.pinv(X)
    history = [w.copy()]

    print("\n" + "=" * 70)
    print("PANEL 2: Fitted Value Iteration (weaken bootstrapping)")
    print("=" * 70)
    print("Update rule: targets = gamma * V_target(7) * ones; w = X_pinv @ targets")
    print(f"  Spectral radius = gamma = {GAMMA}")

    for epoch in range(1, n_epochs + 1):
        w_old = w.copy()
        target_val = GAMMA * x7 @ w
        targets = target_val * np.ones(N_STATES)
        w = X_pinv @ targets
        history.append(w.copy())

        if should_log_fix(epoch):
            V = compute_V(X, w)
            print(f"\n  Epoch {epoch:>4d}: w_8 = {w[7]:>10.4f}, "
                  f"max|V| = {np.max(np.abs(V)):>10.4f}")
            print(f"    w = [{', '.join(f'{wi:.4f}' for wi in w)}]")
            print(f"    V = [{', '.join(f'{v:.4f}' for v in V)}]")
            print(f"    target_val (gamma*V(7)): {target_val:.4f}")

    return history


# ======================================================================
# Panel 3: TDC / Gradient TD (fix the projection)
# ======================================================================

def run_tdc(X, n_epochs, eta_h=10.0):
    """TDC algorithm (Sutton et al. 2009). Two-timescale expected updates."""
    w = make_w0()
    h = np.zeros(N_WEIGHTS)
    x7 = X[6]
    beta = ALPHA * eta_h
    history = [w.copy()]

    print("\n" + "=" * 70)
    print("PANEL 3: TDC / Gradient TD (fix the projection)")
    print("=" * 70)
    print(f"  alpha = {ALPHA}, beta = {beta} (eta_h = {eta_h})")
    print("  w update: alpha * [delta * x(s) - gamma * x(7) * (x(s)^T h)]")
    print("  h update: beta  * [delta - x(s)^T h] * x(s)")

    for epoch in range(1, n_epochs + 1):
        dw = np.zeros(N_WEIGHTS)
        dh = np.zeros(N_WEIGHTS)
        for s in range(N_STATES):
            delta = GAMMA * x7 @ w - X[s] @ w
            dw += (1.0 / N_STATES) * (delta * X[s]
                                       - GAMMA * x7 * (X[s] @ h))
            dh += (1.0 / N_STATES) * (delta - X[s] @ h) * X[s]
        w = w + ALPHA * dw
        h = h + beta * dh
        history.append(w.copy())

        if should_log_fix(epoch):
            V = compute_V(X, w)
            print(f"\n  Epoch {epoch:>4d}: w_8 = {w[7]:>10.4f}, "
                  f"max|V| = {np.max(np.abs(V)):>10.4f}")
            print(f"    w = [{', '.join(f'{wi:.4f}' for wi in w)}]")
            print(f"    h = [{', '.join(f'{hi:.4f}' for hi in h)}]")
            print(f"    V = [{', '.join(f'{v:.4f}' for v in V)}]")

    return history


# ======================================================================
# Panel 4: L2 regularization (shrink the projection)
# ======================================================================

def run_regularized(X, n_epochs, eta_reg=1.0):
    """Semi-gradient TD + L2 penalty: dw -= eta * w per step."""
    w = make_w0()
    A = make_A(X)
    history = [w.copy()]

    print("\n" + "=" * 70)
    print("PANEL 4: L2 Regularization (shrink the projection)")
    print("=" * 70)
    print(f"  Update: w <- w + alpha * (A @ w - eta * w)")
    print(f"  eta = {eta_reg}")

    for epoch in range(1, n_epochs + 1):
        w = w + ALPHA * (A @ w - eta_reg * w)
        history.append(w.copy())

        if should_log_fix(epoch):
            V = compute_V(X, w)
            print(f"\n  Epoch {epoch:>4d}: w_8 = {w[7]:>10.4f}, "
                  f"max|V| = {np.max(np.abs(V)):>10.4f}")
            print(f"    w = [{', '.join(f'{wi:.4f}' for wi in w)}]")
            print(f"    V = [{', '.join(f'{v:.4f}' for v in V)}]")

    return history


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    X = make_features()
    A = make_A(X)

    # ==========================================
    # Pre-simulation diagnostics
    # ==========================================
    print("=" * 70)
    print("Baird's Counterexample: Divergence and Three Fixes")
    print("VERBOSE EDITION — every computation step logged")
    print("=" * 70)
    print(f"\nParameters: gamma={GAMMA}, alpha={ALPHA}, epochs={N_EPOCHS}")

    # Feature matrix
    print(f"\n{'='*70}")
    print("FEATURE MATRIX X (7 states x 8 weights)")
    print(f"{'='*70}")
    print(f"  {'':>8}", end="")
    for j in range(N_WEIGHTS):
        print(f"  w_{j+1:d}", end="")
    print()
    for s in range(N_STATES):
        print(f"  s={s+1:<5}", end="")
        for j in range(N_WEIGHTS):
            print(f"  {X[s,j]:>3.0f}", end="")
        formula = f"V(s{s+1}) = "
        terms = []
        for j in range(N_WEIGHTS):
            if X[s, j] != 0:
                coeff = int(X[s, j])
                terms.append(f"{coeff}*w_{j+1}")
        formula += " + ".join(terms)
        print(f"    {formula}")

    # Initial values
    w0 = make_w0()
    print(f"\nInitial weights: w = [{', '.join(f'{wi:.1f}' for wi in w0)}]")
    print_state_table(X, w0, header="\nInitial V(s):")

    # Expected update matrix
    print(f"\n{'='*70}")
    print("EXPECTED UPDATE MATRIX A = (1/7) sum_s x(s)(gamma*x(7) - x(s))^T")
    print(f"{'='*70}")
    print("  A =")
    for i in range(N_WEIGHTS):
        row = "  [" + "  ".join(f"{A[i,j]:>+8.4f}" for j in range(N_WEIGHTS)) + " ]"
        print(row)

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(A)
    # Sort by real part descending
    idx = np.argsort(-eigvals.real)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    print(f"\n{'='*70}")
    print("EIGENVALUE DECOMPOSITION OF A")
    print(f"{'='*70}")
    for i in range(N_WEIGHTS):
        ev = eigvals[i]
        tag = ""
        if ev.real > 1e-8:
            tag = " <-- UNSTABLE"
        elif abs(ev.real) < 1e-8:
            tag = " <-- neutral"
        evec = eigvecs[:, i].real
        print(f"  lambda_{i+1} = {ev.real:+.6f}{tag}")
        print(f"    eigenvector: [{', '.join(f'{v:+.4f}' for v in evec)}]")

    # Spectral radius of iteration matrix
    rho_A = max(abs(1 + ALPHA * eigvals))
    print(f"\n  Spectral radius of (I + alpha*A): {rho_A.real:.6f}")
    print(f"  System {'DIVERGES' if rho_A.real > 1 else 'converges'} "
          f"(rho {'>' if rho_A.real > 1 else '<='} 1)")

    # Null space of X
    print(f"\n{'='*70}")
    print("NULL SPACE OF X")
    print(f"{'='*70}")
    U, S, Vt = np.linalg.svd(X)
    null_mask = S < 1e-10
    null_dim = N_WEIGHTS - np.sum(~null_mask)
    print(f"  Singular values: [{', '.join(f'{s:.4f}' for s in S)}]")
    # Null space is last (N_WEIGHTS - rank) rows of Vt
    rank = np.sum(S > 1e-10)
    print(f"  Rank of X: {rank}")
    print(f"  Null space dimension: {N_WEIGHTS - rank}")
    if N_WEIGHTS > rank:
        for k in range(rank, N_WEIGHTS):
            nvec = Vt[k]
            print(f"  Null vector {k - rank + 1}: [{', '.join(f'{v:+.4f}' for v in nvec)}]")
        print("  (Any w in the null space satisfies X @ w = 0, i.e., V(s) = 0 for all s)")
        print("  TDC converges to a point in the null space shifted by the projection bias")

    # Fitted VI spectral radius
    x7 = X[6]
    X_pinv = np.linalg.pinv(X)
    M_fvi = GAMMA * np.outer(X_pinv @ np.ones(N_STATES), x7)
    rho_fvi = max(abs(np.linalg.eigvals(M_fvi)))
    print(f"\n  Fitted VI iteration matrix spectral radius: {rho_fvi:.4f} (= gamma = {GAMMA})")

    # Regularized eigenvalues
    A_reg = A - 1.0 * np.eye(N_WEIGHTS)
    eigvals_reg = np.linalg.eigvals(A_reg)
    max_ev_reg = max(eigvals_reg.real)
    rho_reg = max(abs(1 + ALPHA * eigvals_reg))
    print(f"\n  With L2 regularization (eta=1.0):")
    print(f"    Max real eigenvalue of (A - eta*I): {max_ev_reg:+.4f}")
    print(f"    Spectral radius of (I + alpha*(A-eta*I)): {rho_reg.real:.6f}")
    print(f"    All eigenvalues of (A - eta*I):")
    for i, ev in enumerate(sorted(eigvals_reg.real, reverse=True)):
        print(f"      {ev:+.4f}")

    # ==========================================
    # Run simulations (with per-epoch logging)
    # ==========================================
    hist_sg = run_semigradient(X, N_EPOCHS)
    hist_fvi = run_fitted_vi(X, N_EPOCHS)
    hist_tdc = run_tdc(X, N_EPOCHS, eta_h=10.0)
    hist_reg = run_regularized(X, N_EPOCHS, eta_reg=1.0)

    # ==========================================
    # Summary tables
    # ==========================================
    w8_sg = np.array([w[7] for w in hist_sg])
    w8_fvi = np.array([w[7] for w in hist_fvi])
    w8_tdc = np.array([w[7] for w in hist_tdc])
    w8_reg = np.array([w[7] for w in hist_reg])

    maxV_sg = compute_max_V(X, hist_sg)
    maxV_fvi = compute_max_V(X, hist_fvi)
    maxV_tdc = compute_max_V(X, hist_tdc)
    maxV_reg = compute_max_V(X, hist_reg)

    print(f"\n{'='*70}")
    print("SUMMARY TABLES")
    print(f"{'='*70}")

    print(f'\n--- w_8 trajectory ---')
    print(f'{"Method":<25} {"w_8(0)":>8} {"w_8(100)":>10} '
          f'{"w_8(500)":>10} {"w_8(1000)":>11}')
    print('-' * 68)
    for name, trace in [('Semi-gradient TD', w8_sg),
                        ('Fitted VI', w8_fvi),
                        ('TDC (Gradient TD)', w8_tdc),
                        ('L2 regularization', w8_reg)]:
        v = [trace[0], trace[100], trace[500], trace[-1]]
        print(f'{name:<25} {v[0]:>8.2f} {v[1]:>10.2f} '
              f'{v[2]:>10.2f} {v[3]:>11.2f}')

    print(f'\n--- max|V(s)| trajectory ---')
    print(f'{"Method":<25} {"t=0":>8} {"t=100":>10} '
          f'{"t=500":>10} {"t=1000":>11}')
    print('-' * 68)
    for name, trace in [('Semi-gradient TD', maxV_sg),
                        ('Fitted VI', maxV_fvi),
                        ('TDC (Gradient TD)', maxV_tdc),
                        ('L2 regularization', maxV_reg)]:
        v = [trace[0], trace[100], trace[500], trace[-1]]
        print(f'{name:<25} {v[0]:>8.2f} {v[1]:>10.2f} '
              f'{v[2]:>10.2f} {v[3]:>11.2f}')

    # ==========================================
    # Verification cross-checks
    # ==========================================
    print(f'\n{"="*70}')
    print('VERIFICATION (Section 5a cross-checks)')
    print(f'{"="*70}')
    print(f'  w_8 at epoch 0:   {w8_sg[0]:.4f}  (expected 10.0000)')
    print(f'  w_8 at epoch 1:   {w8_sg[1]:.4f}  (expected 10.0747)')
    print(f'  w_8 at epoch 100: {w8_sg[100]:.2f}  (expected ~18.50)')

    V0 = compute_V(X, make_w0())
    print(f'  V(1) at epoch 0:  {V0[0]:.2f}  (expected 12.00)')
    print(f'  V(7) at epoch 0:  {V0[6]:.2f}  (expected 21.00)')
    delta1_0 = GAMMA * V0[6] - V0[0]
    print(f'  delta(1) epoch 0: {delta1_0:.2f}  (expected 8.79)')

    # ==========================================
    # Plot
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = np.arange(N_EPOCHS + 1)

    panels = [
        (axes[0, 0], maxV_sg,  COLORS['red'],
         'Semi-gradient off-policy TD (diverges)'),
        (axes[0, 1], maxV_fvi, COLORS['green'],
         'Fix 1: Fitted value iteration'),
        (axes[1, 0], maxV_tdc, COLORS['orange'],
         'Fix 2: TDC / Gradient TD'),
        (axes[1, 1], maxV_reg, COLORS['purple'],
         r'Fix 3: $\ell_2$ regularization ($\eta = 1.0$)'),
    ]

    for ax, trace, color, title in panels:
        ax.plot(epochs, trace, color=color)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\max_s |V(s)|$')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

    fig.suptitle("Baird's Counterexample ($\\gamma=0.99$, $\\alpha=0.01$)",
                 fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'bairds_counterexample.png')
    fig.savefig(out_path)
    print(f'\nFigure saved: {out_path}')
