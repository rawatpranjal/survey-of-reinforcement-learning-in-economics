"""
Baird's Counterexample: Divergence and Three Fixes
Chapter 3 (Theory) — personal understanding, not for the paper.
Demonstrates off-policy semi-gradient TD divergence and three stabilization mechanisms.
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
    """Expected update matrix: A = (1/7) sum_s x(s)(gamma*x(7) - x(s))^T.

    Under behavior policy b (dashed 6/7, solid 1/7) and target policy pi
    (solid always), only solid transitions contribute. The IS ratio
    b(solid)*rho = (1/7)*7 = 1 cancels, leaving effective weight d(s) = 1/7.
    All solid transitions go to state 7.
    """
    x7 = X[6]
    A = np.zeros((N_WEIGHTS, N_WEIGHTS))
    for s in range(N_STATES):
        A += np.outer(X[s], GAMMA * x7 - X[s])
    A /= N_STATES
    return A


def compute_max_V(X, w_history):
    """Compute max|V(s)| = max_s |x(s)^T w| at each epoch."""
    return np.array([np.max(np.abs(X @ w)) for w in w_history])


# ======================================================================
# Panel 1: Semi-gradient off-policy TD (diverges)
# ======================================================================

def run_semigradient(X, n_epochs):
    """Standard semi-gradient TD with IS correction. Expected updates."""
    w = make_w0()
    A = make_A(X)
    history = [w.copy()]
    for _ in range(n_epochs):
        w = w + ALPHA * A @ w
        history.append(w.copy())
    return history


# ======================================================================
# Panel 2: Fitted value iteration (weaken bootstrapping)
# ======================================================================

def run_fitted_vi(X, n_epochs):
    """Fitted value iteration: solve regression exactly at each step.

    This is the theoretical limit of what target networks approximate
    (K -> infinity). At each iteration, the bootstrap target V_target(7) is
    frozen, the regression V(s) -> gamma * V_target(7) is solved in closed
    form, and the target is updated. Each step is a single application of
    the projected Bellman operator with exact projection.

    Spectral radius = gamma = 0.99, so convergence is guaranteed.
    """
    w = make_w0()
    x7 = X[6]
    X_pinv = np.linalg.pinv(X)
    history = [w.copy()]
    for _ in range(n_epochs):
        target_val = GAMMA * x7 @ w
        targets = target_val * np.ones(N_STATES)
        w = X_pinv @ targets
        history.append(w.copy())
    return history


# ======================================================================
# Panel 3: TDC / Gradient TD (fix the projection)
# ======================================================================

def run_tdc(X, n_epochs, eta_h=10.0):
    """TDC algorithm (Sutton et al. 2009). Two-timescale expected updates.

    w update: alpha * [delta * x(s) - gamma * x(7) * (x(s)^T h)]
    h update: beta  * [delta - x(s)^T h] * x(s)
    where beta = alpha * eta_h (h learns faster).

    Note: TDC stabilizes the learning (no divergence) but converges to a
    biased fixed point in this off-policy setting. The IS correction fixes
    the action distribution but not the state distribution mismatch, so the
    TDC fixed point satisfies (I - gamma B' C^+) A w* = 0 rather than
    A w* = 0. The value function remains bounded but nonzero.
    """
    w = make_w0()
    h = np.zeros(N_WEIGHTS)
    x7 = X[6]
    beta = ALPHA * eta_h
    history = [w.copy()]
    for _ in range(n_epochs):
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
    return history


# ======================================================================
# Panel 4: L2 regularization (shrink the projection)
# ======================================================================

def run_regularized(X, n_epochs, eta_reg=1.0):
    """Semi-gradient TD + L2 penalty: dw -= eta * w per step.

    The effective update matrix becomes (A - eta*I). With eta=1.0, the
    max eigenvalue flips from +0.24 to -0.76, giving strong contraction.
    Converges to w=0 (unique fixed point since A - eta*I is invertible).
    """
    w = make_w0()
    A = make_A(X)
    history = [w.copy()]
    for _ in range(n_epochs):
        w = w + ALPHA * (A @ w - eta_reg * w)
        history.append(w.copy())
    return history


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    X = make_features()
    A = make_A(X)

    # --- Eigenvalue analysis ---
    eigvals = np.linalg.eigvals(A)
    real_parts = sorted(eigvals.real, reverse=True)

    print('=' * 65)
    print('Baird\'s Counterexample: Divergence and Three Fixes')
    print('=' * 65)
    print(f'\nParameters: gamma={GAMMA}, alpha={ALPHA}, epochs={N_EPOCHS}')
    print(f'Initial w_8 = {make_w0()[7]:.1f}')

    print(f'\nExpected update matrix A — eigenvalues (real parts):')
    for i, ev in enumerate(real_parts):
        tag = ' <-- unstable' if ev > 1e-8 else ''
        print(f'  lambda_{i+1} = {ev:+.4f}{tag}')

    rho = max(abs(1 + ALPHA * eigvals))
    print(f'\nSpectral radius of (I + alpha*A): {rho:.6f}')
    print(f'System {"diverges" if rho > 1 else "converges"} '
          f'(rho {">" if rho > 1 else "<="} 1)')

    # Fitted VI spectral radius
    x7 = X[6]
    X_pinv = np.linalg.pinv(X)
    M_fvi = GAMMA * np.outer(X_pinv @ np.ones(N_STATES), x7)
    print(f'\nFitted VI iteration matrix:')
    print(f'  Spectral radius: {max(abs(np.linalg.eigvals(M_fvi))):.4f} '
          f'(= gamma = {GAMMA})')

    # Regularized eigenvalues
    A_reg = A - 1.0 * np.eye(N_WEIGHTS)
    eigvals_reg = np.linalg.eigvals(A_reg)
    print(f'\nWith L2 regularization (eta=1.0):')
    print(f'  Max real eigenvalue of (A - eta*I): {max(eigvals_reg.real):+.4f}')
    print(f'  Spectral radius of (I + alpha*(A-eta*I)): '
          f'{max(abs(1 + ALPHA * eigvals_reg)):.6f}')

    # --- Run simulations ---
    hist_sg = run_semigradient(X, N_EPOCHS)
    hist_fvi = run_fitted_vi(X, N_EPOCHS)
    hist_tdc = run_tdc(X, N_EPOCHS, eta_h=10.0)
    hist_reg = run_regularized(X, N_EPOCHS, eta_reg=1.0)

    # Extract w_8 and max|V(s)| traces
    w8_sg = np.array([w[7] for w in hist_sg])
    w8_fvi = np.array([w[7] for w in hist_fvi])
    w8_tdc = np.array([w[7] for w in hist_tdc])
    w8_reg = np.array([w[7] for w in hist_reg])

    maxV_sg = compute_max_V(X, hist_sg)
    maxV_fvi = compute_max_V(X, hist_fvi)
    maxV_tdc = compute_max_V(X, hist_tdc)
    maxV_reg = compute_max_V(X, hist_reg)

    # --- Summary tables ---
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

    # --- Cross-check against notes ---
    print(f'\nVerification (Section 5a):')
    print(f'  w_8 at epoch 0:   {w8_sg[0]:.4f}  (expected 10.0000)')
    print(f'  w_8 at epoch 1:   {w8_sg[1]:.4f}  (expected 10.0747)')
    print(f'  w_8 at epoch 100: {w8_sg[100]:.2f}  (expected ~18.50)')

    w0 = make_w0()
    V1 = X[0] @ w0
    V7 = X[6] @ w0
    print(f'  V(1) at epoch 0:  {V1:.2f}  (expected 12.00)')
    print(f'  V(7) at epoch 0:  {V7:.2f}  (expected 21.00)')
    print(f'  delta(1) epoch 0: {GAMMA * V7 - V1:.2f}  (expected 8.79)')

    # --- Plot ---
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
