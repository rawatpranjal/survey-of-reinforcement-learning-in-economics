"""
Baird's Counterexample: Divergence and Three Fixes
Chapter 3 (Theory) — numerical companion to Theorem thm:baird (off-policy
divergence on the six-state star MDP of Baird 1995, Figure 1).
Demonstrates off-policy semi-gradient TD divergence and three stabilization
mechanisms, using expected (population) updates, the limit of sampling every
transition equally often.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS

apply_style()

import matplotlib.pyplot as plt

# ======================================================================
# Common setup: Baird (1995), Figure 1 "star problem"
# Six states, seven weights. Spokes 1-5: V(i) = w_0 + 2 w_i.
# Hub (state 6): V(6) = 2 w_0 + w_6. All rewards zero, so V* = 0 is
# exactly representable (w = 0). Every transition leads to the hub;
# the hub self-loops. Training weights all six transitions equally
# (uniform, not the on-policy distribution concentrated on the hub).
# ======================================================================

N_STATES = 6
N_WEIGHTS = 7  # shared w_0 plus one weight per state
GAMMA = 0.99
ALPHA = 0.01
N_EPOCHS = 1000


def make_features():
    """Feature vectors for the six-state star MDP (Baird 1995, Figure 1)."""
    X = np.zeros((N_STATES, N_WEIGHTS))
    for i in range(5):  # spokes 1-5 (0-indexed rows 0-4)
        X[i, 0] = 1.0  # shared weight w_0, coefficient 1
        X[i, i + 1] = 2.0  # own weight with coefficient 2
    X[5, 0] = 2.0  # hub: shared weight w_0, coefficient 2
    X[5, 6] = 1.0  # hub: own weight w_6, coefficient 1
    return X


def make_w0():
    """Initial weights: w_0..w_5 = 1, w_6 = 10.

    Gives V(spoke) = 3 and V(6) = 12, so gamma*V(6) = 11.88 > V(s) and
    the self-sustaining divergence precondition of Theorem thm:baird holds.
    """
    w = np.ones(N_WEIGHTS)
    w[6] = 10.0
    return w


def make_A(X):
    """Expected update matrix: A = (1/6) sum_s x(s)(gamma*x(6) - x(s))^T.

    All six transitions land in the hub (state 6), and each transition is
    weighted 1/6 (Baird's "each possible transition is observed equally
    often"). The on-policy stationary distribution would concentrate on the
    hub instead; the uniform weighting is what breaks the contraction.
    """
    x_hub = X[5]
    A = np.zeros((N_WEIGHTS, N_WEIGHTS))
    for s in range(N_STATES):
        A += np.outer(X[s], GAMMA * x_hub - X[s])
    A /= N_STATES
    return A


def compute_max_V(X, w_history):
    """Compute max|V(s)| = max_s |x(s)^T w| at each epoch."""
    return np.array([np.max(np.abs(X @ w)) for w in w_history])


# ======================================================================
# Panel 1: Semi-gradient off-policy TD (diverges)
# ======================================================================


def run_semigradient(X, n_epochs):
    """Standard semi-gradient TD. Expected updates."""
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
    (K -> infinity). At each iteration, the bootstrap target V_target(6) is
    frozen, the regression V(s) -> gamma * V_target(6) is solved in closed
    form, and the target is updated. Each step is a single application of
    the projected Bellman operator with exact projection.

    Spectral radius = gamma = 0.99, so convergence is guaranteed.
    """
    w = make_w0()
    x_hub = X[5]
    X_pinv = np.linalg.pinv(X)
    history = [w.copy()]
    for _ in range(n_epochs):
        target_val = GAMMA * x_hub @ w
        targets = target_val * np.ones(N_STATES)
        w = X_pinv @ targets
        history.append(w.copy())
    return history


# ======================================================================
# Panel 3: TDC / Gradient TD (fix the projection)
# ======================================================================


def run_tdc(X, n_epochs, eta_h=10.0):
    """TDC algorithm (Sutton et al. 2009). Two-timescale expected updates.

    w update: alpha * [delta * x(s) - gamma * x(6) * (x(s)^T h)]
    h update: beta  * [delta - x(s)^T h] * x(s)
    where beta = alpha * eta_h (h learns faster).

    Note: TDC stabilizes the learning (no divergence) but converges to a
    biased fixed point in this off-policy setting. The uniform weighting
    fixes the action distribution but not the state distribution mismatch,
    so the value function remains bounded but nonzero.
    """
    w = make_w0()
    h = np.zeros(N_WEIGHTS)
    x_hub = X[5]
    beta = ALPHA * eta_h
    history = [w.copy()]
    for _ in range(n_epochs):
        dw = np.zeros(N_WEIGHTS)
        dh = np.zeros(N_WEIGHTS)
        for s in range(N_STATES):
            delta = GAMMA * x_hub @ w - X[s] @ w
            dw += (1.0 / N_STATES) * (delta * X[s] - GAMMA * x_hub * (X[s] @ h))
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
    max real eigenvalue flips from +0.07 to -0.93, giving strong
    contraction. Converges to w=0 (unique fixed point since A - eta*I
    is invertible).
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

if __name__ == "__main__":
    X = make_features()
    A = make_A(X)

    # --- Eigenvalue analysis ---
    eigvals = np.linalg.eigvals(A)
    real_parts = sorted(eigvals.real, reverse=True)

    print("=" * 65)
    print("Baird's Counterexample: Divergence and Three Fixes")
    print("Six-state star MDP (Baird 1995, Figure 1)")
    print("=" * 65)
    print(f"\nParameters: gamma={GAMMA}, alpha={ALPHA}, epochs={N_EPOCHS}")
    print(f"Initial weights: w_0..w_5 = 1, w_6 = {make_w0()[6]:.1f}")

    print("\nExpected update matrix A — eigenvalues (real parts):")
    for i, ev in enumerate(real_parts):
        tag = " <-- unstable" if ev > 1e-8 else ""
        print(f"  lambda_{i + 1} = {ev:+.4f}{tag}")

    rho = max(abs(1 + ALPHA * eigvals))
    print(f"\nSpectral radius of (I + alpha*A): {rho:.6f}")
    print(
        f"System {'diverges' if rho > 1 else 'converges'} "
        f"(rho {'>' if rho > 1 else '<='} 1)"
    )

    # Fitted VI spectral radius
    x_hub = X[5]
    X_pinv = np.linalg.pinv(X)
    M_fvi = GAMMA * np.outer(X_pinv @ np.ones(N_STATES), x_hub)
    print("\nFitted VI iteration matrix:")
    print(
        f"  Spectral radius: {max(abs(np.linalg.eigvals(M_fvi))):.4f} "
        f"(= gamma = {GAMMA})"
    )

    # Regularized eigenvalues
    A_reg = A - 1.0 * np.eye(N_WEIGHTS)
    eigvals_reg = np.linalg.eigvals(A_reg)
    print("\nWith L2 regularization (eta=1.0):")
    print(f"  Max real eigenvalue of (A - eta*I): {max(eigvals_reg.real):+.4f}")
    print(
        f"  Spectral radius of (I + alpha*(A-eta*I)): "
        f"{max(abs(1 + ALPHA * eigvals_reg)):.6f}"
    )

    # --- Run simulations ---
    hist_sg = run_semigradient(X, N_EPOCHS)
    hist_fvi = run_fitted_vi(X, N_EPOCHS)
    hist_tdc = run_tdc(X, N_EPOCHS, eta_h=10.0)
    hist_reg = run_regularized(X, N_EPOCHS, eta_reg=1.0)

    # Extract shared-weight w_0 and max|V(s)| traces
    w0_sg = np.array([w[0] for w in hist_sg])
    w0_fvi = np.array([w[0] for w in hist_fvi])
    w0_tdc = np.array([w[0] for w in hist_tdc])
    w0_reg = np.array([w[0] for w in hist_reg])

    maxV_sg = compute_max_V(X, hist_sg)
    maxV_fvi = compute_max_V(X, hist_fvi)
    maxV_tdc = compute_max_V(X, hist_tdc)
    maxV_reg = compute_max_V(X, hist_reg)

    # --- Summary tables ---
    print("\n--- shared weight w_0 trajectory ---")
    print(
        f"{'Method':<25} {'w_0(0)':>8} {'w_0(100)':>10} "
        f"{'w_0(500)':>10} {'w_0(1000)':>11}"
    )
    print("-" * 68)
    for name, trace in [
        ("Semi-gradient TD", w0_sg),
        ("Fitted VI", w0_fvi),
        ("TDC (Gradient TD)", w0_tdc),
        ("L2 regularization", w0_reg),
    ]:
        v = [trace[0], trace[100], trace[500], trace[-1]]
        print(f"{name:<25} {v[0]:>8.2f} {v[1]:>10.2f} {v[2]:>10.2f} {v[3]:>11.2f}")

    print("\n--- max|V(s)| trajectory ---")
    print(f"{'Method':<25} {'t=0':>8} {'t=100':>10} {'t=500':>10} {'t=1000':>11}")
    print("-" * 68)
    for name, trace in [
        ("Semi-gradient TD", maxV_sg),
        ("Fitted VI", maxV_fvi),
        ("TDC (Gradient TD)", maxV_tdc),
        ("L2 regularization", maxV_reg),
    ]:
        v = [trace[0], trace[100], trace[500], trace[-1]]
        print(f"{name:<25} {v[0]:>8.2f} {v[1]:>10.2f} {v[2]:>10.2f} {v[3]:>11.2f}")

    # --- Hand-derived verification anchors ---
    # V(1) = w_0 + 2 w_1 = 1 + 2 = 3;  V(6) = 2 w_0 + w_6 = 2 + 10 = 12.
    # delta(spoke) = gamma*V(6) - V(s) = 0.99*12 - 3 = 8.88
    # delta(hub)   = gamma*V(6) - V(6) = 11.88 - 12  = -0.12
    # dw_0 = alpha * (1/6) * [5 * delta(spoke) * 1 + delta(hub) * 2]
    #      = 0.01 * (44.4 - 0.24) / 6 = 0.0736  =>  w_0(1) = 1.0736
    w0 = make_w0()
    V1 = X[0] @ w0
    V6 = X[5] @ w0
    d_spoke = GAMMA * V6 - V1
    d_hub = GAMMA * V6 - V6
    dw0_hand = ALPHA * (5 * d_spoke * 1.0 + d_hub * 2.0) / N_STATES
    print("\nVerification (hand-derived, see comments):")
    print(f"  V(1) at epoch 0:  {V1:.2f}  (hand: 3.00)")
    print(f"  V(6) at epoch 0:  {V6:.2f}  (hand: 12.00)")
    print(f"  delta(1) epoch 0: {d_spoke:.2f}  (hand: 8.88)")
    print(f"  delta(6) epoch 0: {d_hub:.2f}  (hand: -0.12)")
    print(f"  w_0 at epoch 1:   {w0_sg[1]:.4f}  (hand: {1 + dw0_hand:.4f})")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = np.arange(N_EPOCHS + 1)

    panels = [
        (axes[0, 0], maxV_sg, COLORS["red"], "Semi-gradient off-policy TD (diverges)"),
        (axes[0, 1], maxV_fvi, COLORS["green"], "Fix 1: Fitted value iteration"),
        (axes[1, 0], maxV_tdc, COLORS["orange"], "Fix 2: TDC / Gradient TD"),
        (
            axes[1, 1],
            maxV_reg,
            COLORS["purple"],
            r"Fix 3: $\ell_2$ regularization ($\eta = 1.0$)",
        ),
    ]

    for ax, trace, color, title in panels:
        ax.plot(epochs, trace, color=color)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\max_s |V(s)|$")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Baird's six-state star counterexample ($\\gamma=0.99$, $\\alpha=0.01$)",
        fontsize=14,
    )
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "bairds_counterexample.png")
    fig.savefig(out_path)
    print(f"\nFigure saved: {out_path}")
