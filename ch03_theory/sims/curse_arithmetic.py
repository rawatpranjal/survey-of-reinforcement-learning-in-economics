"""Numerical illustrations of the curse-of-dimensionality bounds.

Chapter 3 - The Theory of Reinforcement Learning (curse-of-dimensionality section).

Evaluates four published bounds at concrete parameter values, so that every number
appearing in the prose is produced here rather than typed by hand. There is no Monte
Carlo and no randomness: each quantity is a closed-form evaluation of the bound as
stated in its source, and the source page is cited beside each formula below.

  1. Grid dynamic programming: |S| = n^d, sweep cost n^{2d}|A|.
  2. Chow and Tsitsiklis (1989), J. Complexity 5:466-488. Oracle-query complexity:
     Theorem 3.1 (p. 473) under mixing, Theorems 3.2/3.3 (pp. 480, 485) without it,
     matched by the upper bounds Eqs. (2.6)-(2.8) (p. 472), so all three are tight.
  3. Du et al. (2021), Corollary 5.1 (p. 21): trajectory count for a Bilinear Class
     with a finite hypothesis class.
  4. Liu et al. (2022), Theorem 1 (p. 7): regret exponent, prescribed network width,
     and the admissibility condition alpha > d(1/p - 1/4)_+.
  5. Lu et al. (2025), Theorem 5.1 (pp. 12-13): scope-sum sample requirement and the
     misspecification bias floor E_omega, which no amount of data removes.

Emits four LaTeX tables and one figure. Deterministic, so no cache: --data-only is a
no-op and --plots-only runs normally.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_SINGLE

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.dirname(__file__)

# One second of a single core, used to turn operation counts into wall-clock.
FLOPS_PER_SECOND = 1e9
BYTES_PER_VALUE = 8  # float64 value table

# Panel 1: models that appear elsewhere in this survey, plus their grid sizes.
# `states` overrides n**d when a model is not a uniform product grid.
GRID_MODELS = [
    dict(name=r"Bus engine, one bus \citep{Rust1987}", d=1, n=175, actions=2),
    dict(name="Bus engine, fleet of 3", d=3, n=175, actions=2**3),
    dict(name="Bus engine, fleet of 5", d=5, n=175, actions=2**5),
    dict(name="Brock--Mirman growth", d=1, n=500, actions=500, states=1000),
    dict(name="Wind farm, base state", d=3, n=7, actions=11),
    dict(name="Wind farm, six state variables", d=6, n=7, actions=11),
    dict(name="Inventory, five products", d=5, n=20, actions=6**5),
    dict(name="Macro model, ten state variables", d=10, n=10, actions=10),
]

# Panel 2: Chow-Tsitsiklis. Exponent is 2*d_s + d_a.
CT_DIMENSIONS = [(1, 1), (2, 1), (3, 2)]
CT_GAMMAS = [0.9, 0.95, 0.99]
CT_EPSILON = 0.01

# Panel 3: Du et al. Corollary 5.1.
DU_SETTINGS = [
    dict(d=5, H=10, eps=0.1),
    dict(d=10, H=24, eps=0.1),  # H = 24 is the wind farm's own horizon
    dict(d=10, H=24, eps=0.05),
    dict(d=20, H=50, eps=0.1),
]
DU_LOG_HYPOTHESES = 1e6  # |H|, the hypothesis-class cardinality
DU_DELTA = 0.05
DU_NORM_BOUND = 1.0  # B_X = B_W = 1

# Panel 4: Liu et al. Theorem 1. p = 2 gives the condition alpha > d/4.
LIU_BESOV_P = 2.0
LIU_DIMENSIONS = [1, 2, 3, 5, 10, 20, 50]
LIU_T = 1e6
LIU_K = 1  # benign, dense-reward case; K = H is the worst case
# Smoothness levels a modeller might plausibly claim: twice differentiable, and
# far smoother. A cell is empty when the level fails the admissibility condition.
LIU_ALPHAS = [2.0, 8.0]

# Panel 5: Lu et al. Theorem 5.1.
LU_COMPONENTS = 100  # n, the number of state components
LU_STATES_PER_COMPONENT = 10
LU_ACTIONS = 2
LU_SCOPE_SIZES = [1, 2, 3, 4, 6]  # k, the maximum parent-set size
LU_GAMMAS = [0.9, 0.95, 0.99]
LU_COUPLING = [1e-4, 1e-3, 1e-2, 1e-1]
LU_TARGET_EPSILON = 0.01

# Figure: exact DP against the escape routes, as ambient dimension grows.
FIG_DIMENSIONS = list(range(1, 13))
FIG_N = 10
FIG_ACTIONS = 10
FIG_GAMMA = 0.95
FIG_EPSILON = 0.01

# Cross-check against the wind-farm simulation's measured DP scaling.
WIND_FARM_BINS = 7
WIND_FARM_STDOUT = os.path.join(OUTPUT_DIR, "wind_farm_curse_study_stdout.txt")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def sci_math(x, digits=1):
    """Scientific notation without the surrounding $, for use inside existing math."""
    if x == 0:
        return "0"
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / (10.0**exponent)
    # log10 of a float power such as (10^6)^5 can land just under the integer, which
    # would print 10.0 x 10^29 for 10^30. Renormalise before deciding the format.
    if round(mantissa, digits) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    if 0 <= exponent < 4:
        return f"{x:,.0f}" if x >= 1 else f"{x:.{digits}f}"
    if round(mantissa, digits) == 1.0:
        return f"10^{{{exponent}}}"
    return f"{mantissa:.{digits}f} \\times 10^{{{exponent}}}"


def sci(x, digits=1):
    """LaTeX scientific notation as a standalone cell. Exact for ints, so huge counts
    do not lose precision. Small magnitudes stay in plain digits and are not wrapped
    in math mode."""
    body = sci_math(x, digits)
    if "^" not in body:
        return body
    return f"${body}$"


def human_time(seconds):
    """Wall-clock in the largest unit that keeps the number readable."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} $\\mu$s"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    if seconds < 3.15e9:
        return f"{seconds / 86400:.0f} days"
    return f"{sci(seconds / 3.156e7)} yr"


def human_bytes(nbytes):
    for unit, scale in [
        ("EB", 1e18),
        ("PB", 1e15),
        ("TB", 1e12),
        ("GB", 1e9),
        ("MB", 1e6),
        ("kB", 1e3),
    ]:
        if nbytes >= scale:
            value = nbytes / scale
            return f"{value:.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"
    if nbytes >= 1e12:
        return sci(nbytes) + " B"
    return f"{nbytes:.0f} B"


def big_bytes(nbytes):
    """Bytes, falling back to scientific notation past exabytes."""
    if nbytes >= 1e21:
        return sci(nbytes) + " B"
    return human_bytes(nbytes)


# ---------------------------------------------------------------------------
# The bounds
# ---------------------------------------------------------------------------


def read_wind_farm_times(path):
    """Pull the completed tabular-DP solve times out of the wind-farm sim's own stdout.

    That file has a `dim states time_s return` block; the d=3 and d=4 rows are the two
    runs that finished inside the budget. Parsing them keeps this cross-check tied to
    the artifact rather than to a number retyped from the prose.
    """
    times = {}
    in_block = False
    with open(path) as f:
        for line in f:
            fields = line.split()
            if fields[:4] == ["dim", "states", "time_s", "return"]:
                in_block = True
                continue
            if in_block:
                if not fields or not fields[0].isdigit():
                    break
                times[int(fields[0])] = float(fields[2])
    missing = {3, 4} - set(times)
    if missing:
        raise RuntimeError(
            f"{path}: no DP timing row for dimension(s) {sorted(missing)}"
        )
    return dict(t3=times[3], t4=times[4])


def grid_dp_cost(model):
    """|S| = n^d and the n^{2d}|A| sweep cost of one value-iteration pass."""
    states = model.get("states", model["n"] ** model["d"])
    sweep_ops = states * states * model["actions"]
    return dict(
        name=model["name"],
        d=model["d"],
        n=model["n"],
        actions=model["actions"],
        states=states,
        table_bytes=states * BYTES_PER_VALUE,
        sweep_ops=sweep_ops,
        sweep_seconds=sweep_ops / FLOPS_PER_SECOND,
    )


def chow_tsitsiklis(gamma, epsilon, d_s, d_a):
    """Chow-Tsitsiklis (1989).

    These are ORACLE QUERY counts, which is what C_mix, C_prob and C_sub are defined
    to be on p. 472. Upper bounds Eqs. (2.6)-(2.8) (p. 472) and the matching lower
    bounds of Theorems 3.1, 3.2 and 3.3 (pp. 473, 480, 485) make all three tight:
    Theta(((1-gamma) eps)^-(2 d_s + d_a)) under the mixing condition, and
    Theta(((1-gamma)^2 eps)^-(2 d_s + d_a)) without it.

    Arithmetic-operation counts are a separate matter, credited on p. 487 to the
    companion multigrid paper (LIDS-P-1864); there the no-mixing upper bound carries
    an extra factor 1/(1-gamma), which `general_upper` records.
    """
    exponent = 2 * d_s + d_a
    mixing = (1.0 / ((1.0 - gamma) * epsilon)) ** exponent
    general = (1.0 / (((1.0 - gamma) ** 2) * epsilon)) ** exponent
    return dict(
        d_s=d_s,
        d_a=d_a,
        gamma=gamma,
        epsilon=epsilon,
        exponent=exponent,
        mixing=mixing,
        general_lower=general,
        general_upper=general / (1.0 - gamma),
        price_of_no_mixing=general / mixing,  # equals (1-gamma)^-exponent
    )


def du_trajectories(d, H, eps, log_cardinality, delta, norm_bound):
    """Du et al. (2021), Corollary 5.1 (p. 21), with the absolute constant set to one.

    d^2 H^7 ln(d H^2) ln|H| ln(1/delta) / eps^2
        * ln^2( d H B_X B_W ln|H| ln(1/delta) / eps )
    """
    ln_h = math.log(log_cardinality)
    ln_delta = math.log(1.0 / delta)
    leading = (d**2) * (H**7) * math.log(d * H**2) * ln_h * ln_delta / (eps**2)
    inner = d * H * norm_bound * norm_bound * ln_h * ln_delta / eps
    trajectories = leading * (math.log(inner) ** 2)
    return dict(
        d=d,
        H=H,
        eps=eps,
        horizon_factor=float(H) ** 7,
        trajectories=trajectories,
        env_steps=trajectories * H,
    )


def liu_regret_exponent(alpha, d, K):
    """Liu et al. (2022), Theorem 1 (p. 7): the exponent of T in the regret bound."""
    if math.isinf(alpha):
        return (K + 1.0) / (K + 2.0)
    return (alpha * K + (alpha + d) * (K + 2.0)) / ((2.0 * alpha + d) * (K + 2.0))


def liu_width(alpha, d, T):
    """Liu et al. (2022), Eq. (7): m  ~  (d/(2 alpha + d)) T^{d/(2 alpha + d)} log T."""
    if math.isinf(alpha):
        return 0.0
    rate = d / (2.0 * alpha + d)
    return rate * (T**rate) * math.log(T)


def liu_min_smoothness(d, p):
    """The admissibility condition alpha > d(1/p - 1/4)_+ of Theorem 1."""
    return d * max(0.0, 1.0 / p - 0.25)


def liu_row(d, p, T, K, alphas):
    """One ambient dimension, evaluated at each candidate smoothness level.

    `admissible` is the condition alpha > d(1/p - 1/4)_+ of Theorem 1. When it fails
    the theorem says nothing at all, so exponent and width are reported as None rather
    than as a number the bound does not license.
    """
    alpha_min = liu_min_smoothness(d, p)
    cells = []
    for alpha in alphas:
        ok = alpha > alpha_min
        cells.append(
            dict(
                alpha=alpha,
                admissible=ok,
                exponent=liu_regret_exponent(alpha, d, K) if ok else None,
                width=liu_width(alpha, d, T) if ok else None,
            )
        )
    return dict(
        d=d,
        alpha_min=alpha_min,
        cells=cells,
        exponent_limit=liu_regret_exponent(math.inf, d, K),
    )


def lu_scopes(k, n_components, states_per_component, actions):
    """Lu et al. (2025): scope-set cardinalities against the full state-action count.

    A component whose scope covers k components contributes m^k * |A| configurations.
    The sample requirement Eq. (10) charges the sum over the kappa_p largest scopes,
    against |S||A| = m^n |A| for the unstructured minimax rate.
    """
    scope = (states_per_component**k) * actions
    return dict(
        k=k,
        scope=scope,
        scope_sum=n_components * scope,
        global_size=(states_per_component**n_components) * actions,
    )


def lu_bias_floor(gamma, delta_p, delta_r):
    """Lu et al. (2025), Theorem 5.1: E_omega = gamma (1-gamma)^-2 dP + (1-gamma)^-1 dR.

    The accuracy guarantee is ||Qhat - Q*||_inf <= eps + E_omega, so E_omega is a floor
    that additional samples never lower.
    """
    return gamma / ((1.0 - gamma) ** 2) * delta_p + delta_r / (1.0 - gamma)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_data(force=None):
    """All numbers. No plotting, no file writes."""
    grid = [grid_dp_cost(m) for m in GRID_MODELS]

    ct = [
        chow_tsitsiklis(g, CT_EPSILON, d_s, d_a)
        for (d_s, d_a) in CT_DIMENSIONS
        for g in CT_GAMMAS
    ]

    du = [
        du_trajectories(
            s["d"], s["H"], s["eps"], DU_LOG_HYPOTHESES, DU_DELTA, DU_NORM_BOUND
        )
        for s in DU_SETTINGS
    ]

    liu = [liu_row(d, LIU_BESOV_P, LIU_T, LIU_K, LIU_ALPHAS) for d in LIU_DIMENSIONS]

    lu_scope = [
        lu_scopes(k, LU_COMPONENTS, LU_STATES_PER_COMPONENT, LU_ACTIONS)
        for k in LU_SCOPE_SIZES
    ]
    lu_bias = [
        dict(
            gamma=g,
            floors=[lu_bias_floor(g, dp, 0.0) for dp in LU_COUPLING],
            # coupling error at which the floor equals the target accuracy
            breakeven=LU_TARGET_EPSILON * (1.0 - g) ** 2 / g,
        )
        for g in LU_GAMMAS
    ]

    # Figure series: everything as operations (or samples) against ambient dimension.
    exact = [(FIG_N**d) ** 2 * FIG_ACTIONS for d in FIG_DIMENSIONS]
    ct_curve = [
        chow_tsitsiklis(FIG_GAMMA, FIG_EPSILON, d, 1)["general_lower"]
        for d in FIG_DIMENSIONS
    ]
    du_flat = du_trajectories(10, 24, 0.1, DU_LOG_HYPOTHESES, DU_DELTA, DU_NORM_BOUND)[
        "trajectories"
    ]
    lu_flat = lu_scopes(3, LU_COMPONENTS, LU_STATES_PER_COMPONENT, LU_ACTIONS)[
        "scope_sum"
    ] / (FIG_EPSILON**2 * (1.0 - FIG_GAMMA) ** 3)

    # Cross-check: grid DP predicts n^2 per added dimension at the wind farm's binning.
    # The measured side is parsed from that simulation's own saved stdout, not retyped.
    wf = read_wind_farm_times(WIND_FARM_STDOUT)
    cross_check = dict(
        predicted_growth=float(WIND_FARM_BINS**2),
        measured_growth=wf["t4"] / wf["t3"],
        seconds_d3=wf["t3"],
        seconds_d4=wf["t4"],
    )

    return dict(
        grid=grid,
        ct=ct,
        du=du,
        liu=liu,
        lu_scope=lu_scope,
        lu_bias=lu_bias,
        figure=dict(
            dims=FIG_DIMENSIONS,
            exact=exact,
            ct=ct_curve,
            du_flat=du_flat,
            lu_flat=lu_flat,
        ),
        cross_check=cross_check,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Table saved: {path}")


def _table_grid_dp(data):
    rows = data["grid"]
    path = os.path.join(OUTPUT_DIR, "curse_grid_dp.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cost of one exact value-iteration sweep on a uniform grid, for models "
        r"used elsewhere in this survey. States are $n^d$ over $d$ state variables at $n$ "
        r"points each, one sweep costs $n^{2d}|\mathcal{A}|$ operations, and wall-clock "
        r"assumes " + sci(FLOPS_PER_SECOND) + r" operations per second. Memory is one "
        r"float64 value per state. Rows ordered by state count.}",
        r"\label{tab:curse_grid_dp}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $d$ & $n$ & $|\mathcal{A}|$ & $|\mathcal{S}|$ & Value table & One sweep \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda x: x["states"]):
        lines.append(
            f"{r['name']} & {r['d']} & {r['n']} & {sci(r['actions'])} & "
            f"{sci(r['states'])} & {big_bytes(r['table_bytes'])} & "
            f"{human_time(r['sweep_seconds'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, lines)


def _table_chow_tsitsiklis(data):
    path = os.path.join(OUTPUT_DIR, "curse_chow_tsitsiklis.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Oracle queries required by Theorem~\ref{thm:chow_tsitsiklis} at "
        r"$\epsilon = " + f"{CT_EPSILON}" + r"$. The mixing column is "
        r"\eqref{eq:ct_mixing} and the general column is \eqref{eq:ct_general}; both are "
        r"tight, so the gap between them is a real difference in problem difficulty and "
        r"not slack in the analysis. The last column is their ratio, "
        r"$(1-\gamma)^{-(2d_s+d_a)}$, the price of losing the mixing condition.}",
        r"\label{tab:curse_chow_tsitsiklis}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$d_s$ & $d_a$ & $\gamma$ & $2d_s + d_a$ & Mixing & General & Ratio \\",
        r"\midrule",
    ]
    for r in data["ct"]:
        lines.append(
            f"{r['d_s']} & {r['d_a']} & {r['gamma']} & {r['exponent']} & "
            f"{sci(r['mixing'])} & {sci(r['general_lower'])} & "
            f"{sci(r['price_of_no_mixing'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, lines)


def _table_sample_complexity(data):
    path = os.path.join(OUTPUT_DIR, "curse_sample_complexity.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Trajectories required by \eqref{eq:bilin_samples}, with the absolute "
        r"constant set to one, at hypothesis-class cardinality "
        r"$|\mathcal{H}| = " + sci_math(DU_LOG_HYPOTHESES) + r"$, failure probability "
        r"$\delta = " + f"{DU_DELTA}" + r"$ and $B_X = B_W = 1$. The horizon column "
        r"isolates the factor $H^7$. Environment steps are trajectories times $H$. The "
        r"inherent dimension $d$ is the rank of the bilinear factorization, not the "
        r"ambient state dimension.}",
        r"\label{tab:curse_sample_complexity}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$d$ & $H$ & $\epsilon$ & $H^7$ & Trajectories & Environment steps \\",
        r"\midrule",
    ]
    for r in data["du"]:
        lines.append(
            f"{r['d']} & {r['H']} & {r['eps']} & {sci(r['horizon_factor'])} & "
            f"{sci(r['trajectories'])} & {sci(r['env_steps'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, lines)


def _table_smoothness(data):
    path = os.path.join(OUTPUT_DIR, "curse_smoothness.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{What Theorem~\ref{thm:deep_rl_regret} asks of the problem, at Besov "
        r"index $p = "
        + f"{LIU_BESOV_P:.0f}"
        + r"$, benign exploration $K = "
        + f"{LIU_K}"
        + r"$ and $T = "
        + sci_math(LIU_T)
        + r"$ episodes. The admissibility condition $\alpha > d(1/p - 1/4)_+$ demands a "
        r"minimum smoothness that grows linearly in the ambient dimension, reported in the "
        r"second column. Each smoothness level then gives the regret exponent and the width "
        r"\eqref{eq:liu_architecture} the theorem prescribes; n/a marks a level that "
        r"fails the condition, where the theorem gives no guarantee at all. Regret is "
        r"sublinear in every admissible cell, but the exponent approaches one as the "
        r"smoothness-to-dimension ratio falls.}",
        r"\label{tab:curse_smoothness}",
        r"\begin{tabular}{rr" + "rr" * len(LIU_ALPHAS) + r"}",
        r"\toprule",
        r" & & "
        + " & ".join(
            r"\multicolumn{2}{c}{$\alpha = " + f"{a:.0f}" + r"$}" for a in LIU_ALPHAS
        )
        + r" \\",
        # Rules under each group header so the two-column spans read unambiguously.
        " ".join(
            r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}"
            for i in range(len(LIU_ALPHAS))
        ),
        r"$d$ & Required $\alpha$ & "
        + " & ".join(r"Exponent & Width $m$" for _ in LIU_ALPHAS)
        + r" \\",
        r"\midrule",
    ]
    for r in data["liu"]:
        cells = []
        for c in r["cells"]:
            if c["admissible"]:
                cells.append(f"{c['exponent']:.3f} & {sci(c['width'])}")
            else:
                cells.append("n/a & n/a")
        lines.append(f"{r['d']} & {r['alpha_min']:.2f} & " + " & ".join(cells) + r" \\")
    limit = data["liu"][0]["exponent_limit"]
    ncols = 2 + 2 * len(LIU_ALPHAS)
    lines += [
        r"\midrule",
        r"\multicolumn{"
        + str(ncols)
        + r"}{l}{As $\alpha \to \infty$ the exponent falls to "
        + f"{limit:.3f}"
        + r" for every $d$, and the width requirement vanishes.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _write(path, lines)


def _table_factored(data):
    path = os.path.join(OUTPUT_DIR, "curse_factored.tex")
    glob = data["lu_scope"][0]["global_size"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Theorem~\ref{thm:factored_complexity} in two parts, for a system of "
        + f"{LU_COMPONENTS}"
        + r" components with "
        + f"{LU_STATES_PER_COMPONENT}"
        + r" states each and $|\mathcal{A}| = "
        + f"{LU_ACTIONS}"
        + r"$, so the unstructured problem has $|\mathcal{S}||\mathcal{A}| = "
        + sci_math(glob)
        + r"$. Panel (a) is what the factorization buys: the scope sum that "
        r"enters \eqref{eq:lu_samples} in place of $|\mathcal{S}||\mathcal{A}|$, as the "
        r"maximum parent-set size $k$ grows. Panel (b) is what it costs: the bias floor "
        r"$\mathcal{E}_\omega$ of \eqref{eq:lu_bias} at $\Delta_R = 0$, which bounds "
        r"accuracy from below however many samples are drawn. Cells exceeding a target "
        r"$\epsilon = " + f"{LU_TARGET_EPSILON}" + r"$ are the settings where the "
        r"target is unreachable.}",
        r"\label{tab:curse_factored}",
        r"\begin{tabular}{rrr}",
        r"\multicolumn{3}{l}{(a) Scope sizes against the full state-action count} \\",
        r"\toprule",
        r"Parent-set size $k$ & One scope & Scope sum \\",
        r"\midrule",
    ]
    for r in data["lu_scope"]:
        lines.append(f"{r['k']} & {sci(r['scope'])} & {sci(r['scope_sum'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{1em}",
        "",
        r"\begin{tabular}{r" + "r" * len(LU_COUPLING) + r"r}",
        r"\multicolumn{"
        + str(len(LU_COUPLING) + 2)
        + r"}{l}{(b) Misspecification bias floor $\mathcal{E}_\omega$ by coupling error} \\",
        r"\toprule",
        r"$\gamma$ & "
        + " & ".join(r"$\Delta_P = " + sci_math(dp) + "$" for dp in LU_COUPLING)
        + r" & Break-even $\Delta_P$ \\",
        r"\midrule",
    ]
    for r in data["lu_bias"]:
        cells = " & ".join(f"{v:.3g}" for v in r["floors"])
        lines.append(f"{r['gamma']} & {cells} & {sci(r['breakeven'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, lines)


def _figure(data):
    fig_data = data["figure"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    dims = fig_data["dims"]

    ax.semilogy(
        dims,
        fig_data["exact"],
        marker="o",
        color=COLORS["red"],
        label=f"Exact grid DP, $n^{{2d}}|\\mathcal{{A}}|$ at $n={FIG_N}$",
    )
    ax.semilogy(
        dims,
        fig_data["ct"],
        marker="s",
        color=COLORS["orange"],
        label=f"Chow-Tsitsiklis lower bound, $\\gamma={FIG_GAMMA}$",
    )
    ax.semilogy(
        dims,
        [fig_data["du_flat"]] * len(dims),
        marker="^",
        color=COLORS["blue"],
        label="Bilinear class, $d=10$, $H=24$ (flat in ambient $d$)",
    )
    ax.semilogy(
        dims,
        [fig_data["lu_flat"]] * len(dims),
        marker="v",
        color=COLORS["green"],
        label="Approximate factorization, $k=3$ (flat in ambient $d$)",
    )

    for seconds, tag in [
        (1.0, "one second"),
        (86400.0, "one day"),
        (3.156e7, "one year"),
    ]:
        y = seconds * FLOPS_PER_SECOND
        ax.axhline(y, **BENCH_STYLE)
        ax.text(
            dims[-1],
            y * 1.6,
            tag,
            ha="right",
            va="bottom",
            fontsize=8,
            color=BENCH_STYLE["color"],
        )

    ax.set_xlabel("Ambient state dimension $d$")
    ax.set_ylabel(f"Operations or samples ($\\epsilon = {FIG_EPSILON}$)")
    ax.set_title(
        "Cost of exact dynamic programming against the structural escape routes"
    )
    ax.set_ylim(1e2, 1e45)
    ax.legend(loc="upper left", framealpha=0.9)

    path = os.path.join(OUTPUT_DIR, "curse_arithmetic.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


def _report(data):
    """Console tables. Facts only, one line per configuration."""
    print("\nGRID DYNAMIC PROGRAMMING")
    print(
        f"{'Model':42s} {'d':>3s} {'|S|':>12s} {'table':>12s} {'sweep ops':>12s} {'wall clock':>14s}"
    )
    for r in sorted(data["grid"], key=lambda x: x["states"]):
        name = r["name"].replace(r"\citep{Rust1987}", "(Rust 1987)")
        print(
            f"{name:42s} {r['d']:3d} {r['states']:12.3e} "
            f"{r['table_bytes']:12.3e} {r['sweep_ops']:12.3e} "
            f"{r['sweep_seconds']:14.3e}"
        )

    print("\nCHOW-TSITSIKLIS (1989), eps = %g" % CT_EPSILON)
    print(
        f"{'d_s':>4s} {'d_a':>4s} {'gamma':>6s} {'exp':>4s} {'mixing':>12s} "
        f"{'general':>12s} {'ratio':>12s}"
    )
    for r in data["ct"]:
        print(
            f"{r['d_s']:4d} {r['d_a']:4d} {r['gamma']:6.2f} {r['exponent']:4d} "
            f"{r['mixing']:12.3e} {r['general_lower']:12.3e} "
            f"{r['price_of_no_mixing']:12.3e}"
        )

    print("\nDU ET AL. (2021) COROLLARY 5.1")
    print(
        f"{'d':>4s} {'H':>4s} {'eps':>6s} {'H^7':>12s} {'trajectories':>14s} {'env steps':>14s}"
    )
    for r in data["du"]:
        print(
            f"{r['d']:4d} {r['H']:4d} {r['eps']:6.3f} {r['horizon_factor']:12.3e} "
            f"{r['trajectories']:14.3e} {r['env_steps']:14.3e}"
        )

    print(
        "\nLIU ET AL. (2022) THEOREM 1, p = %g, K = %d, T = %g"
        % (LIU_BESOV_P, LIU_K, LIU_T)
    )
    header = "".join(
        f"{('a=%g exp' % a):>11s}{('a=%g width' % a):>13s}" for a in LIU_ALPHAS
    )
    print(f"{'d':>4s} {'alpha_min':>10s}{header}")
    for r in data["liu"]:
        cells = ""
        for c in r["cells"]:
            if c["admissible"]:
                cells += f"{c['exponent']:11.4f}{c['width']:13.3e}"
            else:
                cells += f"{'inadmiss.':>11s}{'-':>13s}"
        print(f"{r['d']:4d} {r['alpha_min']:10.3f}{cells}")
    print(f"  exponent as alpha -> infinity: {data['liu'][0]['exponent_limit']:.4f}")

    print("\nLU ET AL. (2025) THEOREM 5.1, SCOPE SIZES")
    print(f"{'k':>4s} {'one scope':>12s} {'scope sum':>12s} {'|S||A|':>12s}")
    for r in data["lu_scope"]:
        print(
            f"{r['k']:4d} {r['scope']:12.3e} {r['scope_sum']:12.3e} "
            f"{r['global_size']:12.3e}"
        )

    print("\nLU ET AL. (2025) THEOREM 5.1, BIAS FLOOR (Delta_R = 0)")
    header = " ".join(f"{dp:>12.0e}" for dp in LU_COUPLING)
    print(f"{'gamma':>6s} {header} {'breakeven':>12s}")
    for r in data["lu_bias"]:
        cells = " ".join(f"{v:12.4g}" for v in r["floors"])
        print(f"{r['gamma']:6.2f} {cells} {r['breakeven']:12.3e}")
    print(f"  target accuracy eps = {LU_TARGET_EPSILON}")

    cc = data["cross_check"]
    print("\nCROSS-CHECK AGAINST THE WIND-FARM SIMULATION")
    print(
        f"  grid DP predicts a factor n^2 = {cc['predicted_growth']:.1f} per added dimension "
        f"at n = {WIND_FARM_BINS} bins"
    )
    print(
        f"  wind_farm_curse_study measured a factor {cc['measured_growth']:.2f} "
        f"({cc['seconds_d4']:.1f} s at d=4 over {cc['seconds_d3']:.1f} s at d=3, "
        f"parsed from wind_farm_curse_study_stdout.txt)"
    )
    print(
        f"  ratio measured/predicted = {cc['measured_growth'] / cc['predicted_growth']:.3f}; "
        f"the sim integrates with a fixed 20-sample Monte Carlo draw rather than a full "
        f"n^d sweep, so it does not pay the second factor of n^d"
    )


def generate_outputs(data):
    """Figures and LaTeX tables. No computation, no mutation of `data`."""
    _report(data)
    print()
    _table_grid_dp(data)
    _table_chow_tsitsiklis(data)
    _table_sample_complexity(data)
    _table_factored(data)
    _table_smoothness(data)
    _figure(data)


def main():
    parser = argparse.ArgumentParser(
        description="Numerical illustrations of the curse-of-dimensionality bounds"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="No computation to cache (closed-form arithmetic)",
    )
    parser.add_argument(
        "--plots-only", action="store_true", help="Runs normally (same as no flags)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CURSE OF DIMENSIONALITY: NUMERICAL ILLUSTRATIONS")
    print("=" * 70)

    if args.data_only:
        print(
            "No computation to cache (closed-form arithmetic). "
            "Use default or --plots-only."
        )
        return

    data = compute_data()
    generate_outputs(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
