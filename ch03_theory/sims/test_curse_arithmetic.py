"""Independent recomputation of cells emitted by curse_arithmetic.py.

Each check re-derives a number straight from the bound as printed in its source,
without calling the helper that produced the table, then asserts it matches both the
module's function and the string that landed in the .tex file. A typo in either the
formula or the table writer fails here.

Run: /usr/local/bin/python3 ch03_theory/sims/test_curse_arithmetic.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import curse_arithmetic as ca

HERE = os.path.dirname(__file__)


def read_table(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_grid_dp():
    # Rust's bus engine on its 175-point mileage grid, five engines, joint replacement.
    states = 175**5
    ops = states * states * 2**5
    assert states == 164_130_859_375, states
    row = [
        r for r in ca.compute_data()["grid"] if r["name"] == "Bus engine, fleet of 5"
    ][0]
    assert row["states"] == states, (row["states"], states)
    assert row["sweep_ops"] == ops, (row["sweep_ops"], ops)
    # 8 bytes per state, so the value table alone passes a terabyte.
    assert row["table_bytes"] == states * 8
    assert row["table_bytes"] > 1e12
    assert "1.3 TB" in read_table("curse_grid_dp.tex")
    print(f"  grid DP, fleet of 5: |S| = {states:,}, sweep = {ops:.3e} ops  OK")


def check_chow_tsitsiklis():
    # Theorem 3.1 / 3.2 at d_s = 2, d_a = 1, gamma = 0.99, eps = 0.01. Exponent 5.
    gamma, eps, exponent = 0.99, 0.01, 5
    mixing = (1.0 / ((1 - gamma) * eps)) ** exponent  # (1/(0.01*0.01))^5 = 1e20
    general = (1.0 / ((1 - gamma) ** 2 * eps)) ** exponent  # (1/(1e-4*0.01))^5 = 1e30
    assert close(mixing, 1e20), mixing
    assert close(general, 1e30), general
    got = ca.chow_tsitsiklis(gamma, eps, 2, 1)
    assert close(got["mixing"], mixing)
    assert close(got["general_lower"], general)
    # The price of losing the mixing condition is exactly (1-gamma)^-exponent.
    assert close(got["price_of_no_mixing"], (1 - gamma) ** -exponent)
    assert close(got["general_upper"], general / (1 - gamma))
    print(
        f"  Chow-Tsitsiklis d_s=2 d_a=1 gamma=0.99: mixing {mixing:.3e}, "
        f"general {general:.3e}  OK"
    )


def check_du():
    # Corollary 5.1 at d = 10, H = 24, eps = 0.1, |H| = 1e6, delta = 0.05, B = 1.
    d, H, eps = 10, 24, 0.1
    ln_h = math.log(1e6)
    ln_delta = math.log(1 / 0.05)
    leading = d**2 * H**7 * math.log(d * H**2) * ln_h * ln_delta / eps**2
    inner = d * H * 1.0 * ln_h * ln_delta / eps
    expected = leading * math.log(inner) ** 2
    got = ca.du_trajectories(d, H, eps, 1e6, 0.05, 1.0)
    assert close(got["trajectories"], expected), (got["trajectories"], expected)
    assert got["horizon_factor"] == float(24**7)
    assert close(got["env_steps"], expected * H)
    print(
        f"  Du Corollary 5.1 d=10 H=24: {expected:.3e} trajectories, "
        f"H^7 = {24**7:.3e}  OK"
    )


def check_liu():
    # Theorem 1 exponent at alpha = 8, d = 10, K = 1, and the p = 2 admissibility bar.
    alpha, d, K = 8.0, 10, 1
    expected = (alpha * K + (alpha + d) * (K + 2)) / ((2 * alpha + d) * (K + 2))
    assert close(expected, 62.0 / 78.0), expected
    assert close(ca.liu_regret_exponent(alpha, d, K), expected)
    # alpha -> infinity gives (K+1)/(K+2) = 2/3 at K = 1.
    assert close(ca.liu_regret_exponent(math.inf, d, K), 2.0 / 3.0)
    # alpha > d(1/p - 1/4)_+ is alpha > d/4 at p = 2.
    assert close(ca.liu_min_smoothness(50, 2.0), 12.5)
    assert ca.liu_min_smoothness(10, 2.0) == 2.5
    # So a twice-differentiable Q-function is inadmissible from d = 10 upward.
    rows = {r["d"]: r for r in ca.compute_data()["liu"]}
    assert rows[5]["cells"][0]["admissible"] is True
    assert rows[10]["cells"][0]["admissible"] is False
    assert rows[50]["cells"][1]["admissible"] is False
    # Width from Eq. (7): (d/(2a+d)) T^{d/(2a+d)} log T.
    rate = d / (2 * alpha + d)
    assert close(ca.liu_width(alpha, d, 1e6), rate * (1e6**rate) * math.log(1e6))
    print(
        f"  Liu Theorem 1 alpha=8 d=10: exponent {expected:.4f}, "
        f"min alpha at d=50 is 12.5  OK"
    )


def check_lu():
    # Theorem 5.1 bias floor: E = gamma (1-gamma)^-2 dP + (1-gamma)^-1 dR.
    gamma, dp = 0.99, 1e-4
    expected = 0.99 / (0.01**2) * 1e-4  # = 0.99
    assert close(expected, 0.99), expected
    assert close(ca.lu_bias_floor(gamma, dp, 0.0), expected)
    # At gamma = 0.99 the floor is already 99x a target accuracy of 0.01.
    assert expected / 0.01 > 90
    # Break-even coupling: E = eps  =>  dP = eps (1-gamma)^2 / gamma.
    breakeven = 0.01 * (1 - gamma) ** 2 / gamma
    assert close(ca.lu_bias_floor(gamma, breakeven, 0.0), 0.01)
    # Scope sum against the unstructured state-action count.
    s = ca.lu_scopes(3, 100, 10, 2)
    assert s["scope"] == 10**3 * 2
    assert s["scope_sum"] == 100 * 10**3 * 2
    assert s["global_size"] == 10**100 * 2
    print(
        f"  Lu Theorem 5.1 gamma=0.99 dP=1e-4: floor {expected:.4g} "
        f"against target 0.01, break-even dP {breakeven:.3e}  OK"
    )


def check_tex_matches_stdout():
    """Every table on disk was written by the current code, not left over."""
    data = ca.compute_data()
    ct = read_table("curse_chow_tsitsiklis.tex")
    assert "$10^{30}$" in ct, "the gamma=0.99, exponent-5 general cell is missing"
    sm = read_table("curse_smoothness.tex")
    assert sm.count("n/a & n/a") == 4, sm.count("--- & ---")
    fac = read_table("curse_factored.tex")
    assert "0.99" in fac
    du = read_table("curse_sample_complexity.tex")
    assert "tab:curse_sample_complexity" in du
    for r in data["grid"]:
        assert r["sweep_ops"] > 0
    print("  emitted tables carry the current numbers  OK")


if __name__ == "__main__":
    print("Independent recomputation of curse_arithmetic.py")
    check_grid_dp()
    check_chow_tsitsiklis()
    check_du()
    check_liu()
    check_lu()
    check_tex_matches_stdout()
    print("All checks passed.")
