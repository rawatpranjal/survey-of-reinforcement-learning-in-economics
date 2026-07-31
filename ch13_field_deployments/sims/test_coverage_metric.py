# The coverage metric must describe the SOFTENED candidates OPS actually scores. The
# softened no-promo policy settles the reference discount near eps * mean(DISCOUNTS)
# = 0.3 * 0.125 = 0.0375, not 0, so the gap computed for the pure policy overstates.
import numpy as np

import field_ope_reliability as drv


def test_reference_coverage_v2_targets_softened_candidates():
    rc = drv.compute_reference_coverage()
    # softened no-promo settles near eps*mean(D): its median r must sit in a band
    # strictly above 0 (pure no-promo would pile up at ~0) and below r_init
    med = float(np.median(rc["cand_r"]["no_promo"]))
    assert 0.02 < med < 0.08, f"softened no-promo median r {med:.4f}"
    # gaps are per (log, candidate) fractions
    for lg in ("ab", "incumbent", "mixture"):
        g = rc["gaps"][lg]["no_promo"]
        assert 0.0 <= g <= 1.0
    # the mixture log must cover the softened best candidate's states far better
    # than the A/B log (its const-0 stratum lives exactly there)
    assert rc["gaps"]["mixture"]["no_promo"] < rc["gaps"]["ab"]["no_promo"] - 0.2, (
        f"mixture gap {rc['gaps']['mixture']['no_promo']:.3f} not clearly below "
        f"A/B gap {rc['gaps']['ab']['no_promo']:.3f}"
    )
    # and the mixture log's 1st percentile reaches below the softened no-promo median
    assert rc["log_1pct"]["mixture"] < med
