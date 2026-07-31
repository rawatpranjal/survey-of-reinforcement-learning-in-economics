# The clone-fidelity metric: a BC clone of a trivially learnable constant policy must
# match its reference argmax essentially everywhere, and the metric must be computed on
# states that spread the reference-discount axis (uniform-policy occupancy).

from promo_env import PromoConfig, constant_batch
from pipeline import _collect_states_actions, _built_bc, clone_fidelity


CFG = PromoConfig(K=8)


def test_constant_clone_reaches_full_fidelity():
    pol = constant_batch(2)
    obs, act, term = _collect_states_actions(CFG, pol, 100, seed=5)
    net = _built_bc(obs, act, term, seed=5, n_steps=400)
    rate = clone_fidelity(CFG, net, pol, seed=5, n_episodes=100)
    assert rate > 0.999, f"constant-policy clone fidelity {rate:.4f}"


def test_untrained_net_fails_fidelity():
    # negative control against the STATE-DEPENDENT myopic rule (an untrained net's
    # near-constant argmax can match a constant target by init luck, which made a
    # constant-target version of this control flaky), with torch init pinned
    import torch
    from promo_env import myopic_batch

    torch.manual_seed(0)
    pol = myopic_batch
    obs, act, term = _collect_states_actions(CFG, pol, 20, seed=5)
    net = _built_bc(obs, act, term, seed=5, n_steps=0)  # untrained
    rate = clone_fidelity(CFG, net, pol, seed=5, n_episodes=50)
    assert rate < 0.90, (
        f"an untrained net should not pass the fidelity gate (rate {rate:.4f})"
    )
