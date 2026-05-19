"""Schmidhuber baseline must learn reward from data, not hardcode it.

The original implementation computed imagined reward inside REINFORCE as
(s_next_idx == goal_state_id), giving the agent free knowledge of the goal
location. A faithful learned-world-model baseline must train a reward head
on real transitions and use the predicted reward during planning.
"""

import inspect

import numpy as np
import pytest
import torch

import dyna_maze as dm


def test_reinforce_does_not_reference_goal_state_id():
    """The REINFORCE rollout must not analytically compute reward from the
    known goal location — it must use the learned reward predictor."""
    src = inspect.getsource(dm.Schmidhuber1990Agent._train_c_reinforce)
    assert 'goal_state_id' not in src, (
        "Schmidhuber._train_c_reinforce references goal_state_id — reward "
        "is being computed analytically rather than predicted by the "
        "learned world model."
    )


def test_world_model_exposes_reward_prediction():
    """agent.M(x) must produce a reward prediction (as a tuple element, or
    via a reward_head attribute reachable from the model)."""
    agent = dm.Schmidhuber1990Agent(
        n_states=54, n_actions=4, gamma=0.95,
        lr_m=3e-3, lr_c=3e-3, hidden_dim=16,
        K_plan_interval=10, H_plan=5, n_imagined=4,
        entropy_coef=0.01, goal_state_id=8,
        buffer_cap=1000, seed=0,
    )
    s_oh = torch.zeros(1, 54); s_oh[0, 5] = 1.0
    a_oh = torch.zeros(1, 4); a_oh[0, 2] = 1.0
    x = torch.cat([s_oh, a_oh], dim=-1)
    out = agent.M(x)

    if isinstance(out, tuple):
        assert len(out) == 2, f"Expected (state_logits, reward) tuple; got {len(out)}-tuple"
        state_logits, reward_pred = out
        assert state_logits.shape == (1, 54), state_logits.shape
        assert reward_pred.numel() == 1, (
            f"Reward prediction must be scalar per sample; got shape {reward_pred.shape}"
        )
    else:
        assert hasattr(agent.M, 'reward_head') or hasattr(agent, 'reward_head'), (
            "World model must predict reward via a tuple output or a "
            "reward_head attribute."
        )


def test_reward_head_tracks_observed_reward():
    """Push 500 transitions with reward=2.0; the predicted reward must
    converge toward 2.0 (not stay at the maze's natural 0/1 scale)."""
    agent = dm.Schmidhuber1990Agent(
        n_states=54, n_actions=4, gamma=0.95,
        lr_m=1e-2, lr_c=3e-3, hidden_dim=32,
        K_plan_interval=10_000, H_plan=5, n_imagined=4,
        entropy_coef=0.01, goal_state_id=8,
        buffer_cap=2000, seed=0,
    )
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = int(rng.integers(0, 54))
        a = int(rng.integers(0, 4))
        s_next = int(rng.integers(0, 54))
        agent.observe(s, a, 2.0, s_next)

    s_oh = torch.zeros(1, 54); s_oh[0, 10] = 1.0
    a_oh = torch.zeros(1, 4); a_oh[0, 1] = 1.0
    x = torch.cat([s_oh, a_oh], dim=-1)
    with torch.no_grad():
        out = agent.M(x)

    if isinstance(out, tuple):
        r_pred = float(out[1].squeeze())
    elif hasattr(agent.M, 'reward_head'):
        h = agent.M.trunk(x) if hasattr(agent.M, 'trunk') else None
        assert h is not None, "Cannot reach reward_head without a trunk."
        r_pred = float(agent.M.reward_head(h).squeeze())
    else:
        pytest.fail("World model does not expose a reward predictor.")

    assert abs(r_pred - 2.0) < 0.5, (
        f"Reward head should converge toward observed value 2.0; got {r_pred:.3f}"
    )
