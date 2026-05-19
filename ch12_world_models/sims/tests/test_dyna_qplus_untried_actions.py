"""Dyna-Q+ must initialize untried actions at visited states.

Sutton & Barto §8.3: when state s is first visited, all actions enter the
model with t_last=0 so the bonus kappa * sqrt(t_current) drives exploration
of untried actions. The original code only registered (s, a) pairs that
were actually executed, so the bonus could not surface untried actions.
"""

import numpy as np
import pytest

import dyna_maze as dm


def test_dyna_qplus_initializes_all_actions_at_visited_states():
    agent = dm.DynaAgent(
        n_states=54, n_actions=4,
        alpha=0.1, gamma=0.95, epsilon=0.1,
        K=5, bonus=True, bonus_kappa=1e-4, seed=0,
    )
    s, a, r, s_next = 5, 0, 0.0, 14
    agent.observe(s, a, r, s_next)

    actions_at_s = sorted([a_ for a_ in range(4) if (s, a_) in agent.model])
    actions_at_s_next = sorted([a_ for a_ in range(4) if (s_next, a_) in agent.model])

    assert actions_at_s == [0, 1, 2, 3], (
        f"Dyna-Q+ must register all 4 actions at visited state s={s}; "
        f"found {actions_at_s}. Without this, the curiosity bonus cannot "
        f"drive discovery of untried actions."
    )
    assert actions_at_s_next == [0, 1, 2, 3], (
        f"Dyna-Q+ must register all 4 actions at visited state s_next={s_next}; "
        f"found {actions_at_s_next}."
    )


def test_dyna_qplus_untried_actions_have_t_last_zero():
    """Default (r=0, s'=s, t_last=0) lets the bonus kappa * sqrt(t) grow
    monotonically and eventually pull the policy toward the untried action."""
    agent = dm.DynaAgent(
        n_states=54, n_actions=4,
        alpha=0.1, gamma=0.95, epsilon=0.1,
        K=5, bonus=True, bonus_kappa=1e-4, seed=0,
    )
    agent.observe(5, 0, 0.0, 14)
    for a_ in (1, 2, 3):
        r, s_next, t_last = agent.model[(5, a_)]
        assert r == 0.0, f"Untried (5, {a_}) should have r=0; got {r}"
        assert s_next == 5, f"Untried (5, {a_}) should self-loop; got s'={s_next}"
        assert t_last == 0, f"Untried (5, {a_}) should have t_last=0; got {t_last}"


def test_plain_dyna_q_does_not_initialize_untried():
    """Plain Dyna-Q (bonus=False) should not eagerly initialize — there is
    no bonus, so initializing untried actions with r=0 would just slow
    exploitation. The audit confirms the original behavior is correct here."""
    agent = dm.DynaAgent(
        n_states=54, n_actions=4,
        alpha=0.1, gamma=0.95, epsilon=0.1,
        K=5, bonus=False, bonus_kappa=1e-4, seed=0,
    )
    agent.observe(5, 0, 0.0, 14)
    actions_at_s = sorted([a_ for a_ in range(4) if (5, a_) in agent.model])
    assert actions_at_s == [0], (
        f"Plain Dyna-Q should only register executed (s, a); "
        f"found {actions_at_s} (would slow exploitation if changed)."
    )
