# OPS ranking oracle. On a tiny, short-horizon MDP with GOOD coverage (uniform behavior,
# soft candidates, horizon 3), offline policy selection SHOULD work: the estimated ranking
# must correlate with the truth and the selected policy must be near the true best. This
# proves the scope_rl wiring ranks correctly when coverage is adequate, so a failure on the
# real long-horizon dynamic env is genuine OPE hardness, not a bug. Complements the
# finiteness-only smoke gates (_ope_gate.py) which never checked ranking correctness.
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TQDM_DISABLE"] = "1"
import numpy as np
import gym
from gym import spaces as gym_spaces

import pytest

# tqdm silence (d3rlpy's FQE bar passes disable=False explicitly)
try:
    import tqdm.std as _tqdm_std

    _o = _tqdm_std.tqdm.__init__
    _tqdm_std.tqdm.__init__ = lambda self, *a, **k: _o(
        self, *a, **{**k, "disable": True}
    )
except Exception:
    pass

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
from d3rlpy.ope import DiscreteFQE, FQEConfig
from ope_diagnostics import FQE_EPOCH_LEN
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation, OffPolicySelection
from scope_rl.ope.discrete import (
    DirectMethod,
    PerDecisionImportanceSampling,
    DoublyRobust,
)

N_ACT = 3
BEST = (
    1  # action 1 is best everywhere; the reward is state-independent so it is learnable
)
HORIZON = 3


class BanditMDP(gym.Env):
    """Old-gym env: reward = 1 - 0.5*(a-BEST)^2, independent of state. Short horizon so
    importance sampling does not degenerate; action 1 is unambiguously optimal."""

    def __init__(self, seed=0):
        super().__init__()
        self.observation_space = gym_spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = gym_spaces.Discrete(N_ACT)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self.rng.normal(size=2).astype(np.float32).clip(-2, 2), {}

    def step(self, a):
        r = float(1.0 - 0.5 * (a - BEST) ** 2)
        self.t += 1
        s = self.rng.normal(size=2).astype(np.float32).clip(-2, 2)
        return s, r, self.t >= HORIZON, False, {}


def _const_head(env, action_idx, epsilon, seed, n=300):
    """A soft policy whose greedy action is action_idx, via BC on constant-action data."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 2)).astype(np.float32)
    act = np.full(n, action_idx, dtype=np.int64)
    term = (np.arange(n) % HORIZON == HORIZON - 1).astype(np.float32)
    md = MDPDataset(
        observations=obs, actions=act, rewards=np.zeros(n, np.float32), terminals=term
    )
    bc = DiscreteBCConfig(batch_size=64).create(device="cpu")
    bc.build_with_dataset(md)
    bc.fit(md, n_steps=400, n_steps_per_epoch=400, show_progress=False)
    return EpsilonGreedyHead(
        bc, n_actions=N_ACT, epsilon=epsilon, name=f"a{action_idx}", random_state=seed
    )


@pytest.mark.slow
def test_fqe_trains_only_at_or_above_epoch_length():
    """Regression for the bug that voided the first run: d3rlpy runs n_steps // 10000
    epochs with n_steps_per_epoch defaulting to 10000, so any n_steps below trains the
    FQE for ZERO gradient steps and Q stays at random init. Measured directly here; also
    catches a future d3rlpy upgrade that changes the epoch behavior."""
    rng = np.random.default_rng(0)
    n, T = 1200, 40
    obs = rng.normal(size=(n, 6)).astype(np.float32)
    act = rng.integers(0, 3, size=n).astype(np.int64)
    rew = rng.uniform(0, 0.7, size=n).astype(
        np.float32
    )  # positive rewards -> Q grows once trained
    term = np.zeros(n, np.float32)
    term[np.arange(T - 1, n, T)] = 1.0
    md = MDPDataset(observations=obs, actions=act, rewards=rew, terminals=term)
    base = DiscreteCQLConfig(batch_size=128).create(device="cpu")
    base.fit(md, n_steps=200, n_steps_per_epoch=200, show_progress=False)

    def update_calls(n_steps):
        # count actual gradient updates (robust; an untrained net's Q can be unstable/NaN)
        fqe = DiscreteFQE(algo=base, config=FQEConfig(), device="cpu")
        fqe.build_with_dataset(md)
        count = [0]
        orig = fqe.update

        def patched(*a, _c=count, _o=orig, **k):
            _c[0] += 1
            return _o(*a, **k)

        fqe.update = patched
        fqe.fit(md, n_steps=n_steps, show_progress=False)
        return count[0]

    below = update_calls(FQE_EPOCH_LEN - 1)
    above = update_calls(FQE_EPOCH_LEN)
    assert below == 0, f"sub-epoch n_steps must train zero steps, got {below}"
    assert above == FQE_EPOCH_LEN, (
        f"one epoch must train {FQE_EPOCH_LEN} updates, got {above}"
    )


@pytest.mark.slow
def test_ops_ranks_correctly_under_good_coverage():
    seed = 0
    env = BanditMDP(seed=seed)
    # behavior: uniform (full support) -> good coverage
    behavior = _const_head(env, 0, 1.0, seed)  # epsilon=1 -> uniform behavior
    dataset = SyntheticDataset(env=env, max_episode_steps=HORIZON)
    logged = dataset.obtain_episodes(
        behavior_policies=behavior, n_trajectories=600, random_state=seed
    )
    # candidates: soft policies toward each action. True order: toward-1 (best) > others.
    cands = [_const_head(env, a, 0.2, seed + a) for a in range(N_ACT)]
    prep = CreateOPEInput(env=env, gamma=0.9, device="cpu")
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged,
        evaluation_policies=cands,
        require_value_prediction=True,
        n_steps=10000,
        n_trajectories_on_policy_evaluation=300,
        random_state=seed,
    )
    ope = OffPolicyEvaluation(
        logged_dataset=logged,
        ope_estimators=[
            DirectMethod(),
            PerDecisionImportanceSampling(),
            DoublyRobust(),
        ],
    )
    ops = OffPolicySelection(ope=ope)
    sel = ops.select_by_policy_value(
        input_dict, return_metrics=True, return_true_values=True
    )

    # at least one estimator must rank well AND select a near-best policy under good coverage
    best_rc = -2.0
    best_regret = 9.9
    for est, dd in sel.items():
        rc = dd["rank_correlation"]
        rc = float(rc.statistic if hasattr(rc, "statistic") else rc[0])
        reg = float(dd["regret"][0])
        best_rc = max(best_rc, rc)
        best_regret = min(best_regret, reg)
    assert best_rc >= 0.5, (
        f"no estimator ranked well under good coverage (best rc={best_rc:.2f})"
    )
    assert best_regret <= 0.2, (
        f"selected policy far from best (min regret={best_regret:.2f})"
    )
