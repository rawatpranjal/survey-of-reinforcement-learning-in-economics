# Oracle for the SoftTargetEpsilonGreedyHead / _EpsGreedyTargetImplProxy fix. On the
# H=3 bandit with a known analytic value, a plain EpsilonGreedyHead(epsilon=1) makes
# scope-rl's FQE evaluate the GREEDY base net (TD targets via predict_best_action =
# argmax), while the SoftTarget head makes it evaluate the actual epsilon-greedy
# (here: uniform) policy. Locks the d3rlpy 2.8.1 predict_best_action contract the
# proxy depends on -- if a d3rlpy upgrade changes it, this fails loudly.
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TQDM_DISABLE"] = "1"
import numpy as np
import pytest

try:
    import tqdm.std as _tqdm_std

    _o = _tqdm_std.tqdm.__init__
    _tqdm_std.tqdm.__init__ = lambda self, *a, **k: _o(
        self, *a, **{**k, "disable": True}
    )
except Exception:
    pass

from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation
from scope_rl.ope.discrete import DirectMethod

from pipeline import SoftTargetEpsilonGreedyHead
from test_ope_oracle import BanditMDP, _const_head, N_ACT, HORIZON

GAMMA = 0.9
DISCOUNT_SUM = 1 + GAMMA + GAMMA**2
R = np.array([1.0 - 0.5 * (a - 1) ** 2 for a in range(N_ACT)])
V_UNIFORM = float(R.mean() * DISCOUNT_SUM)  # 1.8067
V_GREEDY_TAIL = float(R.mean() + R[0] * (GAMMA + GAMMA**2))  # 1.5217


def _dm_value(head_cls, seed=0):
    env = BanditMDP(seed=seed)
    behavior = _const_head(env, 0, 1.0, seed)  # uniform logging
    dataset = SyntheticDataset(env=env, max_episode_steps=HORIZON)
    logged = dataset.obtain_episodes(
        behavior_policies=behavior, n_trajectories=600, random_state=seed
    )
    base = _const_head(env, 0, 1.0, seed + 7).base_policy  # BC net, argmax action 0
    cand = head_cls(
        base, n_actions=N_ACT, epsilon=1.0, name="unif_cand", random_state=seed
    )
    prep = CreateOPEInput(env=env, gamma=GAMMA, device="cpu")
    inp = prep.obtain_whole_inputs(
        logged_dataset=logged,
        evaluation_policies=[cand],
        require_value_prediction=True,
        n_steps=10000,
        n_trajectories_on_policy_evaluation=300,
        random_state=seed,
    )
    ope = OffPolicyEvaluation(logged_dataset=logged, ope_estimators=[DirectMethod()])
    return float(ope.estimate_policy_value(inp)["unif_cand"]["dm"])


@pytest.mark.slow
def test_soft_target_head_evaluates_the_softened_policy():
    dm_plain = _dm_value(EpsilonGreedyHead)
    dm_soft = _dm_value(SoftTargetEpsilonGreedyHead)
    # plain head: FQE bootstraps on the greedy base net -> near the greedy-tail value
    assert abs(dm_plain - V_GREEDY_TAIL) < 0.2, (
        f"plain-head DM {dm_plain:.3f} not near greedy-tail {V_GREEDY_TAIL:.3f}"
    )
    # soft-target head: FQE bootstraps on epsilon-greedy samples -> the uniform value
    assert abs(dm_soft - V_UNIFORM) < 0.15, (
        f"soft-target DM {dm_soft:.3f} not near uniform value {V_UNIFORM:.3f}"
    )
    assert dm_soft > dm_plain + 0.15, (
        f"the proxy must change the evaluated policy (soft {dm_soft:.3f} "
        f"vs plain {dm_plain:.3f})"
    )
