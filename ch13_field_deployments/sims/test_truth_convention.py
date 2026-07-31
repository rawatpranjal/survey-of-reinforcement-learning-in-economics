# Regression lock for the scope-rl on-policy discount off-by-one. The BaseHead
# sampled-action branch of rollout_policy_online increments t BEFORE accruing
# gamma**t * reward, so the first reward is discounted by gamma^1 and every value is
# exactly gamma times the standard-convention return. fix_on_policy_convention undoes
# it; this test reproduces the library bug numerically and locks both directions.
import numpy as np
import gym
from gym import spaces as gym_spaces

from scope_rl.ope.online import calc_on_policy_policy_value
from scope_rl.policy.head import BaseHead

from pipeline import fix_on_policy_convention, on_policy_values


class _UnitRewardEnv(gym.Env):
    """Deterministic: reward 1.0 every step, horizon 3."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym_spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.action_space = gym_spaces.Discrete(2)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, a):
        self.t += 1
        return np.zeros(2, np.float32), 1.0, self.t >= 3, False, {}


class _ConstHead(BaseHead):
    """Minimal BaseHead: always action 0 (no base net needed for the sampled path)."""

    def __init__(self):
        self.name = "const0"

    def sample_action(self, x):
        return np.zeros(len(x), dtype=int)

    def sample_action_and_output_pscore(self, x):
        a = self.sample_action(x)
        return a, np.ones(len(x))

    def calc_action_choice_probability(self, x):
        p = np.zeros((len(x), 2))
        p[:, 0] = 1.0
        return p

    def calc_pscore_given_action(self, x, action):
        return (np.asarray(action) == 0).astype(float)


GAMMA = 0.5
V_STANDARD = 1.0 + GAMMA + GAMMA**2  # 1.75
V_SHIFTED = GAMMA * V_STANDARD  # 0.875, what scope-rl actually returns


def test_library_off_by_one_reproduced():
    vals = calc_on_policy_policy_value(
        _UnitRewardEnv(),
        _ConstHead(),
        n_trajectories=4,
        step_per_trajectory=3,
        gamma=GAMMA,
        random_state=0,
    )
    vals = np.atleast_1d(np.asarray(vals, dtype=float))
    assert np.allclose(vals, V_SHIFTED), (
        f"scope-rl returned {vals}; expected the gamma-shifted {V_SHIFTED} "
        f"(if this fails with {V_STANDARD}, the library fixed the bug -- "
        f"remove fix_on_policy_convention)"
    )


def test_fix_restores_standard_convention():
    input_dict = {
        "const0": {"on_policy_policy_value": np.full(4, V_SHIFTED)},
        "none_case": {"on_policy_policy_value": None},
    }
    fixed = fix_on_policy_convention(input_dict, GAMMA)
    assert np.allclose(fixed["const0"]["on_policy_policy_value"], V_STANDARD)
    assert fixed["none_case"]["on_policy_policy_value"] is None


def test_on_policy_values_takes_mean_not_first():
    d = {"a": {"on_policy_policy_value": np.array([1.0, 3.0])}}
    assert on_policy_values(d)["a"] == 2.0  # a prior version returned element [0]
