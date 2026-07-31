# OPE-reliability pipeline, factored into small importable units so each is tested
# in isolation. The Scope-RL / d3rlpy wiring lives here; the DGP lives in promo_env.py.
#
# Boundary note: Scope-RL (via old gym) hard-checks isinstance(env, gym.Env) and
# gym.spaces, while PromoEnv is gymnasium. PromoGymAdapter presents an old-gym face
# by delegating every call to a PromoEnv, so the dynamics are untouched.
import warnings

warnings.filterwarnings("ignore")


from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import gym
from gym import spaces as gym_spaces
from sklearn.utils import check_random_state

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig, QLearningAlgoBase
from d3rlpy.models.encoders import VectorEncoderFactory
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
from scope_rl.policy.head import BaseHead
from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation, OffPolicySelection
from scope_rl.ope.discrete import (
    DirectMethod,
    PerDecisionImportanceSampling,
    DoublyRobust,
)

from promo_env import (
    PromoEnv,
    PromoConfig,
    DISCOUNTS,
    N_ACTIONS,
    myopic_batch,
    constant_batch,
    vec_rollout_value,
)


class PromoGymAdapter(gym.Env):
    """Old-gym.Env face over a gymnasium PromoEnv (the Scope-RL boundary)."""

    def __init__(self, config: PromoConfig | None = None, seed: int = 0):
        super().__init__()
        self._env = PromoEnv(config, seed=seed)
        K, c = self._env.cfg.K, self._env.cfg.ctx_clip
        low = np.concatenate([np.full(K, -c), [0.0, 0.0]]).astype(np.float32)
        high = np.concatenate([np.full(K, c), [DISCOUNTS[-1], 1.0]]).astype(np.float32)
        self.observation_space = gym_spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym_spaces.Discrete(N_ACTIONS)

    @property
    def cfg(self):
        return self._env.cfg

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)


# ---------------------------------------------------------------------------
# Policy heads. Three additions over stock scope-rl:
#   EpsilonMyopicHead      -- the incumbent logging policy, computed exactly from the
#                             firm's own myopic rule (no BC imitation, no mode collapse).
#   MixtureBehaviorHead    -- the historical-mixture logging regime: each episode is
#                             served by one of 7 past pricing policies.
#   SoftTargetEpsilonGreedyHead -- candidate head whose FQE evaluates the actual
#                             epsilon-greedy policy (stock heads make FQE evaluate the
#                             greedy base net instead; see proxy docstring).
# ---------------------------------------------------------------------------
class _EpsGreedyTargetImplProxy:
    """Wraps a d3rlpy QLearningAlgoImplBase so predict_best_action returns an
    epsilon-greedy SAMPLE instead of the argmax. d3rlpy's DiscreteFQE bootstraps its
    TD target through algo.predict_best_action (fqe_impl.inner_update), and scope-rl
    hands it head.impl (d3rlpy/ope/fqe.py, algo=self._algo.impl), so a stock
    EpsilonGreedyHead makes FQE learn Q of the GREEDY base net, not of the
    epsilon-greedy candidate OPE scores (measured: on a 3-action bandit with
    epsilon=1, DM = 1.42 vs analytic uniform value 1.81 and greedy-target value
    1.52). With this proxy the TD target action is drawn epsilon-greedily, so FQE
    performs sampled expected-SARSA whose fixed point is Q of the epsilon-greedy
    policy itself. Pinned to d3rlpy==2.8.1 internals; the oracle test in
    test_fqe_target.py fails loudly if the predict_best_action contract moves."""

    def __init__(self, impl, epsilon, n_actions, seed):
        self._impl = impl
        self._epsilon = float(epsilon)
        self._n_actions = int(n_actions)
        self._gen = torch.Generator().manual_seed(int(seed))

    def predict_best_action(self, x):
        greedy = self._impl.predict_best_action(x)
        n = greedy.shape[0]
        explore = torch.rand(n, generator=self._gen) < self._epsilon
        rand_a = torch.randint(self._n_actions, (n,), generator=self._gen)
        return torch.where(explore.to(greedy.device), rand_a.to(greedy.device), greedy)

    def __getattr__(self, name):
        return getattr(self._impl, name)


@dataclass
class SoftTargetEpsilonGreedyHead(EpsilonGreedyHead):
    """EpsilonGreedyHead whose FQE-facing impl samples epsilon-greedy TD-target
    actions, so the direct method estimates the value of the same softened policy
    the importance-sampling estimators and the ground-truth rollouts target."""

    @property
    def impl(self):
        return _EpsGreedyTargetImplProxy(
            self.base_policy.impl, self.epsilon, self.n_actions, self.random_state
        )


@dataclass
class EpsilonMyopicHead(BaseHead):
    """The incumbent logging policy: epsilon-greedy around the EXACT myopic rule
    (argmax immediate expected margin from the firm's own demand model). Analytic
    propensities; nothing is imitated, so no BC mode collapse is possible. The
    base_policy net is untrained and used only for d3rlpy API delegation."""

    base_policy: QLearningAlgoBase
    name: str
    n_actions: int
    epsilon: float
    config: Optional[PromoConfig] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        self.action_type = "discrete"
        self.action_matrix = np.eye(self.n_actions)
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)
        self._env = PromoEnv(self.config or PromoConfig(), seed=self.random_state)

    def _greedy(self, x):
        x = np.asarray(x, dtype=np.float32)
        return np.asarray(myopic_batch(self._env, x)).astype(int)

    def predict(self, x):
        return self._greedy(x)

    def sample_action(self, x):
        greedy = self._greedy(x)
        n = greedy.shape[0]
        explore = self.random_.rand(n) < self.epsilon
        rand_a = self.random_.randint(self.n_actions, size=n)
        return np.where(explore, rand_a, greedy)

    def calc_pscore_given_action(self, x, action):
        greedy = self._greedy(x)
        match = greedy == np.asarray(action)
        return np.where(
            match,
            1.0 - self.epsilon + self.epsilon / self.n_actions,
            self.epsilon / self.n_actions,
        )

    def calc_action_choice_probability(self, x):
        greedy = self._greedy(x)
        greedy_matrix = self.action_matrix[greedy]
        return (1.0 - self.epsilon) * greedy_matrix + self.epsilon / self.n_actions

    def sample_action_and_output_pscore(self, x):
        action = self.sample_action(x)
        return action, self.calc_pscore_given_action(x, action)


@dataclass
class MixtureBehaviorHead(BaseHead):
    """Historical-mixture logging regime: at each episode start one of 7 components
    (6 constant-discount policies + the exact myopic rule) is drawn uniformly, then
    played epsilon-greedily for the whole episode. This is the log a platform
    accumulates over its policy history, and it covers the reference-discount axis
    that any single policy's log cannot.

    Logged pscore = the REALIZED component's epsilon-greedy propensity. Conditional
    on the episode's component draw Z=k, the trajectory is an ordinary log of pi_k
    with exact per-step propensities, so the pooled log is a stratified sample and
    per-decision/trajectory IS stays unbiased (the naive multiple-logger estimator
    of Agarwal et al. 2017). The per-step mixture MARGINAL would be wrong here:
    actions within an episode are correlated through Z, so a product of marginals
    misstates the sequence likelihood. It is also what a real serving system
    records, since the platform knows which policy served each session.

    Episode boundaries are detected from the observation itself: the t/T coordinate
    is exactly 0.0 only at t=0. A step counter cross-checks and raises on any
    disagreement rather than logging silently mislabeled propensities."""

    base_policy: QLearningAlgoBase
    name: str
    n_actions: int
    horizon: int
    epsilon: float
    config: Optional[PromoConfig] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        self.action_type = "discrete"
        self.action_matrix = np.eye(self.n_actions)
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)
        self._env = PromoEnv(self.config or PromoConfig(), seed=self.random_state)
        self._components = [
            (f"const{a}", self._make_const(a)) for a in range(self.n_actions)
        ] + [("myopic", self._myopic)]
        self.n_components = len(self._components)
        self._k = None  # active component of the current episode
        self._steps = 0  # total single-state sampling calls (cross-check)
        self.component_trace = []  # episode index -> component index

    def _make_const(self, a):
        def fn(x):
            return np.full(x.shape[0], a, dtype=int)

        return fn

    def _myopic(self, x):
        return np.asarray(myopic_batch(self._env, x)).astype(int)

    def component_greedy(self, k, x):
        """Greedy action of component k on a state batch (diagnostics/tests)."""
        return self._components[k][1](np.asarray(x, dtype=np.float32))

    def sample_action_and_output_pscore(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != 1:
            raise ValueError(
                "MixtureBehaviorHead logs one state at a time (scope-rl online path)"
            )
        at_start = bool(x[0, -1] == 0.0)
        expected_start = self._steps % self.horizon == 0
        if at_start != expected_start:
            raise RuntimeError(
                f"episode-boundary mismatch: t/T says start={at_start}, "
                f"step counter {self._steps} says start={expected_start}"
            )
        if at_start:
            self._k = int(self.random_.randint(self.n_components))
            self.component_trace.append(self._k)
        self._steps += 1
        greedy = self._components[self._k][1](x)
        explore = self.random_.rand(1) < self.epsilon
        rand_a = self.random_.randint(self.n_actions, size=1)
        action = np.where(explore, rand_a, greedy).astype(int)
        match = action == greedy
        pscore = np.where(
            match,
            1.0 - self.epsilon + self.epsilon / self.n_actions,
            self.epsilon / self.n_actions,
        )
        return action, pscore

    def calc_pscore_given_action(self, x, action):
        """Mixture MARGINAL pscore. For tests/diagnostics only; never valid as the
        logged pscore, which must be the realized component's propensity (above)."""
        probs = self.calc_action_choice_probability(np.asarray(x, dtype=np.float32))
        idx = np.arange(len(np.atleast_1d(action)))
        return probs[idx, np.asarray(action).astype(int)]

    def calc_action_choice_probability(self, x):
        x = np.asarray(x, dtype=np.float32)
        probs = np.zeros((x.shape[0], self.n_actions))
        for _, fn in self._components:
            greedy_matrix = self.action_matrix[fn(x)]
            probs += (
                (1.0 - self.epsilon) * greedy_matrix + self.epsilon / self.n_actions
            ) / self.n_components
        return probs

    def predict(self, x):
        """Greedy of the CURRENT component (API consistency; tests only)."""
        k = self._k if self._k is not None else 0
        return self._components[k][1](np.asarray(x, dtype=np.float32))

    def sample_action(self, x):
        """Batched sampling with an independent component draw per row (tests only;
        the logging path goes through sample_action_and_output_pscore)."""
        x = np.asarray(x, dtype=np.float32)
        n = x.shape[0]
        ks = self.random_.randint(self.n_components, size=n)
        greedy = np.stack(
            [self._components[k][1](x[i : i + 1])[0] for i, k in enumerate(ks)]
        )
        explore = self.random_.rand(n) < self.epsilon
        rand_a = self.random_.randint(self.n_actions, size=n)
        return np.where(explore, rand_a, greedy)


# ---------------------------------------------------------------------------
# Behavior policies (the logging regimes) and log generation.
# ---------------------------------------------------------------------------
def _collect_states_actions(cfg, batch_policy, n_episodes, seed):
    """Roll a batched policy through the DGP, recording (obs, action) pairs and
    episode-terminal flags (for building a d3rlpy dataset)."""
    env = PromoEnv(cfg, seed=seed)
    B, K, T = n_episodes, cfg.K, cfg.horizon
    rng = np.random.default_rng(seed)
    C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
    r = np.full(B, cfg.r_init)
    obs_all, act_all, term_all = [], [], []
    for t in range(T):
        obs = np.concatenate([C, r[:, None], np.full((B, 1), t / T)], axis=1).astype(
            np.float32
        )
        a = np.asarray(batch_policy(env, obs, rng)).astype(np.int64)
        obs_all.append(obs)
        act_all.append(a)
        term_all.append(np.full(B, t == T - 1, dtype=np.float32))
        d = DISCOUNTS[a]
        r = (t * r + d) / (t + 1)
        if t < T - 1:
            C = np.clip(rng.normal(size=(B, K)), -cfg.ctx_clip, cfg.ctx_clip)
    # interleave by episode so terminals fall at episode ends
    obs = np.stack(obs_all, axis=1).reshape(B * T, K + 2)
    act = np.stack(act_all, axis=1).reshape(B * T)
    term = np.stack(term_all, axis=1).reshape(B * T)
    return obs, act, term


def _built_bc(obs, act, term, seed, n_steps):
    md = MDPDataset(
        observations=obs.astype(np.float32),
        actions=act.astype(np.int64),
        rewards=np.zeros(len(obs), dtype=np.float32),
        terminals=term.astype(np.float32),
    )
    bc = DiscreteBCConfig(batch_size=256).create(device="cpu")
    bc.build_with_dataset(md)
    if n_steps > 0:
        bc.fit(md, n_steps=n_steps, n_steps_per_epoch=n_steps, show_progress=False)
    return bc


def behavior_head(cfg, regime, seed, incumbent_epsilon=0.3, mix_epsilon=0.3):
    """Scope-RL head for the logging policy.
    'ab'        = uniform randomization (full action support);
    'incumbent' = epsilon-greedy around the EXACT myopic rule (narrow support,
                  propensities logged; replaces the old BC-imitation 'observational'
                  regime, whose under-trained clone mode-collapsed to max discount);
    'mixture'   = historical mixture: one of 7 past pricing policies per episode."""
    # a built, untrained net for d3rlpy API delegation (epsilon=1.0 ignores it)
    obs, act, term = _collect_states_actions(cfg, myopic_batch, 20, seed)
    net = _built_bc(obs, act, term, seed, n_steps=0)
    if regime == "ab":
        return EpsilonGreedyHead(
            net, n_actions=N_ACTIONS, epsilon=1.0, name="ab", random_state=seed
        )
    if regime == "incumbent":
        return EpsilonMyopicHead(
            net,
            name="incumbent",
            n_actions=N_ACTIONS,
            epsilon=incumbent_epsilon,
            config=cfg,
            random_state=seed,
        )
    if regime == "mixture":
        return MixtureBehaviorHead(
            net,
            name="mixture",
            n_actions=N_ACTIONS,
            horizon=cfg.horizon,
            epsilon=mix_epsilon,
            config=cfg,
            random_state=seed,
        )
    raise ValueError(f"unknown regime {regime!r}")


def generate_log(cfg, regime, n_traj, seed, incumbent_epsilon=0.3, mix_epsilon=0.3):
    """Collect a Scope-RL logged_dataset from the DGP under the given regime.
    Returns (logged_dataset, head); the head carries the mixture component_trace."""
    adapter = PromoGymAdapter(cfg, seed=seed)
    dataset = SyntheticDataset(env=adapter, max_episode_steps=cfg.horizon)
    head = behavior_head(
        cfg,
        regime,
        seed,
        incumbent_epsilon=incumbent_epsilon,
        mix_epsilon=mix_epsilon,
    )
    logged = dataset.obtain_episodes(
        behavior_policies=head, n_trajectories=n_traj, random_state=seed
    )
    return logged, head


# ---------------------------------------------------------------------------
# Candidate deployment policies, their true field values, and the OPE/OPS wiring.
# ---------------------------------------------------------------------------
def _head_batch_policy(head):
    """Wrap a Scope-RL head as a batched policy for true-value rollouts. Uses
    sample_action, not predict, so the MC true value targets the SAME epsilon-greedy
    policy the OPE estimators evaluate (predict returns the greedy action and would
    measure a different, deterministic policy)."""

    def pol(env, obs, rng=None):
        return np.asarray(head.sample_action(obs.astype(np.float32))).astype(int)

    return pol


def clone_fidelity(cfg, net_or_head, ref_pol, seed, n_episodes=500):
    """Argmax match rate of a trained clone against its reference rule, on states
    visited by the uniform policy (so the reference-discount axis is spread). The
    gate for silent BC infidelity (the old observational log's mode collapse)."""
    from promo_env import uniform_batch

    obs, _, _ = _collect_states_actions(cfg, uniform_batch, n_episodes, seed + 977)
    env = PromoEnv(cfg, seed=seed)
    ref_a = np.asarray(ref_pol(env, obs, np.random.default_rng(seed))).astype(int)
    clone_a = np.asarray(net_or_head.predict(obs)).astype(int)
    return float(np.mean(ref_a == clone_a))


def train_candidates(
    cfg, ab_logged, seed, bc_steps=8000, cql_steps=2000, cand_epsilon=0.3
):
    """The deployment menu OPE must rank. Reference-policy clones give a known true
    ordering; a CQL is the offline-RL candidate. All heads are
    SoftTargetEpsilonGreedyHead so FQE evaluates the same epsilon-greedy policy the
    IS estimators and ground-truth rollouts target. `cand_epsilon` softens the
    reference/CQL clones (finite IS overlap; a deterministic target over horizon H
    has match probability ~ |A|^-H); `uniform` stays epsilon=1.

    CQL trains on the FIXED A/B log of this seed regardless of the regime under
    evaluation, so the candidate menu is identical across regimes and the
    cross-regime estimator comparison is apples-to-apples.

    Returns (cands, fidelity) where fidelity maps clone name -> argmax match rate
    against its reference rule (diagnostics gate)."""
    cands = {}
    fidelity = {}
    # constant policies are trivially clonable; the myopic rule's action boundary is
    # not, so its clone gets a larger dataset and budget (probe: 300ep/8k -> 0.895
    # match rate, 1500ep/40k -> 0.962)
    refs = [
        ("no_promo", constant_batch(0), 300, bc_steps),
        ("disc5", constant_batch(1), 300, bc_steps),
        ("disc10", constant_batch(2), 300, bc_steps),
        ("myopic", myopic_batch, 1500, 5 * bc_steps),
    ]
    for i, (nm, pol, n_ep, steps) in enumerate(refs):
        obs, act, term = _collect_states_actions(cfg, pol, n_ep, seed + 11 * (i + 1))
        net = _built_bc(obs, act, term, seed, n_steps=steps)
        fidelity[nm] = clone_fidelity(cfg, net, pol, seed)
        cands[nm] = SoftTargetEpsilonGreedyHead(
            net, n_actions=N_ACTIONS, epsilon=cand_epsilon, name=nm, random_state=seed
        )
    obs0, act0, term0 = _collect_states_actions(cfg, myopic_batch, 20, seed)
    cands["uniform"] = SoftTargetEpsilonGreedyHead(
        _built_bc(obs0, act0, term0, seed, 0),
        n_actions=N_ACTIONS,
        epsilon=1.0,
        name="uniform",
        random_state=seed,
    )
    md = MDPDataset(
        observations=ab_logged["state"].astype(np.float32),
        actions=ab_logged["action"].astype(np.int64),
        rewards=ab_logged["reward"].astype(np.float32),
        terminals=ab_logged["terminal"].astype(np.float32),
    )
    cql = DiscreteCQLConfig(batch_size=256).create(device="cpu")
    cql.build_with_dataset(md)
    cql.fit(md, n_steps=cql_steps, n_steps_per_epoch=cql_steps, show_progress=False)
    cands["cql"] = SoftTargetEpsilonGreedyHead(
        cql, n_actions=N_ACTIONS, epsilon=cand_epsilon, name="cql", random_state=seed
    )
    return cands, fidelity


def true_values(cfg, candidates, n_ep=8000, seed=5000):
    """Independent ground-truth field value of each candidate by Monte Carlo rollout
    in the hidden DGP (cross-checks Scope-RL's on-policy value)."""
    env = PromoEnv(cfg, seed=seed)
    return {
        nm: vec_rollout_value(env, _head_batch_policy(h), n_ep, seed)
        for nm, h in candidates.items()
    }


def build_ope_inputs(
    cfg,
    logged,
    candidates,
    seed,
    fqe_steps=20000,
    n_on_policy=500,
    fqe_hidden=(256, 256),
):
    """CreateOPEInput -> per-candidate FQE value predictions + on-policy true value.
    The FQE encoder is widened via model_args (scope-rl's default is a single
    100-unit layer, undersized for the 10-dim buried-index state). The on-policy
    values are corrected to the standard discount convention before returning."""
    adapter = PromoGymAdapter(cfg, seed=seed)
    prep = CreateOPEInput(
        env=adapter,
        gamma=cfg.gamma,
        device="cpu",
        model_args={
            "fqe": {
                "encoder_factory": VectorEncoderFactory(hidden_units=list(fqe_hidden)),
                "batch_size": 256,
            }
        },
    )
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged,
        evaluation_policies=list(candidates.values()),
        require_value_prediction=True,
        n_steps=fqe_steps,
        n_trajectories_on_policy_evaluation=n_on_policy,
        random_state=seed,
    )
    return fix_on_policy_convention(input_dict, cfg.gamma)


def fix_on_policy_convention(input_dict, gamma):
    """scope-rl's rollout_policy_online (BaseHead sampled-action branch,
    scope_rl/ope/online.py) increments t BEFORE accruing gamma**t * reward, so the
    first reward is discounted by gamma^1 and every on-policy value is exactly
    gamma times the standard-convention return the estimators and MC rollouts
    target. Measured on the uniform policy: scope-rl 3.459 vs MC 3.623, ratio
    0.9547 = gamma (probe, 2026-07-19). Divide by gamma so truth, regret units,
    and DM errors share one convention. Rankings are invariant to the scaling."""
    for nm in input_dict:
        v = input_dict[nm].get("on_policy_policy_value")
        if v is not None:
            input_dict[nm]["on_policy_policy_value"] = (
                np.asarray(v, dtype=float) / gamma
            )
    return input_dict


def _estimators():
    return [DirectMethod(), PerDecisionImportanceSampling(), DoublyRobust()]


def evaluate_ope(logged, input_dict):
    """OffPolicyEvaluation object over the discrete estimators (IS/DR/DM)."""
    return OffPolicyEvaluation(logged_dataset=logged, ope_estimators=_estimators())


def ops_metrics(ope, input_dict):
    """Off-policy SELECTION reliability per estimator: regret@1, rank-corr, rankings."""
    ops = OffPolicySelection(ope=ope)
    sel = ops.select_by_policy_value(
        input_dict, return_metrics=True, return_true_values=True
    )
    out = {}
    for est, d in sel.items():
        rc = d["rank_correlation"]
        out[est] = {
            "regret": float(d["regret"][0]),
            "rank_corr": float(rc.statistic if hasattr(rc, "statistic") else rc[0]),
            "estimated_ranking": list(d["estimated_ranking"]),
            "true_ranking": list(d.get("true_ranking", [])),
        }
    return out


def on_policy_values(input_dict):
    """Scope-RL's own on-policy (env-rollout) true value per candidate, for cross-check.
    Mean over rollout trajectories (a prior version took element [0], the first
    trajectory's return)."""
    return {
        nm: float(np.asarray(d["on_policy_policy_value"], dtype=float).mean())
        for nm, d in input_dict.items()
    }
