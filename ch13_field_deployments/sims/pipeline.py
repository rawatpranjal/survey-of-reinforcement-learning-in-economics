# OPE-reliability pipeline, factored into small importable units so each is tested
# in isolation. The Scope-RL / d3rlpy wiring lives here; the DGP lives in promo_env.py.
#
# Boundary note: Scope-RL (via old gym) hard-checks isinstance(env, gym.Env) and
# gym.spaces, while PromoEnv is gymnasium. PromoGymAdapter presents an old-gym face
# by delegating every call to a PromoEnv, so the dynamics are untouched.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gym
from gym import spaces as gym_spaces


from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
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
# Behavior policies (the two logging regimes) and log generation.
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


def behavior_head(cfg, regime, seed):
    """Scope-RL head for the logging policy. 'ab' = uniform (full support);
    'observational' = an epsilon-greedy imitation of the myopic incumbent
    (state-correlated, support-collapsing)."""
    if regime == "ab":
        # uniform: a built net wrapped with epsilon=1.0 ignores the net entirely.
        obs, act, term = _collect_states_actions(cfg, myopic_batch, 20, seed)
        net = _built_bc(obs, act, term, seed, n_steps=0)
        return EpsilonGreedyHead(
            net, n_actions=N_ACTIONS, epsilon=1.0, name="ab", random_state=seed
        )
    if regime == "observational":
        obs, act, term = _collect_states_actions(cfg, myopic_batch, 400, seed)
        net = _built_bc(obs, act, term, seed, n_steps=1500)
        return EpsilonGreedyHead(
            net,
            n_actions=N_ACTIONS,
            epsilon=0.1,
            name="observational",
            random_state=seed,
        )
    raise ValueError(f"unknown regime {regime!r}")


def generate_log(cfg, regime, n_traj, seed):
    """Collect a Scope-RL logged_dataset from the DGP under the given regime."""
    adapter = PromoGymAdapter(cfg, seed=seed)
    dataset = SyntheticDataset(env=adapter, max_episode_steps=cfg.horizon)
    head = behavior_head(cfg, regime, seed)
    return dataset.obtain_episodes(
        behavior_policies=head, n_trajectories=n_traj, random_state=seed
    )


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


def train_candidates(
    cfg, logged, seed, bc_steps=1200, cql_steps=2000, cand_epsilon=0.0
):
    """The deployment menu OPE must rank. Reference-policy clones give a known true
    ordering; a CQL trained on the log is the offline-RL candidate. `cand_epsilon` softens
    the reference/CQL clones (0.0 = deterministic greedy). A positive epsilon gives the
    candidates finite overlap with the behavior policy so importance-sampling OPE is not
    structurally degenerate (a deterministic target over a long horizon has match
    probability approx |A|^-H and no logged trajectory survives). `uniform` stays epsilon=1."""
    cands = {}
    refs = [
        ("no_promo", constant_batch(0)),
        ("disc5", constant_batch(1)),
        ("disc10", constant_batch(2)),
        ("myopic", myopic_batch),
    ]
    for i, (nm, pol) in enumerate(refs):
        obs, act, term = _collect_states_actions(cfg, pol, 300, seed + 11 * (i + 1))
        net = _built_bc(obs, act, term, seed, n_steps=bc_steps)
        cands[nm] = EpsilonGreedyHead(
            net, n_actions=N_ACTIONS, epsilon=cand_epsilon, name=nm, random_state=seed
        )
    obs0, act0, term0 = _collect_states_actions(cfg, myopic_batch, 20, seed)
    cands["uniform"] = EpsilonGreedyHead(
        _built_bc(obs0, act0, term0, seed, 0),
        n_actions=N_ACTIONS,
        epsilon=1.0,
        name="uniform",
        random_state=seed,
    )
    md = MDPDataset(
        observations=logged["state"].astype(np.float32),
        actions=logged["action"].astype(np.int64),
        rewards=logged["reward"].astype(np.float32),
        terminals=logged["terminal"].astype(np.float32),
    )
    cql = DiscreteCQLConfig(batch_size=256).create(device="cpu")
    cql.build_with_dataset(md)
    cql.fit(md, n_steps=cql_steps, n_steps_per_epoch=cql_steps, show_progress=False)
    cands["cql"] = EpsilonGreedyHead(
        cql, n_actions=N_ACTIONS, epsilon=cand_epsilon, name="cql", random_state=seed
    )
    return cands


def true_values(cfg, candidates, n_ep=8000, seed=5000):
    """Independent ground-truth field value of each candidate by Monte Carlo rollout
    in the hidden DGP (cross-checks Scope-RL's on-policy value)."""
    env = PromoEnv(cfg, seed=seed)
    return {
        nm: vec_rollout_value(env, _head_batch_policy(h), n_ep, seed)
        for nm, h in candidates.items()
    }


def build_ope_inputs(cfg, logged, candidates, seed, fqe_steps=1500, n_on_policy=500):
    """CreateOPEInput -> per-candidate FQE value predictions + on-policy true value."""
    adapter = PromoGymAdapter(cfg, seed=seed)
    prep = CreateOPEInput(env=adapter, gamma=cfg.gamma, device="cpu")
    return prep.obtain_whole_inputs(
        logged_dataset=logged,
        evaluation_policies=list(candidates.values()),
        require_value_prediction=True,
        n_steps=fqe_steps,
        n_trajectories_on_policy_evaluation=n_on_policy,
        random_state=seed,
    )


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
    """Scope-RL's own on-policy (env-rollout) true value per candidate, for cross-check."""
    return {
        nm: float(np.asarray(d["on_policy_policy_value"]).reshape(-1)[0])
        for nm, d in input_dict.items()
    }
