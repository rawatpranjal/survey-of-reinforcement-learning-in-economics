import unittest

import numpy as np

from ch07a_online_optimization.sims.online_optimization_rl import (
    H,
    P_TRUE,
    RHO,
    flow_matrix,
    occupancy,
    policy_loss_by_bellman,
    sim_bellman,
    sim_known_mdp,
    sim_one_step_equivalence,
    sim_soil_policy_regret,
    sim_stability,
)


class OnlineOptimizationTests(unittest.TestCase):
    def test_uniform_policy_satisfies_full_flow_system(self):
        policy = np.full((H, 3, 2), 0.5)
        q = occupancy(P_TRUE, policy)
        aeq, beq = flow_matrix(P_TRUE)
        np.testing.assert_allclose(aeq @ q.ravel(), beq, atol=1e-12)
        np.testing.assert_allclose(q[0].sum(axis=1), RHO, atol=1e-12)

    def test_alternating_losses_make_ftl_regret_linear(self):
        horizons, results = sim_stability()
        np.testing.assert_allclose(results["FTL"], horizons / 2.0)
        self.assertLess(results["Hedge"][-1] / horizons[-1], 0.01)

    def test_one_step_oco_and_mdp_are_identical(self):
        policy_residual, value_residual = sim_one_step_equivalence()
        self.assertLess(policy_residual, 1e-14)
        self.assertLess(value_residual, 1e-14)

    def test_realized_state_regret_understates_soil_policy_regret(self):
        result = sim_soil_policy_regret()
        self.assertEqual(result["best_policy"], "Rotate")
        self.assertGreater(result["policy"][-1], 2.5 * result["external"][-1])

    def test_occupancy_and_bellman_policy_values_match(self):
        policy = np.full((H, 3, 2), 0.5)
        loss = np.array([[0.1, 0.2], [0.3, 0.4], [0.8, 0.6]])
        q = occupancy(P_TRUE, policy)
        occupancy_loss = float(np.sum(q * loss[None, :, :]))
        bellman_loss = policy_loss_by_bellman(P_TRUE, policy, loss)
        self.assertAlmostEqual(occupancy_loss, bellman_loss, places=12)
        *_, representation_residual = sim_known_mdp(K=10)
        self.assertLess(representation_residual, 1e-11)

    def test_averaged_bellman_iterates_reduce_policy_gap(self):
        result = sim_bellman(iterations=2000)
        self.assertLess(result["policy"][-1], result["policy"][0])
        self.assertLess(result["gap"][-1], result["gap"][0])


if __name__ == "__main__":
    unittest.main()
