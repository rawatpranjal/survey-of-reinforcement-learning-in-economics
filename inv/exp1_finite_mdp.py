import matplotlib.pyplot as plt
import numpy as np
from finite_lp_irl import finite_lp_irl, generate_random_mdp, value_iteration

num_states, num_actions, discount_factor, max_reward, regularization = 10, 10, 0.9, 10.0, 0.1
np.random.seed(42)

R_true, Pssa = generate_random_mdp(num_states, num_actions, max_reward)
V_opt, policy_opt = value_iteration(num_states, num_actions, R_true, Pssa, discount_factor)
R_est, V_est = finite_lp_irl(num_states, num_actions, Pssa, discount_factor, max_reward, policy_opt, regularization)
V_est_opt, policy_est = value_iteration(num_states, num_actions, R_est, Pssa, discount_factor)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(num_states), R_true, alpha=0.6, label='True Rewards')
plt.bar(range(num_states), R_est, alpha=0.6, label='Estimated Rewards')
plt.title('True vs. Estimated Rewards')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_states), policy_opt, 'o-', label='Optimal Policy')
plt.plot(range(num_states), policy_est, 's-', label='Estimated Policy')
plt.title('Optimal vs. Estimated Policies')
plt.legend()

plt.tight_layout()
plt.savefig('plots/exp1_finite_mdp.png')
plt.close()
