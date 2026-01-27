import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from finite_lp_irl import finite_lp_irl, value_iteration

grid_size, N, K, gamma, Rmax, lp_lambda = 5, 25, 4, 0.9, 1.0, 2
actions = ['up', 'down', 'left', 'right']
P = np.zeros((N, N, K))
noise_prob = 0.3

# Set up the gridworld transitions with noise (no "stay" action)
for s in range(N):
    x, y = divmod(s, grid_size)
    for a_idx, action in enumerate(actions):
        next_x, next_y = x + (action == 'down') - (action == 'up'), y + (action == 'right') - (action == 'left')
        next_s = s if not (0 <= next_x < grid_size and 0 <= next_y < grid_size) else next_x * grid_size + next_y
        for noise_action in range(K):
            next_x_noise, next_y_noise = x + (noise_action == 1) - (noise_action == 0), y + (noise_action == 3) - (noise_action == 2)
            next_s_noise = s if not (0 <= next_x_noise < grid_size and 0 <= next_y_noise < grid_size) else next_x_noise * grid_size + next_y_noise
            P[s, next_s if noise_action == a_idx else next_s_noise, noise_action] = 1.0 - noise_prob if noise_action == a_idx else noise_prob / (K - 1)

# More complex reward structure
true_R = np.zeros(N)

# Absorbing state at the top-right corner with a reward of 10
absorbing_state = grid_size * grid_size - 1
true_R[absorbing_state] = 10.0

# Add rewards of 5 at specific side locations (non-absorbing)
side_reward_locations = [1, grid_size * (grid_size - 1), grid_size - 2, grid_size * 2 - 1]  # Side states
for loc in side_reward_locations:
    true_R[loc] = 5.0

# Value iteration to find the optimal policy and value function
V_opt, opt_pi = value_iteration(N, K, true_R, P, gamma)

# Perform Inverse Reinforcement Learning using the optimal policy
estimated_R, V_est = finite_lp_irl(N, K, P, gamma, Rmax, opt_pi, lp_lambda)

# Compute the estimated policy using the estimated rewards
_, est_pi = value_iteration(N, K, estimated_R, P, gamma)

# Reshape rewards for plotting
true_R_grid = true_R.reshape(grid_size, grid_size)
estimated_R_grid = estimated_R.reshape(grid_size, grid_size)

# Prepare for quiver plot data (Optimal Policy)
action_to_vector = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

U_true = np.zeros((grid_size, grid_size))
V_true = np.zeros((grid_size, grid_size))
U_estimated = np.zeros((grid_size, grid_size))
V_estimated = np.zeros((grid_size, grid_size))

# Fill quiver data for true policy
for s in range(N):
    x, y = divmod(s, grid_size)
    U_true[x, y], V_true[x, y] = action_to_vector[opt_pi[s]]
    U_estimated[x, y], V_estimated[x, y] = action_to_vector[est_pi[s]]

# Plot the 2x2 grid of True and Estimated Reward with Policies
X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})

# True Reward
ax1 = axs[0, 0]
ax1.plot_surface(X, Y, true_R_grid, cmap='viridis')
ax1.set_title('True Reward Function')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Reward')

# Estimated Reward
ax2 = axs[0, 1]
ax2.plot_surface(X, Y, estimated_R_grid, cmap='viridis')
ax2.set_title('Estimated Reward Function')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Reward')

# Plot the true policy (quiver plot)
ax3 = fig.add_subplot(2, 2, 3)
ax3.quiver(X, Y, U_true, V_true)
ax3.set_title('Optimal Policy (True)')
ax3.set_aspect('equal')
ax3.set_xlim(-0.5, grid_size - 0.5)
ax3.set_ylim(-0.5, grid_size - 0.5)
ax3.invert_yaxis()

# Plot the estimated policy (quiver plot)
ax4 = fig.add_subplot(2, 2, 4)
ax4.quiver(X, Y, U_estimated, V_estimated)
ax4.set_title('Optimal Policy (Estimated)')
ax4.set_aspect('equal')
ax4.set_xlim(-0.5, grid_size - 0.5)
ax4.set_ylim(-0.5, grid_size - 0.5)
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('plots/exp2_finite_gridworld_no_stay.png')
plt.close()
