import numpy as np
import matplotlib.pyplot as plt
from finite_lp_irl import finite_lp_irl, value_iteration

# Grid parameters for cliff walking
grid_height, grid_width = 4, 12
N = grid_height * grid_width  # Total number of states
K = 4  # Actions: up, down, left, right
gamma = 0.9  # Discount factor
Rmax = 1.0  # Maximum reward (for normalization)
lp_lambda = 0.1  # Regularization coefficient

actions = ['up', 'down', 'left', 'right']
P = np.zeros((N, N, K))  # Transition matrix
noise_prob = 0.0  # No noise for deterministic behavior in this example

# Set up the grid transitions without noise
for s in range(N):
    x, y = divmod(s, grid_width)
    for a_idx, action in enumerate(actions):
        next_x, next_y = x + (action == 'down') - (action == 'up'), y + (action == 'right') - (action == 'left')
        next_s = s if not (0 <= next_x < grid_height and 0 <= next_y < grid_width) else next_x * grid_width + next_y
        P[s, next_s, a_idx] = 1.0

# Cliff setup and rewards
true_R = -np.ones(N)  # Default reward is -1 for every move

cliff_states = np.arange(grid_width + 1, 2 * grid_width)  # Cliff is between (1,1) to (1,10) on grid
true_R[cliff_states] = -100.0  # Falling off the cliff gives a reward of -100

start_state = grid_width * (grid_height - 1)  # Start state (bottom-left corner)
goal_state = grid_width * grid_height - 1  # Goal state (bottom-right corner)
true_R[goal_state] = 0.0  # Reaching the goal gives a reward of 0

# Value iteration to find the optimal policy and value function
V_opt, opt_pi = value_iteration(N, K, true_R, P, gamma)

# Perform Inverse Reinforcement Learning using the optimal policy
estimated_R, V_est = finite_lp_irl(N, K, P, gamma, Rmax, opt_pi, lp_lambda)

# Compute the estimated policy using the estimated rewards
_, est_pi = value_iteration(N, K, estimated_R, P, gamma)

# Reshape rewards for plotting
true_R_grid = true_R.reshape(grid_height, grid_width)
estimated_R_grid = estimated_R.reshape(grid_height, grid_width)

# Prepare for quiver plot data (Optimal Policy)
action_to_vector = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

U_true = np.zeros((grid_height, grid_width))
V_true = np.zeros((grid_height, grid_width))
U_estimated = np.zeros((grid_height, grid_width))
V_estimated = np.zeros((grid_height, grid_width))

# Fill quiver data for true policy
for s in range(N):
    x, y = divmod(s, grid_width)
    U_true[x, y], V_true[x, y] = action_to_vector[opt_pi[s]]
    U_estimated[x, y], V_estimated[x, y] = action_to_vector[est_pi[s]]

# Plot the 2x2 grid of True and Estimated Reward with Policies
X, Y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
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
ax3.set_xlim(-0.5, grid_width - 0.5)
ax3.set_ylim(-0.5, grid_height - 0.5)
ax3.invert_yaxis()

# Plot the estimated policy (quiver plot)
ax4 = fig.add_subplot(2, 2, 4)
ax4.quiver(X, Y, U_estimated, V_estimated)
ax4.set_title('Optimal Policy (Estimated)')
ax4.set_aspect('equal')
ax4.set_xlim(-0.5, grid_width - 0.5)
ax4.set_ylim(-0.5, grid_height - 0.5)
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('plots/cliff_walking_gridworld.png')
plt.close()
