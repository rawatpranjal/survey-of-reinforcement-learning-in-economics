import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from finite_lp_irl import finite_lp_irl, value_iteration, generate_random_mdp

grid_size, N, K, gamma, Rmax, lp_lambda = 5, 25, 5, 0.9, 1.0, 1
actions = ['up', 'down', 'left', 'right', 'stay']
P = np.zeros((N, N, K))
noise_prob = 0.3

for s in range(N):
    x, y = divmod(s, grid_size)
    for a_idx, action in enumerate(actions):
        next_x, next_y = x + (action == 'down') - (action == 'up'), y + (action == 'right') - (action == 'left')
        next_s = s if not (0 <= next_x < grid_size and 0 <= next_y < grid_size) else next_x * grid_size + next_y
        for noise_action in range(K):
            next_x_noise, next_y_noise = x + (noise_action == 1) - (noise_action == 0), y + (noise_action == 3) - (noise_action == 2)
            next_s_noise = s if not (0 <= next_x_noise < grid_size and 0 <= next_y_noise < grid_size) else next_x_noise * grid_size + next_y_noise
            P[s, next_s if noise_action == a_idx else next_s_noise, noise_action] = 1.0 - noise_prob if noise_action == a_idx else noise_prob / (K - 1)

true_R = np.zeros(N)
true_R[grid_size - 1] = 1.0  # Reward at the top-right corner
V_opt, opt_pi = value_iteration(N, K, true_R, P, gamma)
estimated_R, V_est = finite_lp_irl(N, K, P, gamma, Rmax, opt_pi, lp_lambda)
_, est_pi = value_iteration(N, K, estimated_R, P, gamma)

true_R_grid, estimated_R_grid = true_R.reshape(grid_size, grid_size), estimated_R.reshape(grid_size, grid_size)
action_to_vector = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

U_true, V_true, U_estimated, V_estimated = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
for s in range(N):
    x, y = divmod(s, grid_size)
    U_true[x, y], V_true[x, y] = action_to_vector[opt_pi[s]]
    U_estimated[x, y], V_estimated[x, y] = action_to_vector[est_pi[s]]

X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})
axs[0, 0].plot_surface(X, Y, true_R_grid, cmap='viridis')
axs[0, 0].set_title('True Reward Function')
axs[0, 1].plot_surface(X, Y, estimated_R_grid, cmap='viridis')
axs[0, 1].set_title('Estimated Reward Function')

ax3 = fig.add_subplot(2, 2, 3)
ax3.quiver(X, Y, U_true, V_true)
ax3.set_title('Optimal Policy (True)')
ax3.set_aspect('equal')
ax3.invert_yaxis()

ax4 = fig.add_subplot(2, 2, 4)
ax4.quiver(X, Y, U_estimated, V_estimated)
ax4.set_title('Optimal Policy (Estimated)')
ax4.set_aspect('equal')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('plots/exp2_finite_gridworld1.png')
plt.close()
