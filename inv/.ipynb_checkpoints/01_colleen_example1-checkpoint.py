import numpy as np
from scipy.optimize import linprog

# Define the state transition probabilities for 2 actions
# Actions are 0: stay, 1: switch

# Transition probabilities:
# - Action 0 (stay): remain in the same state
# - Action 1 (switch): move to the other state

data = [
    # Action 0 (stay)
    1, 0,  # From state 0, stay in state 0 with probability 1
    0, 1,  # From state 1, stay in state 1 with probability 1
    # Action 1 (switch)
    0, 1,  # From state 0, move to state 1 with probability 1
    1, 0   # From state 1, move to state 0 with probability 1
]

p_ssa = np.array(data).reshape((2, 2, 2), order='F')  # Now actions are 0 and 1

print("State transition probabilities (p_ssa):")
for k in range(2):
    print(f"\nAction {k}:")
    print(p_ssa[:, :, k])

# State 0 is preferable to state 1
Rmax = 10

# Optimal policy: stay in state 0, switch in state 1
optimal_p = [2, 7]  # Actions for states 0 and 1

# Transition probability matrix under the optimal policy
p_a1 = np.array([p_ssa[0, :, optimal_p[0]], p_ssa[1, :, optimal_p[1]]])
print("\nTransition probability matrix under the optimal policy (p_a1):")
print(p_a1)

n = p_ssa.shape[0]  # Number of states
k = p_ssa.shape[2]  # Number of actions

# Inverse of (I - beta * P_{a1})
beta = 0.9
p_a1_inv = np.linalg.inv(np.eye(n) - beta * p_a1)
print("\nInverse of (I - beta * P_{a1}):")
print(p_a1_inv)

# Function to compute drop in value from deviating from the optimal policy
def drop_in_v(a, i):
    return np.dot(p_a1[i, :] - p_ssa[i, :, a], p_a1_inv)

# Precompute drop_in_v for all state-action pairs
drop_in_v_values = np.zeros((n, k, n))
for i in range(n):
    for a in range(k):
        drop_in_v_values[i, a, :] = drop_in_v(a, i)

print("\nPrecomputed drop_in_v values:")
print(drop_in_v_values)

# Set up the LP problem
# Variables: x = [z1, z2, R1, R2]
c = [-1, -1, 0, 0]  # Maximize z1 + z2
print("\nObjective function coefficients (c):")
print(c)

# Constraints
A_ub = []
b_ub = []

# First set of constraints: z_i <= drop_in_v(a, i) * R for actions not in the optimal policy
print("\nSetting up the first set of constraints (z_i <= drop_in_v * R)...")
for i in range(n):
    optimal_action = optimal_p[i]
    for a in range(k):
        if a != optimal_action:
            constraint = [0] * 4
            constraint[i] = 1  # Coefficient for z_i
            constraint[2] = -drop_in_v_values[i, a, 0]  # Coefficient for R1
            constraint[3] = -drop_in_v_values[i, a, 1]  # Coefficient for R2
            A_ub.append(constraint)
            b_ub.append(0)
            print(f"Constraint for state {i}, action {a}: {constraint}")

# Second set of constraints: drop_in_v(a, i) * R >= 0
print("\nSetting up the second set of constraints (drop_in_v * R >= 0)...")
for i in range(n):
    for a in range(k):
        constraint = [0, 0, -drop_in_v_values[i, a, 0], -drop_in_v_values[i, a, 1]]
        A_ub.append(constraint)
        b_ub.append(0)
        print(f"Constraint for state {i}, action {a}: {constraint}")

# Third set of constraints: R_i <= Rmax
A_ub.extend([
    [0, 0, 1, 0],  # R1 <= Rmax
    [0, 0, 0, 1],  # R2 <= Rmax
])
b_ub.extend([Rmax, Rmax])

print("\nThird set of constraints (R_i <= Rmax):")
print(A_ub[-2:])
print("Right-hand side (b_ub):", b_ub[-2:])

# Variable bounds
bounds = [(0, None)] * 4  # z1 >= 0, z2 >= 0, R1 >= 0, R2 >= 0
print("\nVariable bounds:")
print(bounds)

# Solve the LP problem
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Check if the optimization was successful
if res.success:
    print("\nOptimal solution found:")
    print(f"z_i (drop in value): {res.x[:2]}")
    print(f"R_i (rewards): {res.x[2:]}")
else:
    print("Optimization failed.")
