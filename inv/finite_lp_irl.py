import numpy as np
from scipy.linalg import inv
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def finite_lp_irl(num_states, num_actions, transition_probabilities,
                                   discount_factor, max_reward, optimal_policy, regularization=0.0):
    """
    Perform Inverse Reinforcement Learning to estimate the reward function.

    Args:
        num_states (int): Number of states in the MDP.
        num_actions (int): Number of actions available in each state.
        transition_probabilities (np.ndarray): Transition probabilities tensor with shape (S, S, A).
        discount_factor (float): Discount factor gamma (0 < gamma < 1).
        max_reward (float): Upper bound on the reward function values.
        optimal_policy (np.ndarray): Optimal policy array of shape (S,).
        regularization (float): Regularization coefficient lambda.

    Returns:
        tuple: Estimated reward function (R_est) and value function (V_est).
    """
    # Compute the transition matrix under the optimal policy
    P_opt = np.vstack([
        transition_probabilities[s, :, optimal_policy[s]] for s in range(num_states)
    ])

    # Compute the inverse of (I - gamma * P_opt)
    inv_matrix = inv(np.eye(num_states) - discount_factor * P_opt)

    # Function to compute the drop in value from deviating from the optimal policy
    def drop_in_value(action, state):
        delta_p = P_opt[state, :] - transition_probabilities[state, :, action]
        return delta_p @ inv_matrix

    # Problem formulation:
    # Variables: x = [z_1, z_2, ..., z_S, R_1, R_2, ..., R_S]
    # Objective: Minimize -sum(z_s) + lambda * sum(R_s)
    # Constraints:
    #   1. z_s - drop_in_value(a, s) @ R <= 0, for all s, a != optimal_policy[s]
    #   2. -drop_in_value(a, s) @ R <= 0, for all s, a
    #   3. R_s <= Rmax, for all s
    #   4. z_s >= 0, R_s >= 0, for all s

    num_variables = num_states * 2  # z_s and R_s for each state

    # Objective function coefficients
    c = np.concatenate([
        -np.ones(num_states),           # Coefficients for z_s (maximize sum of z_s)
        regularization * np.ones(num_states)  # Coefficients for R_s (regularization term)
    ])

    # Initialize lists to construct inequality constraints (A_ub x <= b_ub)
    A_ub = []
    b_ub = []

    # Constraint 1: z_s - drop_in_value(a, s) @ R <= 0
    for s in range(num_states):
        for a in range(num_actions):
            if a != optimal_policy[s]:
                row = np.zeros(num_variables)
                row[s] = 1  # Coefficient for z_s
                drop = drop_in_value(a, s)
                row[num_states:] = -drop  # Coefficients for R_s
                A_ub.append(row)
                b_ub.append(0)

    # Constraint 2: -drop_in_value(a, s) @ R <= 0
    for s in range(num_states):
        for a in range(num_actions):
            row = np.zeros(num_variables)
            drop = drop_in_value(a, s)
            row[num_states:] = -drop  # Coefficients for R_s
            A_ub.append(row)
            b_ub.append(0)

    # Constraint 3: R_s <= Rmax
    for s in range(num_states):
        row = np.zeros(num_variables)
        row[num_states + s] = 1  # Coefficient for R_s
        A_ub.append(row)
        b_ub.append(max_reward)

    # Variable bounds: z_s >= 0, R_s >= 0
    bounds = [(0, None)] * num_variables

    # Print problem formulation
    print("Linear Programming Problem Formulation:")
    print("-------------------------------------")
    print(f"Objective Function: Minimize -sum(z_s) + {regularization} * sum(R_s)")
    print("Number of Variables:", num_variables)
    print("Number of Constraints:", len(A_ub))
    print("\nConstraints:")
    print("1. z_s - drop_in_value(a, s) @ R <= 0, for all s, a != optimal_policy[s]")
    print("2. -drop_in_value(a, s) @ R <= 0, for all s, a")
    print(f"3. R_s <= {max_reward}, for all s")
    print("4. z_s >= 0, R_s >= 0, for all s")
    print("-------------------------------------\n")

    # Convert constraints to numpy arrays
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Print solver results
    print("Solver Results:")
    print("---------------")
    if res.success:
        print("Optimization succeeded.")
        print(f"Objective value: {res.fun}")
        print(f"Number of iterations: {res.nit}")
    else:
        print("Optimization failed.")
        print(f"Message: {res.message}")
        raise ValueError("Optimization failed: " + res.message)

    # Extract the estimated rewards and value function
    z_est = res.x[:num_states]
    R_est = res.x[num_states:]

    # Ensure that R_est is bounded by Rmax
    R_est = np.clip(R_est, None, max_reward)

    # Compute the estimated value function
    V_est = inv_matrix @ R_est

    return R_est, V_est

def generate_random_mdp(num_states, num_actions, max_reward):
    """
    Generate a random MDP.

    Args:
        num_states (int): Number of states.
        num_actions (int): Number of actions.
        max_reward (float): Maximum reward value.

    Returns:
        tuple: Reward vector (R_true) and transition probabilities tensor (Pssa).
    """
    # Random rewards between 0 and max_reward
    R_true = np.random.uniform(0, max_reward, size=num_states)

    # Random transition probabilities
    Pssa = np.zeros((num_states, num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            probabilities = np.random.rand(num_states)
            probabilities /= probabilities.sum()  # Normalize to sum to 1
            Pssa[s, :, a] = probabilities

    return R_true, Pssa

def value_iteration(num_states, num_actions, rewards, transition_probabilities,
                    discount_factor, epsilon=1e-6, max_iterations=1000):
    """
    Perform Value Iteration to compute the optimal policy.

    Args:
        num_states (int): Number of states.
        num_actions (int): Number of actions.
        rewards (np.ndarray): Reward vector of shape (S,).
        transition_probabilities (np.ndarray): Transition probabilities tensor with shape (S, S, A).
        discount_factor (float): Discount factor gamma (0 < gamma < 1).
        epsilon (float): Convergence threshold.
        max_iterations (int): Maximum number of iterations.

    Returns:
        tuple: Optimal value function (V) and optimal policy (policy).
    """
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    for iteration in range(max_iterations):
        V_prev = V.copy()
        Q = np.zeros((num_states, num_actions))

        for s in range(num_states):
            for a in range(num_actions):
                Q[s, a] = rewards[s] + discount_factor * transition_probabilities[s, :, a] @ V_prev

        V = np.max(Q, axis=1)
        policy = np.argmax(Q, axis=1)

        # Check for convergence
        if np.max(np.abs(V - V_prev)) < epsilon:
            break

    return V, policy