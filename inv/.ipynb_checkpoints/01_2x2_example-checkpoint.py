import numpy as np

# MDP
A = np.array([0, 1])
S = np.array([0, 1])
gamma = 0.9
R = np.array([7, 2])
P_ssa = np.array([[[0.8, 0.2], [0.1, 0.9]], [[0.9, 0.1], [0.2, 0.8]]])
Rmax = 10
Rmin = 0

# VFI
V = np.zeros(len(S))
pi = np.zeros(len(S), dtype=int)

for _ in range(10000):
    V_prev = V.copy()
    for s in S:
        action_values = np.zeros(len(A))
        for a in A:
            P = P_ssa[s, :, a]
            action_values[a] = R[s] + gamma * np.sum(P * V_prev)
        pi[s] = np.argmax(action_values)
        V[s] = np.max(action_values)

# Transition matrix P_a1 under the optimal policy
P_a1 = np.zeros((len(S), len(S)))
for s in S:
    P_a1[s, :] = P_ssa[s, :, pi[s]]

# Conditions for both actions a0 and a1
P_a0 = P_ssa[:, :, 0]
P_a1_alt = P_ssa[:, :, 1]

# Matrix to check for action a0: (P_a1 - P_a0) (I - gamma * P_a1)^-1 R
I_2 = np.eye(len(S))
inv_term = np.linalg.inv(I_2 - gamma * P_a1)
condition_a0 = (P_a1 - P_a0) @ inv_term @ R

# Matrix to check for action a1: (P_a1 - P_a1_alt) (I - gamma * P_a1)^-1 R
condition_a1 = (P_a1 - P_a1_alt) @ inv_term @ R

print("Condition Matrix for a0 (P_a1 - P_a0) (I(2) - gamma * P_a1)^-1 R:")
print(condition_a0)
print("Condition Satisfied for a0 (>= 0):", condition_a0 >= 0)

print("Condition Matrix for a1 (P_a1 - P_a1_alt) (I(2) - gamma * P_a1)^-1 R:")
print(condition_a1)
print("Condition Satisfied for a1 (>= 0):", condition_a1 >= 0)
