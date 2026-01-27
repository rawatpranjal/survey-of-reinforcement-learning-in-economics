import numpy as np
import scipy.linalg as la

def mpeprob(inip, maxiter, param):
    """
    Computes players' equilibrium conditional choice probabilities (CCPs)
    using the policy iteration algorithm.

    Parameters:
    -----------
    inip : ndarray
        Matrix of initial choice probabilities with numx rows for each state
        and nplayer columns for each player.
    maxiter : int
        Maximum number of policy iterations.
    param : dict
        Structure containing parameter values and settings:
        - theta_fc: Vector of fixed costs for each player.
        - theta_rs: Coefficient on market size.
        - theta_rn: Coefficient on number of rival firms.
        - theta_ec: Entry cost parameter.
        - disfact: Discount factor.
        - sigmaeps: Standard deviation of choice-specific shocks.
        - sval: Vector of possible market sizes.
        - ptrans: Transition matrix for market size.
        - verbose: Flag for detailed output.

    Returns:
    --------
    prob1 : ndarray
        Matrix of MPE entry probabilities with numx rows and nplayer columns.
    psteady1 : ndarray
        Vector with numx rows containing the steady-state distribution of (s_t, a_{t-1}).
    mstate : ndarray
        Matrix with numx rows and nplayer+1 columns containing state variables (s_t, a_{t-1}).
    dconv : int
        Indicator for convergence (1 = convergence, 0 = no convergence).
    """
    # Calculate dimensions
    nums = len(param['sval'])
    nplayer = inip.shape[1]
    numa = 2**nplayer
    numx = nums * numa

    # Print header information
    if param['verbose']:
        print("\n")
        print("*****************************************************************************************")
        print("   COMPUTING A MPE OF THE DYNAMIC GAME")
        print("*****************************************************************************************")
        print("\n")
        print("----------------------------------------------------------------------------------------")
        print("       Values of the structural parameters")
        print("\n")
        for i in range(nplayer):
            print(f"                       Fixed cost firm {i+1}   = {param['theta_fc'][i]:12.4f}")
        print(f"       Parameter of market size (theta_rs) = {param['theta_rs']:12.4f}")
        print(f"Parameter of competition effect (theta_rn) = {param['theta_rn']:12.4f}")
        print(f"                     Entry cost (theta_ec) = {param['theta_ec']:12.4f}")
        print(f"                       Discount factor     = {param['disfact']:12.4f}")
        print(f"                    Std. Dev. epsilons     = {param['sigmaeps']:12.4f}")
        print("\n")
        print("----------------------------------------------------------------------------------------")
        print("\n")
        print("       BEST RESPONSE MAPPING ITERATIONS")
        print("\n")

    # Construct the mstate matrix with values of s_t, a_{t-1}
    aval = np.zeros((numa, nplayer))
    for i in range(nplayer):
        aval[:, i] = np.kron(np.kron(np.ones(2**(i)), np.array([0, 1])), np.ones(2**(nplayer-i-1)))

    mstate = np.zeros((numx, nplayer + 1))
    mstate[:, 0] = np.kron(param['sval'], np.ones(numa))
    mstate[:, 1:nplayer+1] = np.kron(np.ones((nums, 1)), aval)

    # Initialize vector of probabilities
    prob0 = inip.copy()
    critconv = (1e-3) * (1/numx)
    criter = 1000
    dconv = 1

    # Iterative algorithm
    iter_count = 1
    while (criter > critconv) and (iter_count <= maxiter):
        if param['verbose']:
            print(f"         Best response mapping iteration  = {iter_count}")
            print(f"         Convergence criterion = {criter}")
            print("\n")

        prob1 = prob0.copy()

        # Matrix of transition probs Pr(a_t | s_t, a_{t-1})
        # Initialize ptrana as a numx x numa matrix of ones
        ptrana = np.ones((numx, numa))

        # Build the transition probability matrix
        for i in range(nplayer):
            # Create the appropriate matrices for broadcasting
            mi = np.tile(aval[:, i], (numx, 1))  # Expand to numx rows
            ppi = np.tile(prob1[:, i].reshape(-1, 1), (1, numa))  # Expand to numa columns

            # Calculate transition probability components
            ptrana = ptrana * (ppi ** mi) * ((1 - ppi) ** (1 - mi))

        # For each player, compute best response
        for i in range(nplayer):
            # Matrices Pr(a_t | s_t, a_{t-1}, a_{it})
            mi = np.tile(aval[:, i], (numx, 1))
            ppi = np.tile(prob1[:, i].reshape(-1, 1), (1, numa))

            # Create the conditional transition matrices
            iptran0 = (1 - mi) / ((ppi ** mi) * ((1 - ppi) ** (1 - mi)))
            iptran0 = ptrana * iptran0
            iptran1 = mi / ((ppi ** mi) * ((1 - ppi) ** (1 - mi)))
            iptran1 = ptrana * iptran1

            # Computing h_i = E[ln(N_{-it} + 1)]
            hi = aval.copy()
            hi[:, i] = np.ones(numa)
            hi_sum = np.sum(hi, axis=1)  # Sum across rows
            hi = iptran1 @ np.log(hi_sum)

            # Matrices with Expected Profits of Firm i
            profit1 = param['theta_fc'][i] + \
                     param['theta_rs'] * mstate[:, 0] - \
                     param['theta_ec'] * (1 - mstate[:, i+1]) - \
                     param['theta_rn'] * hi  # Profit if firm is active
            profit0 = np.zeros(numx)  # Profit if firm is not active

            # Transition Probabilities for Firm i
            # Pr(x_{t+1}, a_t | x_t, a_{t-1}, a_{i,t}=0) and Pr(x_{t+1}, a_t | x_t, a_{t-1}, a_{i,t}=1)
            # Create the full transition matrices
            ptrans_expanded = np.kron(param['ptrans'], np.ones((numa, numa)))
            ones_expanded = np.kron(np.ones((1, nums)), iptran0)
            iptran0 = ptrans_expanded * ones_expanded

            ones_expanded = np.kron(np.ones((1, nums)), iptran1)
            iptran1 = ptrans_expanded * ones_expanded

            # Computing Value Function for Firm i using Bellman iteration
            v0 = np.zeros(numx)
            cbell = 1000
            while cbell > critconv:
                # Calculate expected values for each action
                v1_0 = profit0 + param['disfact'] * (iptran0 @ v0)  # Value of being inactive
                v1_1 = profit1 + param['disfact'] * (iptran1 @ v0)  # Value of being active

                # Apply logit smoothing for numerical stability
                v1 = np.column_stack((v1_0, v1_1)) / param['sigmaeps']
                maxv1 = np.max(v1, axis=1, keepdims=True)
                v1_centered = v1 - maxv1  # Subtract max for numerical stability

                # Calculate logsum term (expected maximum of the utility shocks)
                logsumexp = np.log(np.sum(np.exp(v1_centered), axis=1))
                v1 = param['sigmaeps'] * (maxv1.flatten() + logsumexp)

                # Check convergence
                cbell = np.max(np.abs(v1 - v0))
                v0 = v1.copy()  # Update value function

            # Updating Probabilities for Firm i - calculate choice-specific values
            v1_0 = profit0 + param['disfact'] * (iptran0 @ v0)
            v1_1 = profit1 + param['disfact'] * (iptran1 @ v0)

            # Calculate choice probabilities (logit formula)
            v1 = np.column_stack((v1_0, v1_1)) / param['sigmaeps']
            maxv1 = np.max(v1, axis=1, keepdims=True)
            v1_centered = v1 - maxv1

            # Update probability of being active (action 1)
            prob1[:, i] = np.exp(v1_centered[:, 1]) / np.sum(np.exp(v1_centered), axis=1)

        # Update convergence criterion and probabilities
        criter = np.max(np.abs(prob1 - prob0))
        prob0 = prob1.copy()
        iter_count += 1

    # Check for convergence and compute steady state distribution if converged
    if criter > critconv:
        dconv = 0
        psteady1 = 0
        if param['verbose']:
            print("----------------------------------------------------------------------------------------")
            print(f"         CONVERGENCE NOT ACHIEVED AFTER {iter_count} BEST RESPONSE ITERATIONS")
            print("----------------------------------------------------------------------------------------")
    else:
        ptrana = np.ones((numx, numa))
        for i in range(nplayer):
            # Create expanded matrices for proper broadcasting
            ppi = prob1[:, i].reshape(-1, 1)  # Make column vector
            ppi1 = np.tile(ppi, (1, numa))
            ppi0 = 1 - ppi1

            mi = np.tile(aval[:, i], (numx, 1))
            mi0 = 1 - mi

            # Update transition probability
            ptrana = ptrana * (ppi1 ** mi) * (ppi0 ** mi0)

        # Create full transition matrix for steady state calculation
        ptrans_expanded = np.kron(param['ptrans'], np.ones((numa, numa)))
        ones_expanded = np.kron(np.ones((1, nums)), ptrana)
        ptrana = ptrans_expanded * ones_expanded

        # Calculate steady state distribution
        criter = 1000
        psteady0 = (1/numx) * np.ones(numx)
        while criter > critconv:
            psteady1 = ptrana.T @ psteady0
            criter = np.max(np.abs(psteady1 - psteady0))
            psteady0 = psteady1.copy()

        if param['verbose']:
            print("----------------------------------------------------------------------------------------")
            print(f"         CONVERGENCE ACHIEVED AFTER {iter_count} BEST RESPONSE ITERATIONS")
            print("----------------------------------------------------------------------------------------")
            print("         EQUILIBRIUM PROBABILITIES")
            print(prob1)
            print("----------------------------------------------------------------------------------------")
            print("         STEADY STATE DISTRIBUTION")
            print(psteady1)
            print("----------------------------------------------------------------------------------------")

    return prob1, psteady1, mstate, dconv


import numpy as np

def simdygam(nobs, pchoice, psteady, mstate):
    """
    Simulates data of state and decision variables from the steady-state
    distribution of a Markov Perfect Equilibrium in a dynamic game.

    Parameters:
    -----------
    nobs : int
        Number of simulations (markets)
    pchoice : ndarray
        Matrix of MPE probabilities of entry with nstate rows and nplayer columns
    psteady : ndarray
        Vector with nstate rows containing the steady-state distribution of (s_t, a_{t-1})
    mstate : ndarray
        Matrix with nstate rows and nplayer+1 columns containing state variables (s_t, a_{t-1})

    Returns:
    --------
    aobs : ndarray
        Matrix with nobs rows and nplayer columns containing players' observed choices
    aobs_1 : ndarray
        Matrix with nobs rows and nplayer columns containing players' initial states
    sobs : ndarray
        Vector with nobs rows containing the simulated values of market size s_t
    xobs : ndarray
        Vector with nobs rows containing the indices of the resulting full state vectors
    """
    # Calculate dimensions
    nplay = pchoice.shape[1]
    nums = pchoice.shape[0]
    numa = 2**nplay
    numx = nums // numa

    # Generating draws from the ergodic distribution of (s_t, a_{t-1})
    pbuff1 = np.cumsum(psteady)
    pbuff0 = np.concatenate(([0], np.cumsum(psteady[:-1])))

    uobs = np.random.rand(nobs)
    xobs = np.zeros(nobs, dtype=int)

    # Find which interval each uniform draw falls into
    for i in range(nobs):
        for j in range(nums):
            if pbuff0[j] <= uobs[i] < pbuff1[j]:
                xobs[i] = j
                break

    # Extract state components
    sobs = mstate[xobs, 0]
    aobs_1 = mstate[xobs, 1:nplay+1]

    # Generating draws of a_t given (s_t, a_{t-1})
    pchoice_obs = pchoice[xobs]
    uobs = np.random.rand(nobs, nplay)
    aobs = (uobs <= pchoice_obs).astype(int)

    return aobs, aobs_1, sobs, xobs


import numpy as np
import matplotlib.pyplot as plt

# 1. Selection of the Monte Carlo experiment to implement
numexp = 6       # Total number of Monte Carlo experiments
selexper = 1     # Select a Monte Carlo experiment to run (1 to numexp)

# 2. Values of Parameters and Other Constants
nobs = 400       # Number of markets (observations)
nplayer = 5      # Number of players

# Fixed costs for each firm in all experiments
theta_fc = np.zeros((numexp, nplayer))
theta_fc[:, 0] = -1.9   # Fixed cost for firm 1 in all experiments
theta_fc[:, 1] = -1.8   # Fixed cost for firm 2 in all experiments
theta_fc[:, 2] = -1.7   # Fixed cost for firm 3 in all experiments
theta_fc[:, 3] = -1.6   # Fixed cost for firm 4 in all experiments
theta_fc[:, 4] = -1.5   # Fixed cost for firm 5 in all experiments

# Parameter values for all experiments
theta_rs = 1.0 * np.ones(numexp)  # theta_rs for each experiment
disfact = 0.95 * np.ones(numexp)  # discount factor for each experiment
sigmaeps = 1.0 * np.ones(numexp)  # std. dev. epsilon for each experiment

# Points of support and transition probability of market size
sval = np.arange(1, 6)  # Support of market size
numsval = len(sval)     # Number of possible market sizes
nstate = numsval * (2**nplayer)  # Number of points in the state space

# Transition probability matrix for market size
ptrans = np.array([
    [0.8, 0.2, 0.0, 0.0, 0.0],
    [0.2, 0.6, 0.2, 0.0, 0.0],
    [0.0, 0.2, 0.6, 0.2, 0.0],
    [0.0, 0.0, 0.2, 0.6, 0.2],
    [0.0, 0.0, 0.0, 0.2, 0.8]
])

# Parameters that differ across experiments
theta_rn = np.array([0.0, 1.0, 2.0, 1.0, 1.0, 1.0])  # Values of theta_rn
theta_ec = np.array([1.0, 1.0, 1.0, 0.0, 2.0, 4.0])  # Values of theta_ec

# Select parameters for the chosen experiment
theta_fc = theta_fc[selexper-1, :]
theta_rs = theta_rs[selexper-1]
disfact = disfact[selexper-1]
sigmaeps = sigmaeps[selexper-1]
theta_ec = theta_ec[selexper-1]
theta_rn = theta_rn[selexper-1]

# Vector with true values of parameters
trueparam = np.concatenate([theta_fc, [theta_rs, theta_rn, theta_ec, disfact, sigmaeps]])

# Structure for storing parameters and settings
param = {
    'theta_fc': theta_fc,
    'theta_rs': theta_rs,
    'theta_rn': theta_rn,
    'theta_ec': theta_ec,
    'disfact': disfact,
    'sigmaeps': sigmaeps,
    'sval': sval,
    'ptrans': ptrans,
    'verbose': 1
}

# Set random seed
np.random.seed(20150403)

# 3. Computing a Markov Perfect Equilibrium of the Dynamic Game
maxiter = 200    # Maximum number of Policy iterations
prob0 = 0.5 * np.random.rand(nstate, nplayer)
pequil, psteady, vstate, dconv = mpeprob(prob0, maxiter, param)

# 4. Simulating Data from the Equilibrium
nobsfordes = 50000
aobs, aobs_1, sobs = simdygam(nobsfordes, pequil, psteady, vstate)[:3]

# Calculate and report descriptive statistics
print("\n")
print("*****************************************************************************************")
print("   DESCRIPTIVE STATISTICS FROM THE EQUILIBRIUM")
print(f"   BASED ON {nobsfordes} OBSERVATIONS")
print("\n")
print("   TABLE 2 OF THE PAPER AGUIRREGABIRIA AND MIRA (2007)")
print("*****************************************************************************************")
print("\n")

# Number of active firms in market at t and t-1
nf = np.sum(aobs, axis=1)
nf_1 = np.sum(aobs_1, axis=1)

# Regression of (number of firms t) on (number of firms t-1)
X = np.column_stack((np.ones(nobsfordes), nf_1))
beta = np.linalg.lstsq(X, nf, rcond=None)[0]
bareg_nf = beta[1]  # Estimate of autoregressive parameter

# Calculate entries, exits, and excess turnover
entries = np.sum(aobs * (1-aobs_1), axis=1)
exits = np.sum((1-aobs) * aobs_1, axis=1)
excess = np.mean(entries + exits - np.abs(entries - exits))

# Correlation between entries and exits
corr_ent_exit = np.corrcoef(entries, exits)[0, 1]

# Frequencies of being active
freq_active = np.mean(aobs, axis=0)

# Print the descriptive statistics
print('\n')
print('----------------------------------------------------------------------------------------')
print(f'       (1)    Average number of active firms   = {np.mean(nf):12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (2)    Std. Dev. number of firms        = {np.std(nf):12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (3)    Regression N[t] on N[t-1]        = {bareg_nf:12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (4)    Average number of entrants       = {np.mean(entries):12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (5)    Average number of exits          = {np.mean(exits):12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (6)    Excess turnover (in # of firms)  = {excess:12.4f}')
print('----------------------------------------------------------------------------------------')
print(f'       (7)    Correlation entries and exits    = {corr_ent_exit:12.4f}')
print('----------------------------------------------------------------------------------------')
print('       (8)    Probability of being active      =')
print(freq_active)
print('----------------------------------------------------------------------------------------')
print('\n')

# Optional: Create a visualization of the number of active firms
plt.figure(figsize=(10, 6))
plt.hist(nf, bins=range(nplayer+2), alpha=0.7)
plt.title(f'Distribution of Number of Active Firms (Experiment {selexper})')
plt.xlabel('Number of Active Firms')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(f'active_firms_exp{selexper}.png')
plt.close()

print(f"Experiment {selexper} completed successfully.")
print(f"Parameters: theta_rn = {theta_rn}, theta_ec = {theta_ec}")
