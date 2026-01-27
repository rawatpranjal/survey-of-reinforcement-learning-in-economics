# ### Single Agent


import numpy as np

# Parameters
valuation = 50
discount_rate = 0.999
learning_rate = 0.01
epsilon_start = 0.99
epsilon_min = 0.01
epsilon_decay = 0.9999999
num_rounds = 10
num_actions = 5
num_games = 10_000_000
verbose_interval = 10000

actions = np.arange(num_actions)  # Action space 0-10
q_table = np.random.uniform(valuation,valuation+1,(num_rounds + 1, 2, len(actions)))  # Q-table
p = np.arange(num_actions,0,-1)**3.0
p /= p.sum()
print(p)

def train_agent():
    global epsilon_start
    reward_history_q = []
    reward_history_random = []
    epsilon = epsilon_start

    for game in range(num_games):
        t = num_rounds
        s = 0
        current_bid_q = 0
        current_bid_random = 0

        while t > 0:
            state_q = (t, s)
            action_q = np.random.choice(actions) if np.random.rand() < epsilon else actions[np.argmax(q_table[state_q])]
            current_bid_q += action_q
            action_random = np.random.choice(actions, p=p)
            current_bid_random += action_random
            s = 1 if current_bid_q >= current_bid_random else 0
            t -= 1
        reward_q = (valuation - current_bid_q) if (current_bid_q > current_bid_random) and (current_bid_q <= valuation) else 0
        reward_random = (valuation - current_bid_random) if (current_bid_random > current_bid_q) and (current_bid_random <= valuation) else 0
        reward_history_q.append(reward_q)
        reward_history_random.append(reward_random)

        if (game + 1) % verbose_interval == 0:
            print(f"Game {game + 1}: Epsilon: {epsilon:.4f}, Avg Q-Learn: {np.mean(reward_history_q[-verbose_interval:])}, Avg Random: {np.mean(reward_history_random[-verbose_interval:])}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        state_q = (num_rounds, s)  # Reset state for Q-learning update
        update_q(state_q, action_q, reward_q, (t, s), t == 1)

    return q_table, reward_history_q, reward_history_random

def update_q(state, action, reward, next_state, done):
    future = 0 if done else discount_rate * np.max(q_table[next_state])
    q_table[state][action] += learning_rate * (reward + future - q_table[state][action])

q_table, reward_history_q, reward_history_random = train_agent()


import matplotlib.pyplot as plt
import numpy as np
window_size = 100000
rolling_avg_q = np.convolve(reward_history_q, np.ones(window_size)/window_size, mode='valid')
rolling_avg_random = np.convolve(reward_history_random, np.ones(window_size)/window_size, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(rolling_avg_q, label='Q-Learning Agent')
plt.plot(rolling_avg_random, label='Random Bot')
plt.xlabel('Epochs (x100 games)')
plt.ylabel('Average Reward')
plt.title('Learning Curve: Average Reward Over Time with Rolling Averages')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def simulate_game(q_table):
    t = num_rounds
    s = 0
    current_bid_q = 0
    current_bid_random = 0
    highest_bids_q = []
    highest_bids_random = []

    while t > 0:
        state_q = (t, s)
        action_q = actions[np.argmax(q_table[state_q])]  # Choose best action from Q-table
        current_bid_q += action_q
        highest_bids_q.append(current_bid_q)

        action_random = np.random.choice(actions, p=p)
        current_bid_random += action_random
        highest_bids_random.append(current_bid_random)

        s = 1 if current_bid_q >= current_bid_random else 0
        t -= 1

    reward_q = (valuation - current_bid_q) if (current_bid_q > current_bid_random) and (current_bid_q <= valuation) else 0
    reward_random = (valuation - current_bid_random) if (current_bid_random > current_bid_q) and (current_bid_random <= valuation) else 0

    # Plotting the highest bids in correct round order with markers
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_rounds+1), highest_bids_q, 'x-', label='Q-Learning Agent Highest Bids', markersize=8)
    plt.plot(range(1, num_rounds+1), highest_bids_random, 'o-', label='Random Bot Highest Bids', markersize=5)
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.title('Highest Bid Progression Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Reward for Q-Learning Agent: {reward_q}")
    print(f"Reward for Random Bot: {reward_random}")

# Example of calling the function with a pre-trained q_table
simulate_game(q_table)


import matplotlib.pyplot as plt
import numpy as np

# Assuming q_table is the Q-table from the trained Q-learning model
num_rounds = 10  # Total number of rounds, from 10 to 1

# Extract optimal actions for each state of 's' across all rounds
optimal_actions_s0 = [np.argmax(q_table[t, 0]) for t in range(1, num_rounds+1)]
optimal_actions_s1 = [np.argmax(q_table[t, 1]) for t in range(1, num_rounds+1)]

# Plot the optimal actions
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds+1), optimal_actions_s0, 'o-', label='Optimal Actions when s=0')
plt.plot(range(1, num_rounds+1), optimal_actions_s1, 'x-', label='Optimal Actions when s=1')
plt.gca().invert_xaxis()  # Invert x-axis to show decreasing rounds from left to right
plt.xlabel('Remaining Rounds (t)')
plt.ylabel('Optimal Action')
plt.title('Optimal Actions from Q-Table for Different States of s')
plt.legend()
plt.grid(True)
plt.show()


def interactive_game(q_table):
    t = num_rounds
    s = 0  # Initial state: no one has the highest bid
    current_bid_q = 0
    current_bid_your = 0
    past_bids_q = []

    print("Starting the game. You will bid against the Q-learning bot.")

    while t > 0:
        state_q = (t, s)
        action_q = actions[np.argmax(q_table[state_q])]  # Bot selects the best action based on the Q-table
        current_bid_q += action_q
        past_bids_q.append(action_q)  # Store this round's bid for future reference

        # You enter your bid
        print(f"Round {num_rounds - t + 1}")
        print(f"Past bids by bot: {past_bids_q[:-1]}")  # Show all past bids except the current one
        print(f"Currently holding the highest bid: {'You' if s == 0 else 'Bot'}")

        try:
            bid = int(input("Enter your bid (0-4): "))
            if bid < 0 or bid > 10:
                print("Invalid bid. Please enter a value between 0 and 10.")
                continue
        except ValueError:
            print("Invalid input. Please enter an integer value.")
            continue

        current_bid_your += bid
        s = 1 if current_bid_q >= current_bid_your else 0  # Update the state based on who has the highest bid
        t -= 1  # Decrease the round

        print(f"Your total bid so far: {current_bid_your}\n")

    reward_q = (valuation - current_bid_q) if (current_bid_q > current_bid_your) and (current_bid_q <= valuation) else 0

    # Output results
    print("Game over.")
    print("Q-learning bot's total bid:", current_bid_q)
    print("Your total bid:", current_bid_your)
    print("Reward for Q-Learning Agent:", reward_q)

# Call this function to start an interactive game
interactive_game(q_table)


# ### Eligibility Traces


import numpy as np

# Parameters
valuation = 50
discount_rate = 0.999
learning_rate = 0.01
lambda_ = 0.9  # Eligibility trace decay parameter
epsilon_start = 0.99
epsilon_min = 0.01
epsilon_decay = 0.99995
num_rounds = 10
num_actions = 5
num_games = 100000
verbose_interval = 10000

actions = np.arange(num_actions)  # Action space
q_table = np.random.uniform(valuation, valuation + 1, (num_rounds + 1, 2, len(actions)))  # Q-table
p = np.arange(num_actions, 0, -1)**3.0
p /= p.sum()

def train_agent():
    global epsilon_start, q_table  # Add q_table to global declarations

    reward_history_q = []
    reward_history_random = []
    epsilon = epsilon_start

    for game in range(num_games):
        # Initialize eligibility traces
        e_trace = np.zeros_like(q_table)

        t = num_rounds
        s = 0
        current_bid_q = 0
        current_bid_random = 0

        while t > 0:
            state_q = (t, s)
            if np.random.rand() < epsilon:
                action_q = np.random.choice(actions)
            else:
                action_q = actions[np.argmax(q_table[state_q])]

            current_bid_q += action_q
            action_random = np.random.choice(actions, p=p)
            current_bid_random += action_random
            s = 1 if current_bid_q >= current_bid_random else 0

            # Update eligibility trace
            e_trace *= discount_rate * lambda_
            e_trace[state_q][action_q] += 1

            if t == 1:  # Reward only on the final round
                reward_q = (valuation - current_bid_q) if (current_bid_q > current_bid_random) and (current_bid_q <= valuation) else 0
                reward_random = (valuation - current_bid_random) if (current_bid_random > current_bid_q) and (current_bid_random <= valuation) else 0

                # Q-value and eligibility trace update
                delta = reward_q - q_table[state_q][action_q]
                q_table += learning_rate * delta * e_trace

            t -= 1

        reward_history_q.append(reward_q)
        reward_history_random.append(reward_random)

        if (game + 1) % verbose_interval == 0:
            print(f"Game {game + 1}: Epsilon: {epsilon:.4f}, Avg Q-Learn: {np.mean(reward_history_q[-verbose_interval:])}, Avg Random: {np.mean(reward_history_random[-verbose_interval:])}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, reward_history_q, reward_history_random

# Then train the agent
q_table, reward_history_q, reward_history_random = train_agent()


import matplotlib.pyplot as plt
import numpy as np
window_size = 10000
rolling_avg_q = np.convolve(reward_history_q, np.ones(window_size)/window_size, mode='valid')
rolling_avg_random = np.convolve(reward_history_random, np.ones(window_size)/window_size, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(rolling_avg_q, label='Q-Learning Agent')
plt.plot(rolling_avg_random, label='Random Bot')
plt.xlabel('Epochs (x100 games)')
plt.ylabel('Average Reward')
plt.title('Learning Curve: Average Reward Over Time with Rolling Averages')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def simulate_game(q_table):
    t = num_rounds
    s = 0
    current_bid_q = 0
    current_bid_random = 0
    highest_bids_q = []
    highest_bids_random = []

    while t > 0:
        state_q = (t, s)
        action_q = actions[np.argmax(q_table[state_q])]  # Choose best action from Q-table
        current_bid_q += action_q
        highest_bids_q.append(current_bid_q)

        action_random = np.random.choice(actions, p=p)
        current_bid_random += action_random
        highest_bids_random.append(current_bid_random)

        s = 1 if current_bid_q >= current_bid_random else 0
        t -= 1

    reward_q = (valuation - current_bid_q) if (current_bid_q > current_bid_random) and (current_bid_q <= valuation) else 0
    reward_random = (valuation - current_bid_random) if (current_bid_random > current_bid_q) and (current_bid_random <= valuation) else 0

    # Plotting the highest bids in correct round order with markers
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_rounds+1), highest_bids_q, 'x-', label='Q-Learning Agent Highest Bids', markersize=8)
    plt.plot(range(1, num_rounds+1), highest_bids_random, 'o-', label='Random Bot Highest Bids', markersize=5)
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.title('Highest Bid Progression Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Reward for Q-Learning Agent: {reward_q}")
    print(f"Reward for Random Bot: {reward_random}")

# Example of calling the function with a pre-trained q_table
simulate_game(q_table)


import matplotlib.pyplot as plt
import numpy as np

# Assuming q_table is the Q-table from the trained Q-learning model
num_rounds = 10  # Total number of rounds, from 10 to 1

# Extract optimal actions for each state of 's' across all rounds
optimal_actions_s0 = [np.argmax(q_table[t, 0]) for t in range(1, num_rounds+1)]
optimal_actions_s1 = [np.argmax(q_table[t, 1]) for t in range(1, num_rounds+1)]

# Plot the optimal actions
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds+1), optimal_actions_s0, 'o-', label='Optimal Actions when s=0')
plt.plot(range(1, num_rounds+1), optimal_actions_s1, 'x-', label='Optimal Actions when s=1')
plt.gca().invert_xaxis()  # Invert x-axis to show decreasing rounds from left to right
plt.xlabel('Remaining Rounds (t)')
plt.ylabel('Optimal Action')
plt.title('Optimal Actions from Q-Table for Different States of s')
plt.legend()
plt.grid(True)
plt.show()


# ### vs Random-Creeper-OneShotter


# Fixing the probabilities issue by ensuring they sum to 1 for the creeper bot's random action

import numpy as np

# Parameters
val = 100
disc = 0.999
lr = 0.01
lam = 0.9
eps_start = 0.99
eps_min = 0.01
eps_decay = 0.999999  # Reduced epsilon decay
rounds = 10
actions = 10
games = 3000000
verbose = 100000

acts = np.arange(actions)
q_tab = np.random.uniform(val, val + 1, (rounds + 1, 2, 2, actions))

def random_bot_type():
    return np.random.choice(["creeper", "sniper"])

def random_bot_action(bot_type, t, holds_bid):
    if bot_type == "creeper":
        if holds_bid:
            probs = [0.6] + [0.1] * (actions - 1)
            return np.random.choice(acts, p=probs/np.sum(probs))
        else:
            probs = [0.1] * (actions - 1) + [0.2]
            return np.random.choice(acts, p=probs/np.sum(probs))
    elif bot_type == "sniper" and t == 1:
        return actions - 1
    else:
        return 0

def train_agent():
    global eps_start, q_tab

    r_hist_q = []
    r_hist_rnd = []
    eps = eps_start

    for game in range(games):
        e_trace = np.zeros_like(q_tab)
        bot_type = random_bot_type()
        held_count = 0

        t = rounds
        s = 0
        current_bid_q = 0
        current_bid_rnd = 0

        while t > 0:
            state_q = (t, s, min(1, held_count))  # Ensure held_count is within bounds
            if np.random.rand() < eps:
                act_q = np.random.choice(acts)
            else:
                act_q = acts[np.argmax(q_tab[state_q])]

            current_bid_q += act_q
            act_rnd = random_bot_action(bot_type, t, current_bid_rnd >= current_bid_q)
            current_bid_rnd += act_rnd
            s = 1 if current_bid_q >= current_bid_rnd else 0
            if s == 1:
                held_count += 1

            e_trace *= disc * lam
            e_trace[state_q][act_q] += 1

            if t == 1:
                reward_q = (val - current_bid_q) if (current_bid_q > current_bid_rnd) and (current_bid_q <= val) else 0
                reward_rnd = (val - current_bid_rnd) if (current_bid_rnd > current_bid_q) and (current_bid_rnd <= val) else 0
                delta = reward_q - q_tab[state_q][act_q]
                q_tab += lr * delta * e_trace

            t -= 1

        r_hist_q.append(reward_q)
        r_hist_rnd.append(reward_rnd)

        if (game + 1) % verbose == 0:
            print(f"Game {game + 1}: Eps: {eps:.4f}, Avg Q: {np.mean(r_hist_q[-verbose:]):.4f}, Avg Rnd: {np.mean(r_hist_rnd[-verbose:]):.4f}")

        eps = max(eps_min, eps * eps_decay)

    return q_tab, r_hist_q, r_hist_rnd

q_tab, r_hist_q, r_hist_rnd = train_agent()


import numpy as np
import matplotlib.pyplot as plt

def simulate_game(q_table, bot_type="creeper"):
    t = rounds
    s = 0
    held_count = 0
    current_bid_q = 0
    current_bid_rnd = 0
    highest_bids_q = []
    highest_bids_rnd = []

    while t > 0:
        state_q = (t, s, min(1, held_count))  # Ensure held_count is within bounds
        action_q = acts[np.argmax(q_table[state_q])]  # Choose best action from Q-table
        current_bid_q += action_q
        highest_bids_q.append(current_bid_q)

        action_rnd = random_bot_action(bot_type, t, current_bid_rnd >= current_bid_q)
        current_bid_rnd += action_rnd
        highest_bids_rnd.append(current_bid_rnd)

        s = 1 if current_bid_q >= current_bid_rnd else 0
        if s == 1:
            held_count += 1
        t -= 1

    reward_q = (val - current_bid_q) if (current_bid_q > current_bid_rnd) and (current_bid_q <= val) else 0
    reward_rnd = (val - current_bid_rnd) if (current_bid_rnd > current_bid_q) and (current_bid_rnd <= val) else 0

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds+1), highest_bids_q, 'x-', label='Q-Learning Agent Highest Bids', markersize=8)
    plt.plot(range(1, rounds+1), highest_bids_rnd, 'o-', label=f'{bot_type.capitalize()} Bot Highest Bids', markersize=5)
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.title(f'Highest Bid Progression Over Rounds against {bot_type.capitalize()} Bot')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Reward for Q-Learning Agent: {reward_q}")
    print(f"Reward for {bot_type.capitalize()} Bot: {reward_rnd}")

# Example of calling the function with a pre-trained q_table against a creeper bot
simulate_game(q_tab, bot_type="creeper")

# Example of calling the function with a pre-trained q_table against a sniper bot
simulate_game(q_tab, bot_type="sniper")


# ### Vs 3 Types


import numpy as np
import matplotlib.pyplot as plt

# Parameters
val = 100
disc = 0.999
lr = 0.01
lam = 0.9
eps_start = 0.99
eps_min = 0.01
eps_decay = 0.999999  # Reduced epsilon decay
rounds = 10
actions = 10
games = 3000000
verbose = 100000

acts = np.arange(actions)
q_tab = np.random.uniform(val, val + 1, (rounds + 1, 2, 2, actions))

def random_bot_type():
    return np.random.choice(["creeper", "sniper", "one_shotter"], p=[0.33, 0.33, 0.34])

def random_bot_action(bot_type, t, holds_bid):
    if bot_type == "creeper":
        if holds_bid:
            probs = [0.6] + [0.1] * (actions - 1)
            return np.random.choice(acts, p=probs/np.sum(probs))
        else:
            probs = [0.1] * (actions - 1) + [0.2]
            return np.random.choice(acts, p=probs/np.sum(probs))
    elif bot_type == "sniper" and t == 1:
        return actions - 1
    elif bot_type == "one_shotter" and t == rounds:
        return actions - 3  # One-shotter bids fairly high early on
    else:
        return 0

def train_agent():
    global eps_start, q_tab

    r_hist_q = []
    r_hist_rnd = []
    eps = eps_start

    for game in range(games):
        e_trace = np.zeros_like(q_tab)
        bot_type = random_bot_type()
        held_count = 0

        t = rounds
        s = 0
        current_bid_q = 0
        current_bid_rnd = 0

        while t > 0:
            state_q = (t, s, min(1, held_count))  # Ensure held_count is within bounds
            if np.random.rand() < eps:
                act_q = np.random.choice(acts)
            else:
                act_q = acts[np.argmax(q_tab[state_q])]

            current_bid_q += act_q
            act_rnd = random_bot_action(bot_type, t, current_bid_rnd >= current_bid_q)
            current_bid_rnd += act_rnd
            s = 1 if current_bid_q >= current_bid_rnd else 0
            if s == 1:
                held_count += 1

            e_trace *= disc * lam
            e_trace[state_q][act_q] += 1

            if t == 1:
                reward_q = (val - current_bid_q) if (current_bid_q > current_bid_rnd) and (current_bid_q <= val) else 0
                reward_rnd = (val - current_bid_rnd) if (current_bid_rnd > current_bid_q) and (current_bid_rnd <= val) else 0
                delta = reward_q - q_tab[state_q][act_q]
                q_tab += lr * delta * e_trace

            t -= 1

        r_hist_q.append(reward_q)
        r_hist_rnd.append(reward_rnd)

        if (game + 1) % verbose == 0:
            print(f"Game {game + 1}: Eps: {eps:.4f}, Avg Q: {np.mean(r_hist_q[-verbose:]):.4f}, Avg Rnd: {np.mean(r_hist_rnd[-verbose:]):.4f}")

        eps = max(eps_min, eps * eps_decay)

    return q_tab, r_hist_q, r_hist_rnd

q_tab, r_hist_q, r_hist_rnd = train_agent()

def simulate_game(q_table, bot_type="creeper"):
    t = rounds
    s = 0
    held_count = 0
    current_bid_q = 0
    current_bid_rnd = 0
    highest_bids_q = []
    highest_bids_rnd = []

    while t > 0:
        state_q = (t, s, min(1, held_count))  # Ensure held_count is within bounds
        action_q = acts[np.argmax(q_table[state_q])]  # Choose best action from Q-table
        current_bid_q += action_q
        highest_bids_q.append(current_bid_q)

        action_rnd = random_bot_action(bot_type, t, current_bid_rnd >= current_bid_q)
        current_bid_rnd += action_rnd
        highest_bids_rnd.append(current_bid_rnd)

        s = 1 if current_bid_q >= current_bid_rnd else 0
        if s == 1:
            held_count += 1
        t -= 1

    reward_q = (val - current_bid_q) if (current_bid_q > current_bid_rnd) and (current_bid_q <= val) else 0
    reward_rnd = (val - current_bid_rnd) if (current_bid_rnd > current_bid_q) and (current_bid_rnd <= val) else 0

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds+1), highest_bids_q, 'x-', label='Q-Learning Agent Highest Bids', markersize=8)
    plt.plot(range(1, rounds+1), highest_bids_rnd, 'o-', label=f'{bot_type.capitalize()} Bot Highest Bids', markersize=5)
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.title(f'Highest Bid Progression Over Rounds against {bot_type.capitalize()} Bot')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Reward for Q-Learning Agent: {reward_q}")
    print(f"Reward for {bot_type.capitalize()} Bot: {reward_rnd}")

# Example of calling the function with a pre-trained q_table against a creeper bot
simulate_game(q_tab, bot_type="creeper")
simulate_game(q_tab, bot_type="sniper")
simulate_game(q_tab, bot_type="one_shotter")


simulate_game(q_tab, bot_type="creeper")
simulate_game(q_tab, bot_type="sniper")
simulate_game(q_tab, bot_type="one_shotter")
