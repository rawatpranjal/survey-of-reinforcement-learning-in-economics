import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Parameters
num_rounds = 5       # Number of rounds in each game
valuation = 25       # Maximum value that agents can bid for an item
num_actions = 5      # Total number of discrete actions (bids) each agent can take
actions = np.arange(num_actions)  # Possible actions (bids)
epsilon_start = 1.0   # Initial exploration rate (η)
epsilon_min = 0.01    # Minimum exploration rate
epsilon_decay = 0.9999  # Exploration decay rate (small decay)
learning_rate = 0.00025  # Learning rate for neural networks (Q-network and policy network)
discount_factor = 0.99   # Discount factor for future rewards (γ)
anticipatory_param = 0.1  # Probability of using the RL policy (η)
batch_size = 128       # Batch size for training (recommended)
memory_capacity_rl = 500000  # Capacity for RL replay buffer (M_RL)
memory_capacity_sl = 500000  # Capacity for SL replay buffer (M_SL)
num_games = 500_000    # Total number of games to train the agents
verbose_interval = 1000  # Interval at which training progress is printed
update_target_every = 500  # Target network update frequency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience Replay Buffers
class CircularReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg for arg in args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

class ReservoirReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.n = 0  # Total number of items seen

    def push(self, *args):
        args = tuple(arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
        self.n += 1
        if len(self.memory) < self.capacity:
            self.memory.append(args)
        else:
            index = random.randint(0, self.n - 1)
            if index < self.capacity:
                self.memory[index] = args

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# Neural Network Architectures
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# NFSPAgent Class
class NFSPAgent:
    def __init__(self, num_rounds, num_actions, anticipatory_param, agent_id):
        self.num_rounds = num_rounds
        self.num_actions = num_actions
        self.anticipatory_param = anticipatory_param
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.agent_id = agent_id

        # Input size includes the state representation: [rounds_left, holding_highest_bid (0/1), current_bid]
        input_size = 3
        self.q_network = QNetwork(input_size, num_actions).to(device)
        self.q_network_target = QNetwork(input_size, num_actions).to(device)
        self.policy_network = PolicyNetwork(input_size, num_actions).to(device)

        self.q_network_target.load_state_dict(self.q_network.state_dict())
        self.q_network_target.eval()

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Replay Buffers
        self.replay_memory_rl = CircularReplayBuffer(memory_capacity_rl)
        self.replay_memory_sl = ReservoirReplayBuffer(memory_capacity_sl)

        self.steps_done = 0

    def select_action(self, state):
        state_tensor = torch.tensor([state], device=device)
        if random.random() < self.anticipatory_param:
            # Best Response Strategy (ε-greedy)
            if random.random() < self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                with torch.no_grad():
                    action = self.q_network(state_tensor).argmax().item()
            use_best_response = True
        else:
            # Average Strategy (Policy Network)
            with torch.no_grad():
                probabilities = self.policy_network(state_tensor)
                action = np.random.choice(self.num_actions, p=probabilities.cpu().numpy()[0])
            use_best_response = False
        return action, use_best_response

    def add_experience_rl(self, state, action, reward, next_state, done):
        self.replay_memory_rl.push(torch.tensor(state, device=device),
                                   torch.tensor([action], device=device),
                                   torch.tensor([reward], device=device),
                                   torch.tensor(next_state, device=device),
                                   torch.tensor([done], device=device))

    def add_experience_sl(self, state, action):
        self.replay_memory_sl.push(torch.tensor(state, device=device),
                                   torch.tensor([action], device=device))

    def train_q_network(self):
        if len(self.replay_memory_rl) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_memory_rl.sample(batch_size)
        states = torch.stack(states).to(device)
        actions = torch.cat(actions).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.cat(dones).to(device).float()

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.q_network_target(next_states).max(1)[0]
            target_q_values = rewards + (discount_factor * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def train_policy_network(self):
        if len(self.replay_memory_sl) < batch_size:
            return
        states, actions = self.replay_memory_sl.sample(batch_size)
        states = torch.stack(states).to(device)
        actions = torch.cat(actions).to(device)

        probabilities = self.policy_network(states)
        loss = nn.functional.nll_loss(torch.log(probabilities), actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def update_target_network(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

# Korean Auction Environment
class KoreanAuctionEnv:
    def __init__(self, num_rounds, num_actions, valuation):
        self.num_rounds = num_rounds
        self.num_actions = num_actions
        self.valuation = valuation
        self.reset()

    def reset(self):
        self.t = self.num_rounds
        self.s = 0  # 0 if Agent 1 holds the highest bid, 1 if Agent 2
        self.current_bid_q1 = 0
        self.current_bid_q2 = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # State representation: [rounds_left, holding_highest_bid (0 or 1), current_bid]
        state_q1 = [self.t, int(self.s == 0), self.current_bid_q1]
        state_q2 = [self.t, int(self.s == 1), self.current_bid_q2]
        return state_q1, state_q2

    def step(self, action_q1, action_q2):
        self.current_bid_q1 += action_q1
        self.current_bid_q2 += action_q2

        # Update who holds the highest bid
        if self.current_bid_q1 >= self.current_bid_q2:
            self.s = 0
        else:
            self.s = 1

        self.t -= 1
        if self.t <= 0:
            self.done = True
            # Calculate rewards
            if self.current_bid_q1 >= self.current_bid_q2 and self.current_bid_q1 <= self.valuation:
                reward_q1 = self.valuation - self.current_bid_q1
            else:
                reward_q1 = 0
            if self.current_bid_q2 > self.current_bid_q1 and self.current_bid_q2 <= self.valuation:
                reward_q2 = self.valuation - self.current_bid_q2
            else:
                reward_q2 = 0
        else:
            reward_q1 = 0
            reward_q2 = 0

        next_state_q1, next_state_q2 = self.get_state()
        return (next_state_q1, next_state_q2), (reward_q1, reward_q2), self.done

# Training function
def train_agents(agent1, agent2, env):
    reward_history_q1 = []
    reward_history_q2 = []
    steps = 0

    for game in range(num_games):
        state_q1, state_q2 = env.reset()
        while True:
            # Agent 1 selects action
            action_q1, use_br_q1 = agent1.select_action(state_q1)
            # Agent 2 selects action
            action_q2, use_br_q2 = agent2.select_action(state_q2)

            next_state, rewards, done = env.step(action_q1, action_q2)
            next_state_q1, next_state_q2 = next_state
            reward_q1, reward_q2 = rewards

            # Store transitions for RL
            agent1.add_experience_rl(state_q1, action_q1, reward_q1, next_state_q1, done)
            agent2.add_experience_rl(state_q2, action_q2, reward_q2, next_state_q2, done)

            # Store behavior for SL if used best response strategy
            if use_br_q1:
                agent1.add_experience_sl(state_q1, action_q1)
            if use_br_q2:
                agent2.add_experience_sl(state_q2, action_q2)

            # Train agents
            agent1.train_q_network()
            agent1.train_policy_network()
            agent2.train_q_network()
            agent2.train_policy_network()

            # Update target networks periodically
            steps += 1
            if steps % update_target_every == 0:
                agent1.update_target_network()
                agent2.update_target_network()

            if done:
                agent1.decay_epsilon()
                agent2.decay_epsilon()
                reward_history_q1.append(reward_q1)
                reward_history_q2.append(reward_q2)
                break

            state_q1 = next_state_q1
            state_q2 = next_state_q2

        if (game + 1) % verbose_interval == 0:
            avg_reward_q1 = np.mean(reward_history_q1[-verbose_interval:])
            avg_reward_q2 = np.mean(reward_history_q2[-verbose_interval:])
            total_reward = np.sum(reward_history_q1[-verbose_interval:]) + np.sum(reward_history_q2[-verbose_interval:])

            print(f"Game {game + 1}: Avg Reward Agent 1: {avg_reward_q1:.2f}, Avg Reward Agent 2: {avg_reward_q2:.2f}, Total Reward: {total_reward}")

            # Visualize best response and average response bidding functions + a sample game
            visualize_verbose_plots(agent1, env)

    return agent1, agent2, reward_history_q1, reward_history_q2

def visualize_verbose_plots(agent1, env):
    rounds = np.arange(1, num_rounds + 1)

    # 1. Best response bidding function (RL policy)
    best_response_actions = []
    for r in rounds:
        state = [num_rounds - r, 1, 0]  # Example state: round left, holding highest bid, current bid = 0
        action, _ = agent1.select_action(state)
        best_response_actions.append(action)

    # 2. Average response bidding function (SL policy)
    avg_response_actions = []
    for r in rounds:
        state = [num_rounds - r, 1, 0]  # Example state
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=device)
            probabilities = agent1.policy_network(state_tensor)
            avg_action = np.argmax(probabilities.cpu().numpy()[0])
            avg_response_actions.append(avg_action)

    # 3. Sample game between two agents
    highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2 = play_game(agent1, agent1, env)

    plt.figure(figsize=(15, 5))

    # Plot 1: Best response bidding function vs rounds
    plt.subplot(1, 3, 1)
    plt.plot(rounds, best_response_actions, marker='o')
    plt.title("Bidder 1's Best Response Bidding Function")
    plt.xlabel('Round')
    plt.ylabel('Best Response (Bid Increment)')
    plt.grid(True)

    # Plot 2: Average response bidding function vs rounds
    plt.subplot(1, 3, 2)
    plt.plot(rounds, avg_response_actions, marker='x')
    plt.title("Bidder 1's Average Response Bidding Function")
    plt.xlabel('Round')
    plt.ylabel('Average Response (Bid Increment)')
    plt.grid(True)

    # Plot 3: Sample game between two agents
    plt.subplot(1, 3, 3)
    plt.plot(rounds, highest_bids_q1, 'x-', label='Agent 1 Highest Bids', markersize=8)
    plt.plot(rounds, highest_bids_q2, 'o-', label='Agent 2 Highest Bids', markersize=5)
    plt.title('Sample Game: Highest Bids Progression')
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Play and visualize game function
def play_game(agent1, agent2, env):
    state_q1, state_q2 = env.reset()
    highest_bids_q1 = []
    highest_bids_q2 = []
    actions_q1 = []
    actions_q2 = []
    while True:
        action_q1, _ = agent1.select_action(state_q1)
        action_q2, _ = agent2.select_action(state_q2)
        actions_q1.append(action_q1)
        actions_q2.append(action_q2)
        next_state, rewards, done = env.step(action_q1, action_q2)
        next_state_q1, next_state_q2 = next_state
        highest_bids_q1.append(env.current_bid_q1)
        highest_bids_q2.append(env.current_bid_q2)
        if done:
            reward_q1, reward_q2 = rewards
            break
        state_q1 = next_state_q1
        state_q2 = next_state_q2
    return highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2

# Initialize environment and agents
env = KoreanAuctionEnv(num_rounds, num_actions, valuation)
agent1 = NFSPAgent(num_rounds, num_actions, anticipatory_param, agent_id=1)
agent2 = NFSPAgent(num_rounds, num_actions, anticipatory_param, agent_id=2)

# Train the agents
agent1, agent2, reward_history_q1, reward_history_q2 = train_agents(agent1, agent2, env)

# Play and visualize a game
highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2 = play_game(agent1, agent2, env)
visualize_verbose_plots(agent1, env)

print(f"Reward for Agent 1: {reward_q1}")
print(f"Reward for Agent 2: {reward_q2}")


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Parameters
num_rounds = 2       # Number of rounds in each game
valuation = 10       # Maximum value that agents can bid for an item
num_actions = 5      # Total number of discrete actions (bids) each agent can take
actions = np.arange(num_actions)  # Possible actions (bids)
epsilon_start = 0.06   # Initial exploration rate (ε)
epsilon_min = 0.0    # Minimum exploration rate
epsilon_decay = 0.99999  # Exploration decay based on the inverse square root of iterations
learning_rate_rl = 0.1   # Learning rate for RL (Q-network)
learning_rate_sl = 0.005  # Learning rate for SL (policy network)
discount_factor = 0.99   # Discount factor for future rewards (γ)
anticipatory_param = 0.1  # Probability of using the RL policy (η)
batch_size = 128       # Batch size for training
memory_capacity_rl = 200000  # Capacity for RL replay buffer (M_RL)
memory_capacity_sl = 2000000  # Capacity for SL replay buffer (M_SL)
num_games = 10_000_000  # Total number of games to train the agents
verbose_interval = 1000  # Interval at which training progress is printed
update_target_every = 300  # Target network update frequency (refit every 300 updates)
gradient_updates_per_step = 2  # Two gradient updates per step (one for RL and one for SL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience Replay Buffers
class CircularReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg for arg in args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

class ReservoirReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.n = 0  # Total number of items seen

    def push(self, *args):
        args = tuple(arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
        self.n += 1
        if len(self.memory) < self.capacity:
            self.memory.append(args)
        else:
            index = random.randint(0, self.n - 1)
            if index < self.capacity:
                self.memory[index] = args

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# Neural Network Architectures
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# NFSPAgent Class
class NFSPAgent:
    def __init__(self, num_rounds, num_actions, anticipatory_param, agent_id):
        self.num_rounds = num_rounds
        self.num_actions = num_actions
        self.anticipatory_param = anticipatory_param
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.agent_id = agent_id

        # Input size includes the state representation: [rounds_left, holding_highest_bid (0/1), current_bid]
        input_size = 3
        self.q_network = QNetwork(input_size, num_actions).to(device)
        self.q_network_target = QNetwork(input_size, num_actions).to(device)
        self.policy_network = PolicyNetwork(input_size, num_actions).to(device)

        self.q_network_target.load_state_dict(self.q_network.state_dict())
        self.q_network_target.eval()

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Replay Buffers
        self.replay_memory_rl = CircularReplayBuffer(memory_capacity_rl)
        self.replay_memory_sl = ReservoirReplayBuffer(memory_capacity_sl)

        self.steps_done = 0

    def select_action(self, state):
        state_tensor = torch.tensor([state], device=device)
        if random.random() < self.anticipatory_param:
            # Best Response Strategy (ε-greedy)
            if random.random() < self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                with torch.no_grad():
                    action = self.q_network(state_tensor).argmax().item()
            use_best_response = True
        else:
            # Average Strategy (Policy Network)
            with torch.no_grad():
                probabilities = self.policy_network(state_tensor)
                action = np.random.choice(self.num_actions, p=probabilities.cpu().numpy()[0])
            use_best_response = False
        return action, use_best_response

    def add_experience_rl(self, state, action, reward, next_state, done):
        self.replay_memory_rl.push(torch.tensor(state, device=device),
                                   torch.tensor([action], device=device),
                                   torch.tensor([reward], device=device),
                                   torch.tensor(next_state, device=device),
                                   torch.tensor([done], device=device))

    def add_experience_sl(self, state, action):
        self.replay_memory_sl.push(torch.tensor(state, device=device),
                                   torch.tensor([action], device=device))

    def train_q_network(self):
        if len(self.replay_memory_rl) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_memory_rl.sample(batch_size)
        states = torch.stack(states).to(device)
        actions = torch.cat(actions).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.cat(dones).to(device).float()

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.q_network_target(next_states).max(1)[0]
            target_q_values = rewards + (discount_factor * next_q_values * (1 - dones))

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def train_policy_network(self):
        if len(self.replay_memory_sl) < batch_size:
            return
        states, actions = self.replay_memory_sl.sample(batch_size)
        states = torch.stack(states).to(device)
        actions = torch.cat(actions).to(device)

        probabilities = self.policy_network(states)
        loss = nn.functional.nll_loss(torch.log(probabilities), actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def update_target_network(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

# Korean Auction Environment
class KoreanAuctionEnv:
    def __init__(self, num_rounds, num_actions, valuation):
        self.num_rounds = num_rounds
        self.num_actions = num_actions
        self.valuation = valuation
        self.reset()

    def reset(self):
        self.t = self.num_rounds
        self.s = 0  # 0 if Agent 1 holds the highest bid, 1 if Agent 2
        self.current_bid_q1 = 0
        self.current_bid_q2 = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # State representation: [rounds_left, holding_highest_bid (0 or 1), current_bid]
        state_q1 = [self.t, int(self.s == 0), self.current_bid_q1]
        state_q2 = [self.t, int(self.s == 1), self.current_bid_q2]
        return state_q1, state_q2

    def step(self, action_q1, action_q2):
        self.current_bid_q1 += action_q1
        self.current_bid_q2 += action_q2

        # Update who holds the highest bid
        if self.current_bid_q1 >= self.current_bid_q2:
            self.s = 0
        else:
            self.s = 1

        self.t -= 1
        if self.t <= 0:
            self.done = True
            # Calculate rewards
            if self.current_bid_q1 >= self.current_bid_q2 and self.current_bid_q1 <= self.valuation:
                reward_q1 = self.valuation - self.current_bid_q1
            else:
                reward_q1 = 0
            if self.current_bid_q2 > self.current_bid_q1 and self.current_bid_q2 <= self.valuation:
                reward_q2 = self.valuation - self.current_bid_q2
            else:
                reward_q2 = 0
        else:
            reward_q1 = 0
            reward_q2 = 0

        next_state_q1, next_state_q2 = self.get_state()
        return (next_state_q1, next_state_q2), (reward_q1, reward_q2), self.done

# Training function
def train_agents(agent1, agent2, env):
    reward_history_q1 = []
    reward_history_q2 = []
    steps = 0

    for game in range(num_games):
        state_q1, state_q2 = env.reset()
        while True:
            # Agent 1 selects action
            action_q1, use_br_q1 = agent1.select_action(state_q1)
            # Agent 2 selects action
            action_q2, use_br_q2 = agent2.select_action(state_q2)

            next_state, rewards, done = env.step(action_q1, action_q2)
            next_state_q1, next_state_q2 = next_state
            reward_q1, reward_q2 = rewards

            # Store transitions for RL
            agent1.add_experience_rl(state_q1, action_q1, reward_q1, next_state_q1, done)
            agent2.add_experience_rl(state_q2, action_q2, reward_q2, next_state_q2, done)

            # Store behavior for SL if used best response strategy
            if use_br_q1:
                agent1.add_experience_sl(state_q1, action_q1)
            if use_br_q2:
                agent2.add_experience_sl(state_q2, action_q2)

            # Train agents
            agent1.train_q_network()
            agent1.train_policy_network()
            agent2.train_q_network()
            agent2.train_policy_network()

            # Update target networks periodically
            steps += 1
            if steps % update_target_every == 0:
                agent1.update_target_network()
                agent2.update_target_network()

            if done:
                agent1.decay_epsilon()
                agent2.decay_epsilon()
                reward_history_q1.append(reward_q1)
                reward_history_q2.append(reward_q2)
                break

            state_q1 = next_state_q1
            state_q2 = next_state_q2

        if (game + 1) % verbose_interval == 0:
            avg_reward_q1 = np.mean(reward_history_q1[-verbose_interval:])
            avg_reward_q2 = np.mean(reward_history_q2[-verbose_interval:])
            total_reward = (np.sum(reward_history_q1[-verbose_interval:]) + np.sum(reward_history_q2[-verbose_interval:])) / (2 * verbose_interval)
            print(f"Game {game + 1}: Avg Reward Agent 1: {avg_reward_q1:.2f}, Avg Reward Agent 2: {avg_reward_q2:.2f}, Epsilon: {agent1.epsilon:.4f}, Total Avg Reward: {total_reward}")

            # Visualize best response and average response bidding functions + a sample game
            visualize_verbose_plots(agent1, env)

    return agent1, agent2, reward_history_q1, reward_history_q2

def visualize_verbose_plots(agent1, env):
    rounds = np.arange(1, num_rounds + 1)

    # 1. Best response bidding function (RL policy)
    best_response_actions = []
    for r in rounds:
        state = [num_rounds - r, 1, 0]  # Example state: round left, holding highest bid, current bid = 0
        action, _ = agent1.select_action(state)
        best_response_actions.append(action)

    # 2. Average response bidding function (SL policy)
    avg_response_actions = []
    for r in rounds:
        state = [num_rounds - r, 1, 0]  # Example state
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=device)
            probabilities = agent1.policy_network(state_tensor)
            avg_action = np.argmax(probabilities.cpu().numpy()[0])
            avg_response_actions.append(avg_action)

    # 3. Sample game between two agents
    highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2 = play_game(agent1, agent1, env)

    plt.figure(figsize=(15, 5))

    # Plot 1: Best response bidding function vs rounds
    plt.subplot(1, 3, 1)
    plt.plot(rounds, best_response_actions, marker='o')
    plt.title("Bidder 1's Best Response Bidding Function")
    plt.xlabel('Round')
    plt.ylabel('Best Response (Bid Increment)')
    plt.grid(True)

    # Plot 2: Average response bidding function vs rounds
    plt.subplot(1, 3, 2)
    plt.plot(rounds, avg_response_actions, marker='x')
    plt.title("Bidder 1's Average Response Bidding Function")
    plt.xlabel('Round')
    plt.ylabel('Average Response (Bid Increment)')
    plt.grid(True)

    # Plot 3: Sample game between two agents
    plt.subplot(1, 3, 3)
    plt.plot(rounds, highest_bids_q1, 'x-', label='Agent 1 Highest Bids', markersize=8)
    plt.plot(rounds, highest_bids_q2, 'o-', label='Agent 2 Highest Bids', markersize=5)
    plt.title('Sample Game: Highest Bids Progression')
    plt.xlabel('Round')
    plt.ylabel('Highest Bid Amount')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Play and visualize game function
def play_game(agent1, agent2, env):
    state_q1, state_q2 = env.reset()
    highest_bids_q1 = []
    highest_bids_q2 = []
    actions_q1 = []
    actions_q2 = []
    while True:
        action_q1, _ = agent1.select_action(state_q1)
        action_q2, _ = agent2.select_action(state_q2)
        actions_q1.append(action_q1)
        actions_q2.append(action_q2)
        next_state, rewards, done = env.step(action_q1, action_q2)
        next_state_q1, next_state_q2 = next_state
        highest_bids_q1.append(env.current_bid_q1)
        highest_bids_q2.append(env.current_bid_q2)
        if done:
            reward_q1, reward_q2 = rewards
            break
        state_q1 = next_state_q1
        state_q2 = next_state_q2
    return highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2

# Initialize environment and agents
env = KoreanAuctionEnv(num_rounds, num_actions, valuation)
agent1 = NFSPAgent(num_rounds, num_actions, anticipatory_param, agent_id=1)
agent2 = NFSPAgent(num_rounds, num_actions, anticipatory_param, agent_id=2)

# Train the agents
agent1, agent2, reward_history_q1, reward_history_q2 = train_agents(agent1, agent2, env)

# Play and visualize a game
highest_bids_q1, highest_bids_q2, reward_q1, reward_q2, actions_q1, actions_q2 = play_game(agent1, agent2, env)
visualize_verbose_plots(agent1, env)

print(f"Reward for Agent 1: {reward_q1}")
print(f"Reward for Agent 2: {reward_q2}")
