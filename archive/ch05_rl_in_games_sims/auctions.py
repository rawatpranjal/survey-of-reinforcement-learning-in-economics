import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Simple Leduc Poker Game Environment
class LeducPoker:
    """
    A simplified implementation of Leduc Poker:
    - 6 cards (2 suits, 3 ranks)
    - 2 players
    - 1 private card per player
    - 1 community card
    - 2 betting rounds
    - Actions: fold, check/call, raise
    """
    def __init__(self):
        self.cards = [0, 1, 2, 0, 1, 2]  # 0: Jack, 1: Queen, 2: King (two suits)
        self.num_actions = 3  # fold, check/call, raise
        self.reset()

    def reset(self):
        # Shuffle cards
        np.random.shuffle(self.cards)

        # Deal private cards to players
        self.private_cards = [self.cards[0], self.cards[1]]

        # Set community card (revealed in second round)
        self.community_card = self.cards[2]

        # Initialize game state
        self.current_player = 0
        self.round = 0  # 0: first betting round, 1: second betting round
        self.folded = [False, False]
        self.pot = [1, 1]  # Initial antes
        self.round_raises = 0
        self.history = []

        # Create initial observation for first player
        obs = self._get_observation(self.current_player)

        return obs

    def _get_observation(self, player_id):
        """Create observation for the specified player"""
        obs = {
            'private_card': self.private_cards[player_id],
            'community_card': self.community_card if self.round == 1 else -1,
            'pot': self.pot.copy(),
            'round': self.round,
            'current_player': self.current_player,
            'history': self.history.copy(),
            'folded': self.folded.copy()
        }
        return obs

    def step(self, action):
        """
        Execute action and return next state, reward, done flag
        action: 0 (fold), 1 (check/call), 2 (raise)
        """
        player = self.current_player
        opponent = 1 - player

        # Record action in history
        self.history.append(action)

        # Process action
        if action == 0:  # Fold
            self.folded[player] = True
            reward = [-self.pot[player], self.pot[player]]
            done = True

        elif action == 1:  # Check/Call
            # Match the bet
            call_amount = self.pot[opponent] - self.pot[player]
            self.pot[player] += call_amount

            # Check if round is over
            if len(self.history) > 1 and (self.history[-2] == 1 or self.round_raises > 0):
                # Both players checked or last raise was called
                if self.round == 0:
                    # Move to next round
                    self.round = 1
                    self.round_raises = 0
                    self.current_player = 0  # First player starts next round
                    self.history = []  # Clear history for new round
                    reward = [0, 0]
                    done = False
                else:
                    # Game is over, showdown
                    reward = self._showdown()
                    done = True
            else:
                # Switch player
                self.current_player = opponent
                reward = [0, 0]
                done = False

        elif action == 2:  # Raise
            # Simple raise implementation: double the current bet
            raise_amount = self.pot[opponent]
            self.pot[player] = self.pot[opponent] * 2
            self.round_raises += 1
            self.current_player = opponent
            reward = [0, 0]
            done = False

        # Create next observation
        if not done:
            obs = self._get_observation(self.current_player)
        else:
            obs = self._get_observation(player)  # Final observation for terminal state

        return obs, reward, done

    def _showdown(self):
        """Determine winner at showdown"""
        if self.folded[0]:
            return [-self.pot[0], self.pot[0]]
        if self.folded[1]:
            return [self.pot[1], -self.pot[1]]

        player0_card = self.private_cards[0]
        player1_card = self.private_cards[1]

        # Check for pairs
        if player0_card == self.community_card and player1_card != self.community_card:
            # Player 0 has a pair
            return [self.pot[1], -self.pot[1]]
        elif player1_card == self.community_card and player0_card != self.community_card:
            # Player 1 has a pair
            return [-self.pot[0], self.pot[0]]
        elif player0_card == self.community_card and player1_card == self.community_card:
            # Both have pairs, it's a tie
            return [0, 0]
        else:
            # No pairs, higher card wins
            if player0_card > player1_card:
                return [self.pot[1], -self.pot[1]]
            elif player1_card > player0_card:
                return [-self.pot[0], self.pot[0]]
            else:
                # Tie
                return [0, 0]

# Neural Network Models
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SupervisedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SupervisedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

# NFSP Agent Implementation
class NFSPAgent:
    def __init__(self, state_dim, action_dim, nu=0.1, lr_q=0.01, lr_sl=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nu = nu  # Probability of using Q-network for action selection

        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.sl_network = SupervisedNetwork(state_dim, action_dim)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr_q)
        self.sl_optimizer = optim.Adam(self.sl_network.parameters(), lr=lr_sl)

        # Initialize replay buffers
        self.rl_buffer = deque(maxlen=100000)  # RL experience replay buffer
        self.sl_buffer = deque(maxlen=100000)  # Supervised learning reservoir buffer

        # Initialize training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.target_update_freq = 100
        self.training_steps = 0

        # Mode for current episode
        self.use_q_policy = None
        self.reset()

    def reset(self):
        self.last_state = None
        self.last_action = None
        # Determine policy type for this episode
        self.use_q_policy = random.random() < self.nu

    def state_to_tensor(self, state):
        """Convert observation dictionary to tensor format"""
        # Simplified state representation
        # In a real implementation, you would create a more sophisticated state representation
        tensor = [
            state['private_card'] / 2.0,  # Normalize card value
            state['community_card'] / 2.0 if state['community_card'] != -1 else -0.5,
            float(state['pot'][0]) / 10.0,  # Normalize pot size
            float(state['pot'][1]) / 10.0,
            float(state['round']),
            float(state['current_player']),
            1.0 if state['folded'][0] else 0.0,
            1.0 if state['folded'][1] else 0.0
        ]

        # Add one-hot encoding of the history (last 4 actions)
        history = state['history'][-4:] if len(state['history']) > 0 else []
        while len(history) < 4:
            history.append(-1)  # Padding

        for action in history:
            if action == -1:  # Padding
                tensor.extend([0.0, 0.0, 0.0])
            else:
                # One-hot encoding
                one_hot = [0.0, 0.0, 0.0]
                if 0 <= action < 3:
                    one_hot[action] = 1.0
                tensor.extend(one_hot)

        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    def act(self, state):
        """Select action based on the current state"""
        state_tensor = self.state_to_tensor(state)

        if self.use_q_policy:
            # Best response policy (RL policy)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()

            # Store action in the SL reservoir buffer
            if self.last_state is not None:
                self.sl_buffer.append((self.last_state, action))
        else:
            # Average policy (SL policy)
            with torch.no_grad():
                probs = self.sl_network(state_tensor).squeeze(0)
                # Convert to numpy for sampling
                probs_np = probs.numpy()
                action = np.random.choice(self.action_dim, p=probs_np)

        self.last_state = state_tensor
        self.last_action = action
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in the RL replay buffer"""
        state_tensor = self.state_to_tensor(state)

        if next_state is not None:
            next_state_tensor = self.state_to_tensor(next_state)
        else:
            # If terminal state
            next_state_tensor = None

        self.rl_buffer.append((state_tensor, action, reward, next_state_tensor, done))

    def train(self):
        """Train both networks"""
        # Train Q-network
        if len(self.rl_buffer) >= self.batch_size:
            self._train_q_network()

        # Train supervised learning network
        if len(self.sl_buffer) >= self.batch_size:
            self._train_sl_network()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_q_network(self):
        """Train the Q-network using DQN"""
        batch = random.sample(self.rl_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Prepare batch data
        state_batch = torch.cat(states, dim=0)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

        # Compute Q-values for current states and actions
        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute target Q-values
        next_q_values = torch.zeros((self.batch_size, 1), dtype=torch.float32)
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool)

        if any(non_final_mask):
            non_final_next_states = torch.cat([s for s, d in zip(next_states, dones) if not d], dim=0)
            next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]

            next_q_values = next_state_values.unsqueeze(1)

        expected_q_values = reward_batch + (self.gamma * next_q_values)

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def _train_sl_network(self):
        """Train the supervised learning network"""
        batch = random.sample(self.sl_buffer, self.batch_size)
        states, actions = zip(*batch)

        # Prepare batch data
        state_batch = torch.cat(states, dim=0)
        action_batch = torch.tensor(actions, dtype=torch.long)

        # Forward pass
        action_probs = self.sl_network(state_batch)

        # Compute loss (cross-entropy loss)
        loss = nn.CrossEntropyLoss()(action_probs, action_batch)

        # Update
        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()

        return loss.item()

# Training loop example
def train_nfsp():
    # Environment setup
    env = LeducPoker()

    # Agent setup
    state_dim = 20  # 8 base features + 12 for history encoding (4 steps * 3 actions)
    action_dim = env.num_actions
    agents = [
        NFSPAgent(state_dim, action_dim),
        NFSPAgent(state_dim, action_dim)
    ]

    num_episodes = 10000

    for episode in range(num_episodes):
        # Reset environment and agents
        state = env.reset()
        for agent in agents:
            agent.reset()

        done = False

        while not done:
            # Current player takes action
            player = env.current_player
            action = agents[player].act(state)

            # Environment step
            next_state, rewards, done = env.step(action)

            # Store transition for the current player
            agents[player].store_transition(state, action, rewards[player], next_state, done)

            # Update state
            state = next_state

        # Train agents
        for agent in agents:
            agent.train()

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")

    return agents

# Evaluation function
def evaluate_agents(agents, num_games=1000):
    env = LeducPoker()
    wins = [0, 0]
    ties = 0

    for game in range(num_games):
        state = env.reset()
        for agent in agents:
            agent.reset()

        done = False

        while not done:
            player = env.current_player
            action = agents[player].act(state)
            state, rewards, done = env.step(action)

        # Determine winner
        if rewards[0] > 0:
            wins[0] += 1
        elif rewards[1] > 0:
            wins[1] += 1
        else:
            ties += 1

    print(f"Agent 0 wins: {wins[0]}, Agent 1 wins: {wins[1]}, Ties: {ties}")
    return wins, ties

# Run training and evaluation
if __name__ == "__main__":
    # Train NFSP agents
    print("Training NFSP agents...")
    trained_agents = train_nfsp()

    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    evaluate_agents(trained_agents)
