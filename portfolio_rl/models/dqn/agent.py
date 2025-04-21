"""
Deep Q-Network (DQN) Reinforcement Learning Agent for Portfolio Optimization.

This module implements a DQN agent that uses financial ratios, news sentiment,
and economic indicators to make investment decisions.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from portfolio_rl.models.dqn.environment import PortfolioEnv

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Define a named tuple for storing experiences
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling past experiences."""

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a fixed capacity.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            A list of sampled experiences
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for approximating the Q-function."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize the Q-network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent for portfolio optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the DQN agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which epsilon decays
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0

        # Initialize Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            eval_mode: Whether to use evaluation mode (no exploration)

        Returns:
            Selected action index
        """
        if (not eval_mode) and (random.random() < self.epsilon):
            # Exploration: select a random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select the action with highest Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self):
        """Update the Q-network using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)

        # Convert experiences to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = (
            torch.LongTensor([e.action for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )
        rewards = (
            torch.FloatTensor([e.reward for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(
            self.device
        )
        dones = (
            torch.FloatTensor([e.done for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """Save the agent's Q-network to a file.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        """Load the agent's Q-network from a file.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]



def train_dqn_agent(
    env: PortfolioEnv,
    state_dim: int,
    action_dim: int,
    n_episodes: int = 1000,
    max_steps: int = 252,
    save_path: str = "models/dqn_agent.pth",
):
    """Train a DQN agent for portfolio optimization.

    Args:
        env: Portfolio environment
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        n_episodes: Number of episodes to train for
        max_steps: Maximum number of steps per episode
        save_path: Path to save the trained agent

    Returns:
        Trained DQN agent
    """
    agent = DQNAgent(state_dim, action_dim)

    # Training loop
    episode_rewards = []

    for episode in range(n_episodes):
        # Reset environment (handle both old and new gym API)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # New gym API returns (obs, info)
        else:
            state = reset_result     # Old gym API returns just obs

        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action (handle both old and new gym API)
            step_result = env.step(action)

            if len(step_result) == 4:
                # Old gym API: (next_state, reward, done, info)
                next_state, reward, done, _ = step_result
            else:
                # New gym API: (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated

            # Store experience
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Update agent
            loss = agent.update()

            # Update state and reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # Track progress
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}"
            )

    # Save the trained agent
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    return agent


def test_dqn_agent():
    """Test the DQN agent with real project data."""
    # Define paths to different data types
    financial_statements_dir = os.path.join(
        DATA_DIR, "processed", "financial_statements"
    )
    ratios_dir = os.path.join(DATA_DIR, "processed", "ratios")
    economic_indicators_path = os.path.join(
        DATA_DIR, "processed", "economic_indicators_quarterly_2014_2024.csv"
    )
    sentiment_path = os.path.join(
        DATA_DIR, "processed", "sector_sentiment_with_metrics_quarterly_2014_2024.csv"
    )
    stock_returns_path = os.path.join(DATA_DIR, "processed", "stock_return_data.csv")

    try:
        # Load economic and sentiment data
        economic_data = pd.read_csv(economic_indicators_path)
        sentiment_data = pd.read_csv(sentiment_path)

        print("Economic and sentiment data loaded successfully!")
        print(f"Economic data shape: {economic_data.shape}")
        print(f"Sentiment data shape: {sentiment_data.shape}")

        # Standardize column names (convert 'Date' to 'date')
        if 'Date' in economic_data.columns and 'date' not in economic_data.columns:
            economic_data = economic_data.rename(columns={'Date': 'date'})

        if 'Date' in sentiment_data.columns and 'date' not in sentiment_data.columns:
            sentiment_data = sentiment_data.rename(columns={'Date': 'date'})

        # Extract tickers from the ratios directory
        tickers = []
        for filename in os.listdir(ratios_dir):
            if filename.endswith('_ratios.csv'):
                ticker = filename.split('_ratios.csv')[0]
                tickers.append(ticker)

        print(f"Found {len(tickers)} tickers with financial ratios")

        # For testing, limit to a few tickers
        test_tickers = tickers[:4]  # e.g., ['AAPL', 'AMD', 'AMZN', 'AVGO']
        print(f"Testing with tickers: {test_tickers}")

        # Load and consolidate financial ratios for test tickers
        consolidated_ratios = []
        for ticker in test_tickers:
            ratio_path = os.path.join(ratios_dir, f"{ticker}_ratios.csv")
            if os.path.exists(ratio_path):
                ratio_data = pd.read_csv(ratio_path)

                # Standardize date column name
                if 'Date' in ratio_data.columns and 'date' not in ratio_data.columns:
                    ratio_data = ratio_data.rename(columns={'Date': 'date'})

                # Print column names for first ticker to help debug
                if ticker == test_tickers[0]:
                    print(f"Column names in {ticker} ratios: {ratio_data.columns.tolist()}")

                # Convert date to datetime if not already
                ratio_data['date'] = pd.to_datetime(ratio_data['date'])
                ratio_data['ticker'] = ticker
                consolidated_ratios.append(ratio_data)

        # Combine all ratios into one DataFrame
        if consolidated_ratios:
            financial_data = pd.concat(consolidated_ratios, ignore_index=True)
            print(f"Financial ratios consolidated for {len(consolidated_ratios)} tickers")
            print(f"Consolidated financial data shape: {financial_data.shape}")

            # Convert dates in other datasets to datetime for consistency
            economic_data['date'] = pd.to_datetime(economic_data['date'])
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])

            # Sort all data by date
            financial_data = financial_data.sort_values('date')
            economic_data = economic_data.sort_values('date')
            sentiment_data = sentiment_data.sort_values('date')

            # Create list of unique dates from financial data for time steps
            unique_dates = sorted(financial_data['date'].unique())

            print(f"Financial data date range: {financial_data['date'].min()} to {financial_data['date'].max()}")
            print(f"Economic data date range: {economic_data['date'].min()} to {economic_data['date'].max()}")
            print(f"Sentiment data date range: {sentiment_data['date'].min()} to {sentiment_data['date'].max()}")
            print(f"Number of unique dates/time steps: {len(unique_dates)}")
        else:
            print("No financial ratios found for test tickers")
            return

        # Create the environment with your data
        environment = PortfolioEnv(
            financial_data=financial_data,
            sentiment_data=sentiment_data,
            economic_data=economic_data,
            tickers=test_tickers,
            window_size=4,  # For quarterly data, use fewer periods
            max_steps=20,  # Reduce for testing
            #evaluation_frequency=1,  # Evaluate after each step for testing
            #dates=unique_dates  # Pass the sorted unique dates
        )

        # Initialize state to determine state dimension
        reset_result = environment.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # New gym API returns (obs, info)
        else:
            state = reset_result     # Old gym API returns just obs

        state_dim = len(state)

        # Determine action dimension from environment's action space
        if hasattr(environment, 'action_space'):
            if hasattr(environment.action_space, 'n'):
                # Discrete action space
                action_dim = environment.action_space.n
            elif hasattr(environment.action_space, 'shape'):
                # Continuous action space
                action_dim = environment.action_space.shape[0]
            else:
                # Fallback to default
                action_dim = 7
                print("Warning: Could not determine action dimension from environment, using default value of 7")
        else:
            # Fallback to default
            action_dim = 7
            print("Warning: Environment has no action_space attribute, using default action dimension of 7")

        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")

        # Create and test agent
        agent = DQNAgent(state_dim, action_dim)
        print("DQN Agent created successfully!")

        # Test a few steps
        for step in range(5):
            # Select action
            action = agent.select_action(state)

            # Check if action is compatible with environment's action space
            if hasattr(environment, 'action_space'):
                if hasattr(environment.action_space, 'contains'):
                    # For continuous action spaces, we need to convert the discrete action
                    # to a continuous action vector
                    if isinstance(action, (int, np.integer)) and hasattr(environment.action_space, 'shape'):
                        # Convert discrete action to continuous action vector
                        # This is a simple example; you might need a more sophisticated mapping
                        continuous_action = np.zeros(environment.action_space.shape[0])
                        if action < environment.action_space.shape[0]:
                            continuous_action[action] = 1.0
                        else:
                            # Equal allocation as fallback
                            continuous_action = np.ones(environment.action_space.shape[0]) / environment.action_space.shape[0]

                        print(f"Converting discrete action {action} to continuous action {continuous_action}")
                        action = continuous_action

            # Take action (handle both old and new gym API)
            step_result = environment.step(action)

            if len(step_result) == 4:
                # Old gym API: (next_state, reward, done, info)
                next_state, reward, done, info = step_result
            else:
                # New gym API: (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            print(
                f"Step {step + 1} - Action: {action}, Reward: {reward:.4f}, Portfolio Value: {info['portfolio_value']:.2f}, Done: {done}")
            state = next_state

            if done:
                break

        print("Agent test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dqn_agent()
