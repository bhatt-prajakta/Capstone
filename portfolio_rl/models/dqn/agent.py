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
from typing import List, Dict, Tuple, Optional, Union

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

    def sample(self, batch_size: int) -> List[Experience]:
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


class PortfolioEnv:
    """Environment for portfolio optimization using financial data."""

    def __init__(
        self,
        financial_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        economic_data: pd.DataFrame,
        tickers: List[str],
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        window_size: int = 10,
        max_steps: int = 252,  # Approximately one trading year
    ):
        """Initialize the portfolio environment.

        Args:
            financial_data: DataFrame containing financial ratios
            sentiment_data: DataFrame containing news sentiment metrics
            economic_data: DataFrame containing economic indicators
            tickers: List of stock tickers to consider
            initial_balance: Initial portfolio balance
            transaction_cost: Cost per transaction as a fraction of trade value
            window_size: Number of past time steps to include in state
            max_steps: Maximum number of steps per episode
        """
        self.financial_data = financial_data
        self.sentiment_data = sentiment_data
        self.economic_data = economic_data
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.max_steps = max_steps

        self.n_assets = len(tickers)
        self.reset()

    def reset(self):
        """Reset the environment to initial state.

        Returns:
            Initial state observation
        """
        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.n_assets)  # Holdings of each asset
        self.portfolio_value = self.initial_balance
        self.current_step = self.window_size
        self.done = False

        return self._get_state()

    def _get_state(self):
        """Construct the state representation.

        Returns:
            State vector containing financial ratios, sentiment metrics, and economic indicators
        """
        # Get financial features for each ticker
        financial_features = []
        for ticker in self.tickers:
            ticker_data = self.financial_data[self.financial_data["ticker"] == ticker]
            if not ticker_data.empty:
                # Get the most recent window_size data points
                recent_data = ticker_data.iloc[
                    self.current_step - self.window_size : self.current_step
                ]
                # Extract relevant financial ratios
                ratios = recent_data[
                    [
                        "current_ratio",
                        "quick_ratio",
                        "debt_to_equity_ratio",
                        "return_on_assets",
                        "return_on_equity",
                        "gross_profit_margin",
                        "operating_profit_margin",
                        "net_profit_margin",
                    ]
                ].values.flatten()
                financial_features.extend(ratios)

        # Get sentiment features
        sentiment_features = []
        recent_sentiment = self.sentiment_data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        sentiment_metrics = recent_sentiment[
            ["weighted_sentiment_Automotive", "avg_momentum_Automotive", "sentiment_dispersion_Automotive"]
        ].values.flatten()
        sentiment_features.extend(sentiment_metrics)

        # Get economic indicators
        economic_features = []
        recent_economic = self.economic_data.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        economic_indicators = recent_economic[
            ["RGDP", "UR", "UC", "X10YT", "VIX"]
        ].values.flatten()
        economic_features.extend(economic_indicators)

        # Combine all features
        state = np.concatenate(
            [
                financial_features,
                sentiment_features,
                economic_features,
                self.portfolio,  # Current portfolio allocation
                [self.balance / self.initial_balance],  # Normalized cash balance
            ]
        )

        return state

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Action to take (index representing portfolio reallocation)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Map action to portfolio allocation
        new_allocation = self._action_to_allocation(action)

        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(new_allocation)

        # Update portfolio
        self.portfolio = new_allocation
        self.balance -= transaction_cost

        # Move to next time step
        self.current_step += 1

        # Calculate new portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self._calculate_portfolio_value()

        # Calculate reward (portfolio return)
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value

        # Check if episode is done
        self.done = (self.current_step >= len(self.financial_data) - 1) or (
            self.current_step >= self.max_steps
        )

        # Get next state
        next_state = self._get_state()

        # Additional info
        info = {
            "portfolio_value": self.portfolio_value,
            "transaction_cost": transaction_cost,
        }

        return next_state, reward, self.done, info

    def _action_to_allocation(self, action):
        """Convert action index to portfolio allocation.

        Args:
            action: Action index

        Returns:
            Array of portfolio allocations
        """
        # This is a simplified implementation
        # In a real application, you would define a more sophisticated mapping
        # from actions to portfolio allocations

        # Example: Predefined allocations
        allocations = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # All in asset 0
            np.array([0.0, 1.0, 0.0, 0.0]),  # All in asset 1
            np.array([0.0, 0.0, 1.0, 0.0]),  # All in asset 2
            np.array([0.0, 0.0, 0.0, 1.0]),  # All in asset 3
            np.array([0.25, 0.25, 0.25, 0.25]),  # Equal allocation
            np.array([0.4, 0.3, 0.2, 0.1]),  # Weighted allocation 1
            np.array([0.1, 0.2, 0.3, 0.4]),  # Weighted allocation 2
        ]

        return allocations[action % len(allocations)]

    def _calculate_transaction_cost(self, new_allocation):
        """Calculate transaction cost for portfolio reallocation.

        Args:
            new_allocation: New portfolio allocation

        Returns:
            Transaction cost
        """
        # Calculate the absolute difference between current and new allocations
        allocation_diff = np.abs(self.portfolio - new_allocation)

        # Calculate the total value being reallocated
        value_reallocated = np.sum(allocation_diff) * self.portfolio_value

        # Apply transaction cost
        return value_reallocated * self.transaction_cost

    def _calculate_portfolio_value(self):
        """Calculate the current portfolio value.

        Returns:
            Current portfolio value
        """
        # In a real implementation, this would use actual asset prices
        # For simplicity, we're using a placeholder implementation

        # Get current asset prices (simplified)
        asset_prices = np.ones(self.n_assets)  # Placeholder

        # Calculate portfolio value
        portfolio_value = self.balance + np.sum(self.portfolio * asset_prices)

        return portfolio_value


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
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

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
        state = environment.reset()
        state_dim = len(state)
        action_dim = 7  # Based on your _action_to_allocation method

        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")

        # Create and test agent
        agent = DQNAgent(state_dim, action_dim)
        print("DQN Agent created successfully!")

        # Test a few steps
        for step in range(5):
            action = agent.select_action(state)
            next_state, reward, done, info = environment.step(action)
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