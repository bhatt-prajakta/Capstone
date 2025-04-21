"""
Training script for DQN Reinforcement Learning Agent.

This script trains a DQN agent on real financial data for portfolio optimization.
It loads financial, sentiment, and economic data, creates a portfolio environment,
and trains a DQN agent to make investment decisions.

Usage:
    python train_dqn.py [--episodes EPISODES] [--seed SEED] [--save_path SAVE_PATH]

Arguments:
    --episodes: Number of episodes to train for (default: 100)
    --seed: Random seed for reproducibility (default: None)
    --save_path: Path to save the trained model (default: 'trained_models/dqn_agent.pth')
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Import project modules
from portfolio_rl.models.dqn.environment import PortfolioEnv
from portfolio_rl.models.dqn.agent import DQNAgent
from portfolio_rl.models.dqn.reward import RewardCalculator

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RATIO_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "ratios")
MODELS_DIR = os.path.join(PROJECT_ROOT, "portfolio_rl/models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(train_period_end=None):
    """
    Load all necessary data for training the DQN agent.

    Args:
        train_period_end: Optional datetime to filter data for training period

    Returns:
        tuple: (financial_data, sentiment_data, economic_data, tickers)
    """
    print("Loading data for training...")

    # Load sentiment data
    sentiment_data_path = os.path.join(
        PROCESSED_DATA_DIR, "sector_sentiment_with_metrics_quarterly_2014_2024.csv"
    )
    print(f"Loading sentiment data from: {sentiment_data_path}")
    sentiment_data = pd.read_csv(sentiment_data_path)
    sentiment_data.columns = sentiment_data.columns.str.lower().str.replace(" ", "_")
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    print(f"Loaded sentiment data: {sentiment_data.shape}")

    # Load economic data
    economic_data_path = os.path.join(
        PROCESSED_DATA_DIR, "economic_indicators_quarterly_2014_2024.csv"
    )
    print(f"Loading economic data from: {economic_data_path}")
    economic_data = pd.read_csv(economic_data_path)
    economic_data.columns = economic_data.columns.str.lower().str.replace(" ", "_")
    economic_data["date"] = pd.to_datetime(economic_data["date"])
    print(f"Loaded economic data: {economic_data.shape}")

    # Get list of available ratio files
    ratio_files = [f for f in os.listdir(RATIO_DATA_DIR) if f.endswith("_ratios.csv")]
    tickers = [f.split("_")[0] for f in ratio_files]
    print(f"Found {len(tickers)} tickers with financial ratios")

    # Load and consolidate financial ratios
    financial_data_list = []
    for ticker in tickers:
        try:
            ratio_path = os.path.join(RATIO_DATA_DIR, f"{ticker}_ratios.csv")
            ratio_data = pd.read_csv(ratio_path)
            ratio_data["ticker"] = ticker
            ratio_data["date"] = pd.to_datetime(ratio_data["date"])
            financial_data_list.append(ratio_data)
        except FileNotFoundError:
            print(f"Warning: Ratio file for {ticker} not found")

    # Consolidate financial data
    if financial_data_list:
        financial_data = pd.concat(financial_data_list, ignore_index=True)
        print(f"Consolidated financial data: {financial_data.shape}")
    else:
        raise ValueError("No financial ratio files found!")

    # Filter data for training period if specified
    if train_period_end:
        print(f"Filtering data for training period ending at {train_period_end}")
        financial_data = financial_data[financial_data["date"] <= train_period_end]
        sentiment_data = sentiment_data[sentiment_data["date"] <= train_period_end]
        economic_data = economic_data[economic_data["date"] <= train_period_end]

    return financial_data, sentiment_data, economic_data, tickers


def train_agent(
    financial_data,
    sentiment_data,
    economic_data,
    tickers,
    n_episodes=100,
    seed=None,
    save_path=None,
):
    """
    Train a DQN agent for portfolio optimization.

    Args:
        financial_data: DataFrame with financial ratios
        sentiment_data: DataFrame with sentiment metrics
        economic_data: DataFrame with economic indicators
        tickers: List of stock tickers
        n_episodes: Number of episodes to train for
        seed: Random seed for reproducibility
        save_path: Path to save the trained model

    Returns:
        tuple: (agent, training_history)
    """
    print(f"\nTraining DQN agent for {n_episodes} episodes...")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment
    env = PortfolioEnv(
        financial_data=financial_data,
        sentiment_data=sentiment_data,
        economic_data=economic_data,
        tickers=tickers,
        window_size=4,  # For quarterly data
        max_steps=252,  # Approximately one trading year
        use_old_step_api=True,  # For compatibility with the agent
    )

    # Get state and action dimensions from environment
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        state, _ = reset_result  # New gym API
    else:
        state = reset_result  # Old gym API

    state_dim = len(state)

    # For DQN, we need discrete actions
    # We'll use predefined allocations (equal weight, single asset, etc.)
    action_dim = len(tickers) + 2  # Equal weight, each asset, and 60/40 portfolio

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,  # Discount factor
        epsilon_start=1.0,  # Initial exploration rate
        epsilon_end=0.01,  # Final exploration rate
        epsilon_decay=0.995,  # Exploration decay rate
        buffer_size=10000,  # Replay buffer size
        batch_size=64,  # Training batch size
        target_update_freq=10,  # Target network update frequency
    )

    # Training loop
    training_history = {
        "episode_rewards": [],
        "portfolio_values": [],
        "epsilon_values": [],
        "portfolio_allocations": [],
    }

    for episode in range(n_episodes):
        # Reset environment
        reset_result = env.reset(seed=seed)
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # New gym API
        else:
            state = reset_result  # Old gym API

        episode_reward = 0
        episode_steps = 0
        done = False

        # Store initial portfolio value
        initial_portfolio_value = env.portfolio_value

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            step_result = env.step(action)

            if len(step_result) == 4:
                # Old gym API: (next_state, reward, done, info)
                next_state, reward, done, info = step_result
            else:
                # New gym API: (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # Store experience in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Update agent
            loss = agent.update()

            # Update state and reward
            state = next_state
            episode_reward += reward
            episode_steps += 1

        # Record training metrics
        training_history["episode_rewards"].append(episode_reward)
        training_history["portfolio_values"].append(env.portfolio_value)
        training_history["epsilon_values"].append(agent.epsilon)
        training_history["portfolio_allocations"].append(env.portfolio.copy())

        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            portfolio_return = (env.portfolio_value / initial_portfolio_value - 1) * 100
            print(
                f"Episode {episode + 1}/{n_episodes}, "
                f"Reward: {episode_reward:.4f}, "
                f"Return: {portfolio_return:.2f}%, "
                f"Steps: {episode_steps}, "
                f"Epsilon: {agent.epsilon:.4f}"
            )

            # Portfolio allocation analysis
            print("Portfolio allocation:")
            for i, ticker in enumerate(env.tickers):
                print(f"  {ticker}: {env.portfolio[i] * 100:.2f}%")
            print("-----------------------------------")

    # Save the trained agent
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Agent saved to {save_path}")

    return agent, training_history


def plot_training_results(training_history, save_path=None):
    """
    Plot training results.

    Args:
        training_history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(training_history["episode_rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    # Plot portfolio values
    plt.subplot(3, 1, 2)
    plt.plot(training_history["portfolio_values"])
    plt.title("Final Portfolio Value")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    # Plot epsilon values
    plt.subplot(3, 1, 3)
    plt.plot(training_history["epsilon_values"])
    plt.title("Exploration Rate (Epsilon)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training results plot saved to {save_path}")

    plt.show()


def main():
    """Main function to train the DQN agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for portfolio optimization"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="trained_models/dqn_agent.pth",
        help="Path to save the trained model",
    )
    args = parser.parse_args()

    try:
        train_period_end = "2022-12-31"
        # Load data
        financial_data, sentiment_data, economic_data, all_tickers = load_data(train_period_end)

        # Choose tickers for training
        selected_tickers = [
            "AAPL",
            "AMD",
            "AMZN",
            "AVGO",
            "AX",
            "BAC",
            "BAX",
            "BLK",
            "F",
            "GM",
            "GOOGL",
            "GS",
            "IBM",
            "INTC",
            "JNJ",
            "JPM",
            "META",
            "MRK",
            "MRNA",
            "MS",
            "MSFT",
            "NFLX",
            "NVDA",
            "PFE",
            "STLA",
            "TM",
            "TSLA",
            "TSMC",
            "WFC",
        ]
        # Filter to ensure all selected tickers have data
        selected_tickers = [t for t in selected_tickers if t in all_tickers]

        if not selected_tickers:
            # Fallback to first 5 tickers if none of the selected ones are available
            selected_tickers = all_tickers[:10]

        print(f"Training with tickers: {selected_tickers}")

        # Train agent
        agent, training_history = train_agent(
            financial_data=financial_data,
            sentiment_data=sentiment_data,
            economic_data=economic_data,
            tickers=selected_tickers,
            n_episodes=args.episodes,
            seed=args.seed,
            save_path=args.save_path,
        )

        # Plot training results
        plot_training_results(
            training_history=training_history, save_path="training_results.png"
        )

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
