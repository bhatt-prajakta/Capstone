"""
Test script for the PortfolioEnv environment.

This script uses actual data files to test the environment with random actions
to ensure it functions correctly. It demonstrates how to:

1. Load and prepare data for the environment
2. Initialize the PortfolioEnv environment
3. Interact with the environment using its step() and reset() methods
4. Generate random actions that comply with the environment's action space
5. Track and visualize the results of a simulation

The script is designed to work with the environment.py module in the same directory.
It serves both as a test and as an example of how to use the PortfolioEnv class.

Usage:
    python env_test_script.py

Requirements:
    - NumPy
    - Pandas
    - Matplotlib
    - The environment.py module in the same directory
    - Data files in the expected locations (see load_real_data function)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

# Import the environment
from environment import PortfolioEnv


# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RATIO_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "ratios")

def load_real_data(data_dir=DATA_DIR):
    """
    Load real data from the data directory for use with the PortfolioEnv.

    This function loads:
    1. Price data (stock returns)
    2. Sentiment data (sector sentiment metrics)
    3. Economic indicator data
    4. Financial ratio data for specific tickers

    The function expects specific file names and directory structures.
    See the implementation for details on expected file paths.

    Args:
        data_dir: Path to the root data directory containing processed data

    Returns:
        tuple: (financial_data, sentiment_data, economic_data, price_data)
            - financial_data: DataFrame with financial ratios for each ticker
            - sentiment_data: DataFrame with sentiment metrics by sector
            - economic_data: DataFrame with economic indicators
            - price_data: DataFrame with stock price and return data

    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If no financial ratio files are found
    """
    print("Loading real data files...")

    # Load price data (stock_return_data.csv)
    price_data_path = os.path.join(PROCESSED_DATA_DIR, "stock_return_data.csv")
    print(f"Loading price data from: {price_data_path}")
    price_data = pd.read_csv(price_data_path)
    print(f"Loaded price data: {price_data.shape}")

    # Load sentiment data
    sentiment_data_path = os.path.join(PROCESSED_DATA_DIR, "sector_sentiment_with_metrics_quarterly_2014_2024.csv")
    print(f"Loading sentiment data from: {sentiment_data_path}")
    sentiment_data = pd.read_csv(sentiment_data_path)
    print(f"Loaded sentiment data: {sentiment_data.shape}")

    # Load economic data
    economic_data_path = os.path.join(PROCESSED_DATA_DIR, "economic_indicators_quarterly_2014_2024.csv")
    print(f"Loading economic data from: {economic_data_path}")
    economic_data = pd.read_csv(economic_data_path)
    print(f"Loaded economic data: {economic_data.shape}")

    # Load and consolidate financial ratios for multiple tickers
    ratios_dir = os.path.join(PROCESSED_DATA_DIR, "ratios")
    print(f"Loading financial ratios from: {ratios_dir}")

    # Load and consolidate financial ratios for multiple tickers
    financial_data_list = []
    ratio_files = [
        "TSMC_ratios.csv", "PFE_ratios.csv", "GOOGL_ratios.csv", "WFC_ratios.csv"
    ]

    for ratio_file in ratio_files:
        try:
            ratio_data = pd.read_csv(os.path.join(RATIO_DATA_DIR, ratio_file))
            ticker = ratio_file.split('_')[0]
            ratio_data['ticker'] = ticker
            financial_data_list.append(ratio_data)
            print(f"Loaded {ticker} financial ratios: {ratio_data.shape}")
        except FileNotFoundError:
            print(f"File {ratio_file} not found in {data_dir}")

    # Consolidate financial data
    if financial_data_list:
        financial_data = pd.concat(financial_data_list, ignore_index=True)
        print(f"Consolidated financial data: {financial_data.shape}")
    else:
        raise ValueError("No financial ratio files found!")

    return financial_data, sentiment_data, economic_data, price_data


def test_environment(seed=None, max_steps=20):
    """
    Test the portfolio environment with random actions using real data.

    This function:
    1. Loads real financial, sentiment, and economic data
    2. Creates a PortfolioEnv environment with the loaded data
    3. Prints detailed information about the environment
    4. Runs a simulation with random actions for up to max_steps steps
    5. Plots the results (portfolio value, rewards, and allocations)
    6. Tests the rebalancing trigger function

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        max_steps (int, optional): Maximum number of steps to run. Defaults to 20.

    Returns:
        tuple: (env, portfolio_values, rewards) - The environment instance and history data

    The simulation uses random portfolio allocations at each step to test the
    environment's functionality. This is not an optimal strategy but serves to
    verify that the environment works correctly.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # Define test parameters
    tickers = ['GOOGL', 'PFE', 'TSMC', 'WFC', "META", "AX"]  # Based on available ratio files

    # Load real data
    print("Loading real data...")
    financial_data, sentiment_data, economic_data, price_data = load_real_data("data")

    # Create environment (sector mapping is built into the environment)
    print("Creating environment...")
    env = PortfolioEnv(
        financial_data=financial_data,
        sentiment_data=sentiment_data,
        economic_data=economic_data,
        tickers=tickers,
        window_size=4,  # Quarterly data
        max_steps=max_steps,
        use_old_step_api=True  # Use old step API for backward compatibility
    )

    # Print environment information
    print("\nEnvironment Information:")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Number of assets: {env.n_assets}")
    print(f"Tickers: {env.tickers}")
    print(f"Environment date range: {min(env.dates)} to {max(env.dates)}")
    print(f"Number of time steps: {len(env.dates)}")

    # Reset environment with seed for reproducibility
    state = env.reset(seed=seed)
    print(f"\nInitial state shape: {state.shape}")
    print(f"Expected observation space shape: {env.observation_space.shape}")

    # Check if state shape matches observation space shape
    if state.shape != env.observation_space.shape:
        print(f"Warning: State shape {state.shape} does not match observation space shape {env.observation_space.shape}")
        print("This might cause issues with some RL algorithms that expect consistent state shapes.")

    # Run simulation with random actions
    print("\nRunning simulation with random actions...")
    done = False
    total_steps = 0
    rewards = []
    portfolio_values = [env.portfolio_value]

    while not done and total_steps < max_steps:
        # Select random action
        # Option 1: Use integer actions (0 to n_assets + 1 for predefined allocations)
        # action = random.randint(0, len(tickers) + 1)

        # Option 2: Use continuous actions (random weights that sum to 1)
        # Generate random weights for each asset
        action = np.random.random(len(tickers))
        # Normalize to sum to 1
        action = action / np.sum(action)

        # Take action
        next_state, reward, done, info = env.step(action)

        # Record results
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])

        # Print progress
        if total_steps % 2 == 0:
            # Format action for display (if it's a numpy array)
            action_display = f"[{', '.join([f'{a:.2f}' for a in action])}]" if isinstance(action, np.ndarray) else action

            print(f"\nStep {total_steps}: Reward={reward:.4f}, "
                  f"Portfolio Value=${info['portfolio_value']:.2f}, "
                  f"Date={info['date'].strftime('%Y-%m-%d')}")

            # Print action weights
            print("Action weights:")
            for i, ticker in enumerate(tickers):
                if isinstance(action, np.ndarray) and i < len(action):
                    print(f"  {ticker}: {action[i]:.4f}")
                else:
                    print(f"  {ticker}: N/A")

            # Print sector allocation
            sector_allocation = env.get_sector_allocations()
            print("Sector Allocation:")
            for sector, allocation in sector_allocation.items():
                print(f"  {sector}: {allocation:.4f}")

            # Print individual stock allocation
            print("Individual Stock Allocation:")
            for i, ticker in enumerate(tickers):
                print(f"  {ticker}: {env.portfolio[i]:.4f}")

        # Update state
        state = next_state
        total_steps += 1

    print(f"\nSimulation completed after {total_steps} steps")
    print(f"Final portfolio value: ${env.portfolio_value:.2f}")
    print(f"Total return: {(env.portfolio_value / env.initial_balance - 1) * 100:.2f}%")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Reward standard deviation: {np.std(rewards):.4f}")
    print(f"Maximum reward: {np.max(rewards):.4f}")
    print(f"Minimum reward: {np.min(rewards):.4f}")

    # Print final portfolio allocation
    print("\nFinal Portfolio Allocation:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {env.portfolio[i]:.4f}")

    # Print final sector allocation
    print("\nFinal Sector Allocation:")
    sector_allocation = env.get_sector_allocations()
    for sector, allocation in sector_allocation.items():
        print(f"  {sector}: {allocation:.4f}")

    # Plot results
    plt.figure(figsize=(14, 10))

    # Portfolio value over time
    plt.subplot(3, 1, 1)
    plt.plot(range(len(portfolio_values)), portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)

    # Rewards over time
    plt.subplot(3, 1, 2)
    plt.plot(range(len(rewards)), rewards)
    plt.title('Rewards Over Time')
    plt.xlabel('Step')
    plt.ylabel('Return (%)')
    plt.grid(True)

    # Stock allocations over time
    plt.subplot(3, 1, 3)
    allocations_history = np.array(env.portfolio_history)
    for i, ticker in enumerate(tickers):
        plt.plot(range(len(allocations_history)), allocations_history[:, i], label=ticker)
    plt.title('Stock Allocations Over Time')
    plt.xlabel('Step')
    plt.ylabel('Allocation (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('portfolio_simulation_real_data.png')
    plt.close()

    print("Results plot saved as 'portfolio_simulation_real_data.png'")

    # Test rebalancing logic
    print("\nTesting rebalancing trigger function:")
    for ticker in tickers:
        needs_rebalance = env.check_rebalancing_trigger(ticker)
        print(f"  {ticker} needs rebalancing: {needs_rebalance}")

    return env, portfolio_values, rewards


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test the portfolio environment with random actions.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum number of steps to run')
    args = parser.parse_args()

    try:
        print("Starting portfolio environment test...")
        print(f"Using seed: {args.seed}, max_steps: {args.max_steps}")

        env, portfolio_values, rewards = test_environment(
            seed=args.seed,
            max_steps=args.max_steps
        )

        print("\nTest completed successfully!")
    except FileNotFoundError as e:
        print(f"\nError: Required data file not found: {e}")
        print("Please ensure all required data files are in the expected locations.")
        print("See the load_real_data function for expected file paths.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check your data files and environment configuration.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and fix the issue.")
