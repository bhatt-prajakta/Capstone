"""
Test script for the PortfolioEnv environment.

This script uses actual data files to test the environment with random actions
to ensure it functions correctly.
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
    Load real data from the data directory.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (financial_data, sentiment_data, economic_data, price_data)
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


def test_environment():
    """
    Test the portfolio environment with random actions using real data.
    """
    # Define test parameters
    tickers = ['GOOGL', 'PFE', 'TSMC', 'WFC']  # Based on available ratio files

    # Load real data
    print("Loading real data...")
    financial_data, sentiment_data, economic_data, price_data = load_real_data("data")

    # Create environment (sector mapping is built into the environment)
    print("Creating environment...")
    env = PortfolioEnv(
        financial_data=financial_data,
        sentiment_data=sentiment_data,
        economic_data=economic_data,
        price_data=price_data,
        tickers=tickers,
        window_size=4,  # Quarterly data
        max_steps=20
    )

    # Reset environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Environment date range: {min(env.dates)} to {max(env.dates)}")

    # Run simulation with random actions
    print("\nRunning simulation with random actions...")
    done = False
    total_steps = 0
    rewards = []
    portfolio_values = [env.portfolio_value]

    while not done and total_steps < 20:
        # Select random action (0 to n_assets + 1 for predefined allocations)
        action = random.randint(0, len(tickers) + 1)

        # Take action
        next_state, reward, done, info = env.step(action)

        # Record results
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])

        # Print progress
        if total_steps % 2 == 0:
            print(f"\nStep {total_steps}: Action={action}, Reward={reward:.4f}, "
                  f"Portfolio Value=${info['portfolio_value']:.2f}, "
                  f"Date={info['date'].strftime('%Y-%m-%d')}")

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
    env, portfolio_values, rewards = test_environment()