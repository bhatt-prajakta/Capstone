"""
Testing script for DQN Reinforcement Learning Agent.

This script evaluates a trained DQN agent on real financial data for portfolio optimization.
It loads a trained model, runs it on test data, and visualizes the results.

Usage:
    python test_dqn.py [--model_path MODEL_PATH] [--seed SEED] [--steps STEPS]

Arguments:
    --model_path: Path to the trained model (default: 'trained_models/dqn_agent.pth')
    --seed: Random seed for reproducibility (default: None)
    --steps: Maximum number of steps to run (default: 252, approximately one trading year)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pathlib import Path
import torch
from datetime import datetime

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "portfolio_rl/models/dqn/results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(test_period_start=None):
    """
    Load all necessary data for testing the DQN agent.

    Args:
        test_period_start: Optional datetime to filter data for testing period

    Returns:
        tuple: (financial_data, sentiment_data, economic_data, tickers)
    """
    print("Loading data for testing...")

    # Load sentiment data
    sentiment_data_path = os.path.join(
        PROCESSED_DATA_DIR, "sector_sentiment_with_metrics_quarterly_2014_2024.csv"
    )
    print(f"Loading sentiment data from: {sentiment_data_path}")
    sentiment_data = pd.read_csv(sentiment_data_path)
    sentiment_data.columns = sentiment_data.columns.str.lower().str.replace(" ", "_")
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])

    # Load economic data
    economic_data_path = os.path.join(
        PROCESSED_DATA_DIR, "economic_indicators_quarterly_2014_2024.csv"
    )
    print(f"Loading economic data from: {economic_data_path}")
    economic_data = pd.read_csv(economic_data_path)
    economic_data.columns = economic_data.columns.str.lower().str.replace(" ", "_")
    economic_data["date"] = pd.to_datetime(economic_data["date"])

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

    # Filter data for test period if specified
    if test_period_start:
        print(f"Filtering data for test period starting from {test_period_start}")
        financial_data = financial_data[financial_data["date"] >= test_period_start]
        sentiment_data = sentiment_data[sentiment_data["date"] >= test_period_start]
        economic_data = economic_data[economic_data["date"] >= test_period_start]

    print(f"Data date ranges:")
    print(
        f"Financial data: {financial_data['date'].min()} to {financial_data['date'].max()}"
    )
    print(
        f"Sentiment data: {sentiment_data['date'].min()} to {sentiment_data['date'].max()}"
    )
    print(
        f"Economic data: {economic_data['date'].min()} to {economic_data['date'].max()}"
    )

    return financial_data, sentiment_data, economic_data, tickers


def test_agent(
    agent,
    financial_data,
    sentiment_data,
    economic_data,
    tickers,
    max_steps=252,
    seed=None,
):
    """
    Test a trained DQN agent on the environment.

    Args:
        agent: Trained DQN agent
        financial_data: DataFrame with financial ratios
        sentiment_data: DataFrame with sentiment metrics
        economic_data: DataFrame with economic indicators
        tickers: List of stock tickers
        max_steps: Maximum number of steps to run
        seed: Random seed for reproducibility

    Returns:
        tuple: (env, test_results)
    """
    print(f"\nTesting DQN agent for up to {max_steps} steps...")

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
        max_steps=max_steps,
        use_old_step_api=True,  # For compatibility with the agent
    )

    # Reset environment
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        state, _ = reset_result  # New gym API
    else:
        state = reset_result  # Old gym API

    # Initialize test results
    test_results = {
        "portfolio_values": [env.portfolio_value],
        "rewards": [],
        "actions": [],
        "dates": [env.current_date],
        "allocations": [env.portfolio.copy()],
    }

    # Run test loop
    done = False
    step = 0

    while not done and step < max_steps:
        # Select action (no exploration during testing)
        action = agent.select_action(state, eval_mode=True)

        # Take action
        step_result = env.step(action)

        if len(step_result) == 4:
            # Old gym API: (next_state, reward, done, info)
            next_state, reward, done, info = step_result
        else:
            # New gym API: (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        # Record results
        test_results["portfolio_values"].append(info["portfolio_value"])
        test_results["rewards"].append(reward)
        test_results["actions"].append(action)
        test_results["dates"].append(info["date"])
        test_results["allocations"].append(info["portfolio_allocation"])

        # Print progress
        if step % 10 == 0:
            print(
                f"Step {step}: Date={info['date'].strftime('%Y-%m-%d')}, "
                f"Portfolio Value=${info['portfolio_value']:.2f}, "
                f"Reward={reward:.4f}"
            )

        # Update state
        state = next_state
        step += 1

    # Calculate performance metrics
    initial_value = test_results["portfolio_values"][0]
    final_value = test_results["portfolio_values"][-1]
    total_return = (final_value / initial_value - 1) * 100

    # Calculate annualized return
    days = (test_results["dates"][-1] - test_results["dates"][0]).days
    years = max(days / 365, 0.01)  # Avoid division by zero
    annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100

    # Calculate Sharpe ratio (if we have enough data)
    returns = (
        np.diff(test_results["portfolio_values"])
        / test_results["portfolio_values"][:-1]
    )
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0

    # Calculate maximum drawdown
    peak = test_results["portfolio_values"][0]
    max_drawdown = 0
    for value in test_results["portfolio_values"]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    # Print performance summary
    print("\nPerformance Summary:")
    print(
        f"Testing period: {test_results['dates'][0].strftime('%Y-%m-%d')} to {test_results['dates'][-1].strftime('%Y-%m-%d')}"
    )
    print(f"Initial portfolio value: ${initial_value:.2f}")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")
    print(f"Maximum drawdown: {max_drawdown*100:.2f}%")

    # Add performance metrics to results
    test_results["performance"] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }

    return env, test_results


def plot_test_results(test_results, tickers, save_path=None):
    """
    Plot test results.

    Args:
        test_results: Dict with keys
            - "dates": list of datetime.date or datetime.datetime
            - "portfolio_values": list of floats
            - "rewards": list of floats (len = len(dates)-1)
            - "allocations": list of lists (shape = [len(dates), len(tickers)])
        tickers: List of stock tickers
        save_path: Optional path to save the plot
    """
    dates = test_results["dates"]  # keep as datetime objects

    # Set up figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Locator & formatter for dates
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # Portfolio value
    ax = axes[0]
    ax.plot(dates, test_results["portfolio_values"], lw=2)
    ax.set_title("Portfolio Value Over Time")
    ax.set_ylabel("Value ($)")
    ax.grid(True)

    # Rewards
    ax = axes[1]
    ax.plot(dates[1:], test_results["rewards"], color="tab:orange", lw=2)
    ax.set_title("Rewards Over Time")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # Allocations
    ax = axes[2]
    allocations = np.array(test_results["allocations"])
    for i, ticker in enumerate(tickers):
        ax.plot(dates, allocations[:, i], label=ticker)
    ax.set_title("Asset Allocations Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocation")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(True)

    # Legend outside on the right
    ax.legend(
        ncol=1,
        fontsize="small",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )

    # Apply date formatting to all subplots
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Test results plot saved to {save_path}")

    plt.show()


def compare_to_benchmark(test_results, benchmark_data=None):
    """
    Compare agent performance to benchmark (e.g., S&P 500).

    Args:
        test_results: Dictionary with test metrics
        benchmark_data: Optional DataFrame with benchmark data

    Returns:
        Dictionary with comparison metrics
    """
    # If no benchmark data provided, use equal-weight portfolio as benchmark
    if benchmark_data is None:
        print("No benchmark data provided. Using equal-weight portfolio as benchmark.")

        # Calculate equal-weight portfolio performance
        initial_value = test_results["portfolio_values"][0]
        equal_weight_values = [initial_value]

        # Assume equal weight portfolio grows at 8% annually (historical average)
        annual_return = 0.08
        daily_return = (1 + annual_return) ** (1 / 252) - 1

        for i in range(1, len(test_results["dates"])):
            days_passed = (test_results["dates"][i] - test_results["dates"][i - 1]).days
            growth_factor = (1 + daily_return) ** days_passed
            equal_weight_values.append(equal_weight_values[-1] * growth_factor)

        benchmark_return = (equal_weight_values[-1] / equal_weight_values[0] - 1) * 100
    else:
        # Use provided benchmark data
        # This would require aligning dates and calculating returns
        benchmark_return = 0  # Placeholder

    # Calculate outperformance
    agent_return = test_results["performance"]["total_return"]
    outperformance = agent_return - benchmark_return

    print("\nBenchmark Comparison:")
    print(f"Agent return: {agent_return:.2f}%")
    print(f"Benchmark return: {benchmark_return:.2f}%")
    print(f"Outperformance: {outperformance:.2f}%")

    return {
        "agent_return": agent_return,
        "benchmark_return": benchmark_return,
        "outperformance": outperformance,
    }


def plot_performance_comparison(
    test_results_csv, financial_data, tickers, start_date="2023-01-09", end_date="2024-12-29"
):
    """
    Plot performance comparison between the DQN model and an equally weighted portfolio.
    
    Args:
        test_results_csv: Path to the CSV file with test results
        start_date: Start date for the comparison
        end_date: End date for the comparison
    """
    # Load test results
    results_df = pd.read_csv(test_results_csv)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    # Get model performance data
    model_dates = results_df["date"]
    model_values = results_df["portfolio_value"]
    
    # Normalize to starting value of 1.0
    initial_value = model_values.iloc[0]
    model_returns = model_values / initial_value - 1.0
    
    # Create equal-weight benchmark (8% annual growth)
    annual_return = 0.08
    daily_return = (1 + annual_return) ** (1 / 252) - 1
    
    # Create benchmark series with same dates as model
    benchmark_values = [1.0]  # Start with $1
    for i in range(1, len(model_dates)):
        days_passed = (model_dates.iloc[i] - model_dates.iloc[i-1]).days
        growth_factor = (1 + daily_return) ** days_passed
        benchmark_values.append(benchmark_values[-1] * growth_factor)
    
    benchmark_returns = np.array(benchmark_values) - 1.0
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    
    # Plot equal-weight portfolio returns
    plt.plot(
        model_dates,
        benchmark_returns * 100,  # Convert to percentage
        "b-",
        linewidth=2,
        label="Equal Weight Portfolio"
    )
    
    # Plot DQN model returns
    plt.plot(
        model_dates,
        model_returns * 100,  # Convert to percentage
        "r-",
        linewidth=2,
        label="DQN Model"
    )
    
    # Format the plot
    plt.title("Cumulative Returns Comparison", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend(loc="upper left", fontsize=10)
    
    # Show return percentages
    final_model_return = model_returns.iloc[-1]
    final_benchmark_return = benchmark_returns[-1]
    
    plt.figtext(
        0.15,
        0.02,
        f"DQN Model Return: {final_model_return:.2%}",
        fontsize=12,
        color="red"
    )
    plt.figtext(
        0.55,
        0.02,
        f"Equal Weight Portfolio Return: {final_benchmark_return:.2%}",
        fontsize=12,
        color="blue"
    )
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "performance_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nPerformance Summary:")
    print(f"DQN Model Final Return: {final_model_return:.2%}")
    print(f"Equal Weight Portfolio Return: {final_benchmark_return:.2%}")
    print(f"Outperformance: {(final_model_return - final_benchmark_return):.2%}")
    
    # Calculate Sharpe ratio
    model_daily_returns = model_values.pct_change().dropna()
    sharpe_ratio = model_daily_returns.mean() / model_daily_returns.std() * np.sqrt(252)
    print(f"Model Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return {
        "model_return": final_model_return,
        "benchmark_return": final_benchmark_return,
        "outperformance": final_model_return - final_benchmark_return,
        "sharpe_ratio": sharpe_ratio
    }


def main():
    """Main function to test the DQN agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test a trained DQN agent for portfolio optimization"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dqn/trained_models/dqn_agent.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps", type=int, default=252, help="Maximum number of steps to run"
    )
    args = parser.parse_args()

    try:
        # Check if model exists
        model_path = os.path.join(MODELS_DIR, args.model_path)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please train a model first using train_dqn.py")
            return

        # Load data for testing (use most recent data)
        # For a proper train/test split, we could use a specific date cutoff
        test_period_start = datetime(
            2022, 12, 31
        )  # Use data from 2023 onwards for testing
        financial_data, sentiment_data, economic_data, all_tickers = load_data(
            test_period_start
        )

        # Choose tickers for testing
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

        print(f"Testing with tickers: {selected_tickers}")

        # Create a temporary environment to get state and action dimensions
        temp_env = PortfolioEnv(
            financial_data=financial_data,
            sentiment_data=sentiment_data,
            economic_data=economic_data,
            tickers=selected_tickers,
            window_size=4,
            max_steps=1,
            use_old_step_api=True,
        )

        state = temp_env.reset()
        state_dim = len(state)
        action_dim = len(selected_tickers) + 2  # Same as in training

        # Create agent with the same architecture as during training
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu",  # Use CPU for testing
        )

        # Load trained weights
        print(f"Loading trained model from {model_path}")
        agent.load(model_path)

        # Test the agent
        env, test_results = test_agent(
            agent=agent,
            financial_data=financial_data,
            sentiment_data=sentiment_data,
            economic_data=economic_data,
            tickers=selected_tickers,
            max_steps=args.steps,
            seed=args.seed,
        )

        # Plot test results
        timestamp = datetime.now().strftime("%Y%m%d")
        plot_save_path = os.path.join(RESULTS_DIR, f"test_results_{timestamp}.png")
        plot_test_results(
            test_results=test_results,
            tickers=selected_tickers,
            save_path=plot_save_path,
        )

        # Compare to benchmark
        comparison = compare_to_benchmark(test_results)

        # Save detailed results to CSV
        results_df = pd.DataFrame(
            {
                "date": test_results["dates"],
                "portfolio_value": test_results["portfolio_values"],
            }
        )

        # Add allocation columns
        allocations = np.array(test_results["allocations"])
        for i, ticker in enumerate(selected_tickers):
            results_df[f"{ticker}_allocation"] = allocations[:, i]

        # Add rewards (shift to align with dates)
        results_df["reward"] = [0] + test_results[
            "rewards"
        ]  # No reward for initial state

        # Save to CSV
        csv_save_path = os.path.join(RESULTS_DIR, f"test_results_{timestamp}.csv")
        results_df.to_csv(csv_save_path, index=False)
        print(f"Detailed results saved to {csv_save_path}")

        print("\nTesting completed successfully!")

        # After testing is completed
        try:
            # Call the comparison function
            test_results_csv = csv_save_path
            plot_performance_comparison(
                test_results_csv=test_results_csv,
                financial_data=financial_data,
                tickers=selected_tickers,
                start_date="2023-01-03",
                end_date="2024-12-31",
            )

            print("\nProcess completed successfully!")
        except Exception as e:
            print(f"Error creating performance comparison: {e}")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
