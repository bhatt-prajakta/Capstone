# VISUALIZATION FUNCTIONS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")

def plot_portfolio_performance(dates, portfolio_values, weights_history, return_cols, title=None):
    """
    Plot portfolio performance, including cumulative return and weights over time
    
    Args:
        dates (list): List of dates
        portfolio_values (list): List of portfolio values over time
        weights_history (list): List of weight vectors over time
        return_cols (list): List of asset/sector names
        title (str): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot portfolio value
    ax1.plot(dates, portfolio_values, 'b-', linewidth=2)
    ax1.set_title(title or "Portfolio Performance")
    ax1.set_ylabel("Portfolio Value")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    
    # Format dates on x-axis
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot weights over time
    weights_array = np.array(weights_history)
    
    # Define a colormap
    colors = plt.cm.tab10.colors
    # If we need more colors than available in tab10
    if len(return_cols) > len(colors):
        # Use more colors from another colormap or cycle through
        additional_colors = plt.cm.tab20.colors
        colors = colors + additional_colors
    
    bottom = np.zeros(len(dates))
    
    for i, asset in enumerate(return_cols):
        color_idx = i % len(colors)  # Use modulo to cycle through colors if needed
        asset_weights = weights_array[:, i]
        ax2.bar(dates, asset_weights, bottom=bottom, width=1, label=asset, color=colors[color_idx])
        bottom += asset_weights
    
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Weights")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)
    ax2.set_ylim(0, 1)
    
    # Format dates on x-axis
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(strategies_returns, benchmark_returns, strategy_names, title=None):
    """
    Plot performance comparison between multiple strategies and a benchmark
    
    Args:
        strategies_returns (list): List of return arrays for each strategy
        benchmark_returns (np.array): Returns for the benchmark
        strategy_names (list): Names of the strategies
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative returns
    for i, returns in enumerate(strategies_returns):
        cum_returns = (1 + np.array(returns)).cumprod()
        plt.plot(cum_returns, linewidth=2, label=strategy_names[i])
    
    # Add benchmark
    benchmark_cum_returns = (1 + np.array(benchmark_returns)).cumprod()
    plt.plot(benchmark_cum_returns, 'k--', linewidth=2, label='Benchmark')
    
    plt.title(title or "Performance Comparison")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(metrics_df):
    """
    Plot comparison of key metrics across all validation splits
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics for each split
    """
    # Select key metrics to plot
    key_metrics = ['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        sns.barplot(x='Split', y=metric, data=metrics_df, ax=axes[i])
        axes[i].set_title(f"{metric} by Validation Split")
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("Split")
        
        # Add mean line
        mean_val = metrics_df[metric].mean()
        axes[i].axhline(mean_val, color='r', linestyle='--', alpha=0.7)
        axes[i].text(0, mean_val, f' Mean: {mean_val:.4f}', color='r')
    
    plt.tight_layout()
    plt.show()

def plot_reward_components(reward_components_history):
    """
    Plot the components of the reward function over time
    
    Args:
        reward_components_history (list): List of dictionaries with reward components
    """
    # Extract components
    components = list(reward_components_history[0].keys())
    dates = list(range(len(reward_components_history)))
    
    component_data = {comp: [d[comp] for d in reward_components_history] for comp in components}
    
    plt.figure(figsize=(12, 6))
    
    for comp in components:
        if comp != 'total_reward':  # Plot all except total_reward
            plt.plot(dates, component_data[comp], label=comp)
    
    plt.plot(dates, component_data['total_reward'], 'k-', linewidth=3, label='Total Reward')
    
    plt.title("Reward Components Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sharpe_ratios(results_df: pd.DataFrame, benchmark: float = 1.0) -> None:
    """
    Plots the annualized Sharpe Ratio across test splits.
    
    Parameters:
    - results_df (pd.DataFrame): A DataFrame containing at least 'Split' and 'Sharpe_Ratio' columns.
    - benchmark (float): A reference Sharpe Ratio line to plot for comparison. Default is 1.0.
    """
    # Ensure Sharpe Ratio column is float
    results_df['Sharpe_Ratio'] = results_df['Sharpe_Ratio'].astype(float)

    # Sort by split number for visual continuity
    results_df_sorted = results_df.sort_values(by='Split')

    # Plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=results_df_sorted, x='Split', y='Sharpe_Ratio', marker='o', linewidth=1)

    # Titles and labels
    plt.title("Sharpe Ratio Across Test Splits", fontsize=16)
    plt.xlabel("Split Index", fontsize=12)
    plt.ylabel("Annualized Sharpe Ratio", fontsize=12)
    
    # Reference lines
    plt.axhline(y=benchmark, color='red', linestyle='--', label=f'Sharpe = {benchmark:.1f} (Benchmark)')
    plt.axhline(y=0, color='gray', linestyle='--', label='Break-even')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram of Sharpe Ratio distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(results_df['Sharpe_Ratio'], bins=30, kde=True)
    plt.title("Distribution of Sharpe Ratios Across Test Splits")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Cumulative return plot
def plot_total_cumulative_return(results_df: pd.DataFrame,
                                 return_col: str = 'Total_Return',
                                 date_col: str = 'Test_End') -> None:
    """
    Plots cumulative returns over time using total return.

    Parameters:
    - results_df (pd.DataFrame): DataFrame with total return data.
    - return_col (str): Column name for total returns. Default is 'Total_Return'.
    - date_col (str): Column name for test end dates. Default is 'Test_End'.
    """

    # Sort by date for proper cumulative calculation
    results_df = results_df.sort_values(by=date_col).copy()

    # Compute cumulative return
    results_df['Cumulative_Return'] = (1 + results_df[return_col]).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df[date_col], results_df['Cumulative_Return'], marker='o')
    plt.title('Cumulative Returns Over Time', fontsize=16)
    plt.xlabel('Test End Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Sharpe and return
def plot_sharpe_and_return(results_df: pd.DataFrame,
                           split_col: str = 'Split',
                           sharpe_col: str = 'Sharpe_Ratio',
                           return_col: str = 'Annualized_Return') -> None:
    """
    Plots Sharpe Ratio and Annualized Return across data splits.

    Parameters:
    - results_df (pd.DataFrame): DataFrame with split evaluation metrics.
    - split_col (str): Column representing data splits. Default is 'Split'.
    - sharpe_col (str): Column for Sharpe Ratio values. Default is 'Sharpe_Ratio'.
    - return_col (str): Column for Annualized Return values. Default is 'Annualized_Return'.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(results_df[split_col], results_df[sharpe_col],
             label='Sharpe Ratio', color='green', marker='o')
    plt.plot(results_df[split_col], results_df[return_col],
             label='Annualized Return', color='blue', marker='s')
    plt.axhline(y=0, color='black', linestyle='--')
    
    plt.title('Sharpe Ratio and Annualized Return Across Splits', fontsize=16)
    plt.xlabel(split_col, fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Risk and return plot
def plot_risk_return_profile(results_df: pd.DataFrame,
                              return_col: str = 'Annualized_Return',
                              drawdown_col: str = 'Max_Drawdown',
                              sharpe_col: str = 'Sharpe_Ratio') -> None:
    """
    Plots a risk-return profile scatter plot with color indicating Sharpe Ratio.
    
    Parameters:
    - results_df (pd.DataFrame): DataFrame containing return, drawdown, and Sharpe ratio data.
    - return_col (str): Column name for annualized returns.
    - drawdown_col (str): Column name for max drawdowns.
    - sharpe_col (str): Column name for Sharpe ratios (used for coloring points).
    """

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(results_df[return_col],
                          results_df[drawdown_col],
                          c=results_df[sharpe_col],
                          cmap='viridis',
                          edgecolor='k')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Annualized Return')
    plt.ylabel('Max Drawdown')
    plt.title('Risk-Return Profile of Each Test Period')
    plt.grid(True)
    plt.tight_layout()
    plt.show()