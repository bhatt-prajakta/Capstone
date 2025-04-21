# BASELINE COMPARISON FUNCTIONS

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from evaluate import evaluate_portfolio


def create_baseline_portfolios(test_data, start_date, end_date):
    """
    Create and evaluate baseline portfolio strategies for comparison.
    
    Args:
        test_data: DataFrame with test period data
        start_date: Start date for evaluation
        end_date: End date for evaluation
        
    Returns:
        DataFrame with performance metrics for each baseline
    """
    # Filter test data to evaluation period
    eval_data = test_data[(test_data.index >= start_date) & (test_data.index <= end_date)].copy()
    
    # Get return columns
    return_cols = [col for col in eval_data.columns if '_Return' in col]
    
    # Create baselines
    baselines = {
        'Even_Weighted': {col: 1.0/len(return_cols) for col in return_cols}
    }
    
    # Calculate minimum variance portfolio
    returns_matrix = eval_data[return_cols].dropna()
    cov_matrix = returns_matrix.cov()
    
    def portfolio_variance(weights):
        weights = np.array(weights)
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def sum_to_one(weights):
        return np.sum(weights) - 1.0
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': sum_to_one})
    bounds = tuple((0.0, 1.0) for _ in range(len(return_cols)))
    
    # Initial guess (equal weights)
    initial_weights = np.ones(len(return_cols)) / len(return_cols)
    
    # Optimize
    from scipy.optimize import minimize
    result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                    bounds=bounds, constraints=constraints)
    
    # Store minimum variance weights
    min_var_weights = result['x']
    baselines['Minimum_Variance'] = {col: weight for col, weight in zip(return_cols, min_var_weights)}
    
    # Calculate performance for each baseline
    baseline_results = {}
    
    for name, weights in baselines.items():
        # Calculate daily portfolio returns
        daily_returns = np.zeros(len(eval_data))
        
        for col, weight in weights.items():
            daily_returns += eval_data[col].values * weight
        
        # Calculate performance metrics
        metrics = evaluate_portfolio(daily_returns)
        baseline_results[name] = metrics
    
    # Convert to DataFrame
    baseline_df = pd.DataFrame.from_dict(baseline_results, orient='index')
    
    return baseline_df, eval_data

def compare_model_with_baselines(model_results, test_data, best_split_idx, results_df):
    """
    Compare RL model performance with baseline strategies.
    
    Args:
        model_results: DataFrame with RL model performance
        test_data: Test period data
        best_split_idx: Index of the best split
        results_df: DataFrame with results from all splits
    """
    # Get start and end dates for best split
    start_date = results_df.iloc[best_split_idx]['Test_Start']
    end_date = results_df.iloc[best_split_idx]['Test_End']
    
    # Get baseline results
    baseline_results, eval_data_with_baselines = create_baseline_portfolios(
        test_data, start_date, end_date)
    
    # Add RL model results
    rl_model_metrics = {
        'total_return': results_df.iloc[best_split_idx]['Total_Return'],
        'annualized_return': results_df.iloc[best_split_idx]['Annualized_Return'],
        'sharpe_ratio': results_df.iloc[best_split_idx]['Sharpe_Ratio'],
        'max_drawdown': results_df.iloc[best_split_idx]['Max_Drawdown'],
        'win_rate': results_df.iloc[best_split_idx]['Win_Rate']
    }
    
    comparison = pd.concat([pd.DataFrame(rl_model_metrics, index=['RL_PPO']).T, baseline_results.T], axis=1)
    
    # Calculate outperformance vs even-weighted portfolio
    even_weighted_sharpe = baseline_results.loc['Even_Weighted', 'sharpe_ratio']
    relative_improvement = {}
    for strategy in comparison.columns:
        relative_improvement[strategy] = {
            'Sharpe_Ratio_Improvement': comparison.loc['sharpe_ratio', strategy] - even_weighted_sharpe,
            'Relative_Improvement(%)': ((comparison.loc['sharpe_ratio', strategy] / even_weighted_sharpe) - 1) * 100 
                if even_weighted_sharpe > 0 else float('inf')
        }
    
    improvement_df = pd.DataFrame.from_dict(relative_improvement, orient='columns')
    
    # Print comparison
    print("\nModel vs Baseline Performance Comparison:")
    print("-" * 50)
    print(comparison)
    print("\nRelative Improvement vs Even-Weighted Portfolio:")
    print("-" * 50)
    print(improvement_df)
    
    # Visualize comparison
    plt.figure(figsize=(14, 10))
    
    # Plot metrics comparison
    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
    x = np.arange(len(metrics))
    width = 0.2
    
    strategies = comparison.columns
    for i, strategy in enumerate(strategies):
        values = [comparison.loc[metric, strategy] for metric in metrics]
        plt.bar(x + i*width, values, width=width, label=strategy)
    
    plt.title('Performance Metrics Comparison', fontsize=14)
    plt.xticks(x + width, metrics, fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return comparison, improvement_df