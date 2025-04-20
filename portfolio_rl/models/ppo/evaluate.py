from matplotlib import pyplot as plt
import numpy as np
from environment import PortfolioEnv
from utils import seed
import pandas as pd
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def evaluate_model(model, test_data, initial_weights=None):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained PPO model
        test_data (pd.DataFrame): Test data
        initial_weights (np.array): Initial portfolio weights
        
    Returns:
        dict: Dictionary with evaluation results
    """
    # Create test environment
    test_env = PortfolioEnv(data=test_data, initial_weights=initial_weights,
                           transaction_cost=0.001, diversification_penalty=0.1)
    test_env.seed(SEED)
    
    # Run evaluation
    obs = test_env.reset()
    done = False
    rewards = []
    portfolio_returns = []
    portfolio_values = []
    weights_history = []
    dates = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        rewards.append(reward)
        portfolio_returns.append(info['portfolio_return'])
        portfolio_values.append(info['portfolio_value'])
        weights_history.append(info['weights'])
        dates.append(test_data.index[len(dates)])
    
    # Calculate performance metrics
    metrics = evaluate_portfolio(portfolio_returns)
    
    # Prepare results
    results = {
        'dates': dates,
        'returns': portfolio_returns,
        'cumulative_values': portfolio_values,
        'weights_history': weights_history,
        'metrics': metrics
    }
    
    return results

# PORTFOLIO EVALUATION METRICS
def evaluate_portfolio(returns, risk_free_rate=0.02/252):
    """
    Calculate comprehensive portfolio performance metrics
    
    Args:
        returns (np.array): Daily returns of the portfolio
        risk_free_rate (float): Daily risk-free rate
        
    Returns:
        dict: Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0
        }
    
    # Calculate cumulative return (compounded)
    cumulative_return = np.prod(1 + np.array(returns)) - 1
    
    # Annualized return (assuming 252 trading days per year)
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    
    # Calculate volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)
    
    # Calculate Sharpe Ratio (annualized)
    excess_returns = np.array(returns) - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    # Calculate maximum drawdown
    wealth_index = (1 + np.array(returns)).cumprod()
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = np.min(drawdowns)
    
    # Calculate win rate
    win_rate = np.mean(np.array(returns) > 0)
    
    # Calculate Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate Sortino Ratio (downside risk)
    downside_returns = np.array([min(ret - risk_free_rate, 0) for ret in returns])
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
    
    # Return all metrics as a dictionary
    return {
        'total_return': cumulative_return,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio
    }

# DATA CONTRIBUTION ANALYSIS
def evaluate_feature_importance(model, test_data, best_split_idx, results_df):
    """
    Evaluate the importance of different data sources in the model's decision-making.
    
    Args:
        model: Trained RL model
        test_data: Test dataset
        best_split_idx: Index of the best split
        results_df: DataFrame with results from all splits
    
    Returns:
        DataFrame with feature importance scores
    """
    # Get start and end dates for best split
    start_date = results_df.iloc[best_split_idx]['Test_Start']
    end_date = results_df.iloc[best_split_idx]['Test_End']
    
    # Filter test data to best split period
    test_period_data = test_data[(test_data.index >= start_date) & 
                                (test_data.index <= end_date)].copy()
    
    # Extract feature names by data source
    fundamental_features = [col for col in test_period_data.columns if 'fund_' in col]
    sentiment_features = [col for col in test_period_data.columns if 'sent_' in col]
    economic_features = [col for col in test_period_data.columns if 'econ_' in col]
    
    feature_groups = {
        'Fundamental': fundamental_features,
        'Sentiment': sentiment_features,
        'Economic': economic_features
    }
    
    # Create baseline and modified portfolios
    baseline_sharpe = results_df.iloc[best_split_idx]['Sharpe_Ratio']
    
    # Function to zero out features and compute performance impact
    results = {}
    
    for group_name, features in feature_groups.items():
        # Skip if no features in this group
        if not features:
            results[group_name] = {'Impact': 0, 'Relative_Importance': 0}
            continue
            
        # Create a modified environment with zeroed features
        modified_data = test_period_data.copy()
        
        # Zero out the features in this group
        for feature in features:
            modified_data[feature] = 0
            
        # Create a PortfolioEnv with the modified data
        mod_env = PortfolioEnv(data=modified_data)
        mod_env.seed(SEED)
        
        # Evaluate with the existing model
        obs = mod_env.reset()
        done = False
        portfolio_returns = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = mod_env.step(action)
            portfolio_returns.append(info['portfolio_return'])
        
        # Calculate Sharpe ratio
        portfolio_returns = np.array(portfolio_returns)
        modified_metrics = evaluate_portfolio(portfolio_returns)
        modified_sharpe = modified_metrics['sharpe_ratio']
        
        # Calculate impact
        sharpe_impact = baseline_sharpe - modified_sharpe
        relative_importance = (sharpe_impact / baseline_sharpe) * 100 if baseline_sharpe != 0 else 0
        
        results[group_name] = {
            'Impact': sharpe_impact,
            'Relative_Importance': relative_importance
        }
    
    # Convert to DataFrame
    importance_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    
    # Plot relative importance
    plt.bar(importance_df.index, importance_df['Relative_Importance'])
    plt.title('Relative Importance of Data Sources (%)', fontsize=14)
    plt.ylabel('Importance (%)', fontsize=12)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df