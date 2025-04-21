import numpy as np
import pandas as pd
import random

def seed(self, seed=None):
    """Set random seed for reproducibility"""
    self.rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    return [seed]

# PORTFOLIO OPTIMIZATION FUNCTIONS
def monte_carlo_portfolio_optimization(returns_data, num_simulations=10000):
    """
    Use Monte Carlo simulations to find optimal initial weights.
    
    Parameters:
        returns_data (pd.DataFrame): DataFrame with sector returns
        num_simulations (int): Number of portfolio simulations
        
    Returns:
        pd.Series: Optimal initial weights based on Sharpe ratio
    """
    print(f"Running Monte Carlo simulation with {num_simulations} iterations...")
    
    # Extract return columns
    return_cols = [col for col in returns_data.columns if '_Return' in col]
    returns = returns_data[return_cols].dropna()
    
    n_assets = len(return_cols)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Run simulations
    results = np.zeros((4, num_simulations))
    weights_record = np.zeros((num_simulations, n_assets))
    
    for i in range(num_simulations):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        weights_record[i] = weights
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        results[2, i] = (portfolio_return - 0.02) / portfolio_std_dev
    
    # Find best Sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    best_weights = weights_record[max_sharpe_idx]
    
    # Create a Series with asset names
    best_weights_series = pd.Series(best_weights, index=return_cols)
    
    print(f"Monte Carlo simulation complete. Optimal Sharpe ratio: {results[2, max_sharpe_idx]:.4f}")
    return best_weights_series

def calculate_volatility_adjusted_weights(data):
    """
    Calculate portfolio weights inversely proportional to sector volatility.
    """
    return_cols = [col for col in data.columns if '_Return' in col]
    sector_volatility = data[return_cols].std()
    inverse_volatility = 1 / sector_volatility
    weights = inverse_volatility / inverse_volatility.sum()
    return weights

def enforce_diversification(weights, min_weight=0.05, max_weight=0.3):
    """
    Adjust portfolio weights to ensure diversification by capping and flooring weights.
    """
    weights = np.array(weights)
    weights = np.clip(weights, min_weight, max_weight)
    return weights / weights.sum()  # Normalize to ensure weights sum to 1

def calculate_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown of a portfolio.
    """
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

def calculate_dynamic_risk_free_rate(economic_data, current_date):
    """
    Calculate a dynamic risk-free rate based on economic indicators.
    Simplification: Use treasury yield if available, otherwise fixed rate.
    """
    if 'econ_TreasuryYield_10Y' in economic_data.columns:
        treasury_data = economic_data[['econ_TreasuryYield_10Y']]
        current_yield = treasury_data[treasury_data.index <= current_date].iloc[-1]['econ_TreasuryYield_10Y']
        # Convert annual yield to daily rate
        daily_rf = (1 + current_yield) ** (1/252) - 1
        return daily_rf
    else:
        # Default to 2% annual rate
        return (1.02 ** (1/252)) - 1