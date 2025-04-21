"""
Reward Functions for portfolio optimization.

This module implements various reward functions that can be used to train
reinforcement learning agents for portfolio optimization tasks.
"""

import numpy as np


class RewardCalculator:
    """Class for calculating rewards in portfolio optimization."""

    @staticmethod
    def simple_return(
        new_portfolio_value: float, 
        old_portfolio_value: float
    ) -> float:
        """
        Calculate reward based on simple portfolio return.

        Args:
            new_portfolio_value: Current portfolio value
            old_portfolio_value: Previous portfolio value

        Returns:
            Reward as the percentage change in portfolio value
        """
        return (new_portfolio_value - old_portfolio_value) / max(old_portfolio_value, 1e-6)

    @staticmethod
    def risk_adjusted_return(
        new_portfolio_value: float,
        old_portfolio_value: float,
        portfolio_volatility: float,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 0.2,
        num_days: int = 63,  # approx. 1 quarter of trading days
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate reward based on risk-adjusted return using log return.

        Args:
            new_portfolio_value: Current portfolio value
            old_portfolio_value: Previous portfolio value
            portfolio_volatility: Volatility of the portfolio
            risk_free_rate: Risk-free rate of return (in decimal form, e.g., 0.02)
            risk_aversion: Risk aversion parameter (0 = full Sharpe ratio, 1 = raw return)

        Returns:
            Risk-adjusted reward
        """
        if old_portfolio_value <= 0 or new_portfolio_value <= 0:
            return -np.inf

        # Calculate log return
        log_return = np.log(new_portfolio_value / old_portfolio_value)
        rf_adjusted = risk_free_rate * (num_days / periods_per_year)
        
        # Adjust for risk (similar to Sharpe ratio concept)
        if portfolio_volatility > 0:
            risk_adjusted = (log_return - rf_adjusted) / portfolio_volatility
        else:
            risk_adjusted = log_return - rf_adjusted
            
        # Apply risk aversion
        return risk_adjusted * (1 - risk_aversion) + log_return * risk_aversion

    @staticmethod
    def sharpe_ratio(
        return_series: list[float],
        risk_free_rate: float = 0.02,
        num_days: int = 63,  # approx. 1 quarter of trading days
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate reward based on Sharpe ratio.

        Args:
            returns: List of historical returns
            risk_free_rate: Risk-free rate of return

        Returns:
            Sharpe ratio as reward
        """
        if len(return_series) < 2:
            return 0.0
            
        returns_array = np.array(return_series)
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns_array - rf_per_period

        std = np.std(excess_returns, ddof=1)
        if std <= 0:
            return 0.0
        
        # If we can't calculate Sharpe ratio, return the mean excess return
        return np.mean(excess_returns)

    @staticmethod
    def combined_reward(
        new_portfolio_value: float,
        old_portfolio_value: float,
        portfolio_values: list[float],
        return_series: list[float],
        portfolio_volatility: float,
        risk_free_rate: float = 0.0,
        risk_weight: float = 0.5,
    ) -> float:
        """
        Calculate a combined reward using multiple metrics.

        Args:
            new_portfolio_value: Current portfolio value
            old_portfolio_value: Previous portfolio value
            portfolio_values: List of historical portfolio values
            return_series: List of historical returns
            portfolio_volatility: Volatility of the portfolio
            risk_free_rate: Risk-free rate of return
            risk_weight: Weight for the risk-adjusted component

        Returns:
            Combined reward
        """
        # Calculate simple return (base component)
        simple_return = RewardCalculator.simple_return(
            new_portfolio_value, old_portfolio_value
        )
        
        # Calculate risk-adjusted component
        risk_adjusted = RewardCalculator.risk_adjusted_return(
            new_portfolio_value, old_portfolio_value, 
            portfolio_volatility, risk_free_rate
        )

        # Combine components with weights
        # The weights should sum to 1.0
        return_weight = 1.0 - risk_weight
        
        combined = (
            return_weight * simple_return +
            risk_weight * risk_adjusted
        )
        
        return combined