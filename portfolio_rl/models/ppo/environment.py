import gym
import numpy as np
import pandas as pd
import random
from gym import spaces

# PORTFOLIO ENVIRONMENT
class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_weights=None, transaction_cost=0.001, 
                 diversification_penalty=0.1, volatility_penalty=0.05):
        super(PortfolioEnv, self).__init__()
        
        # Store the data
        self.data = data.sort_index().copy()
        self.dates = self.data.index.unique()
        
        # Identify feature columns
        self.return_cols = [col for col in data.columns if '_Return' in col]
        self.num_assets = len(self.return_cols)
        
        # Extract sector names from return column names
        self.sector_names = [col.split('_')[0] for col in self.return_cols]
        
        # Get feature columns by type
        self.fundamental_cols = [col for col in data.columns if 'fund_' in col]
        self.sentiment_cols = [col for col in data.columns if 'sent_' in col]
        self.economic_cols = [col for col in data.columns if 'econ_' in col]
        
        # Define feature space
        self.all_feature_cols = self.return_cols + self.fundamental_cols + self.sentiment_cols + self.economic_cols
        
        # Environment parameters
        self.current_step = 0
        self.transaction_cost = transaction_cost
        self.diversification_penalty = diversification_penalty
        self.volatility_penalty = volatility_penalty
        
        # Portfolio tracking
        if initial_weights is None:
            self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        else:
            self.portfolio_weights = np.array(initial_weights)
            
        # History tracking
        self.returns_history = []
        self.weights_history = [self.portfolio_weights.copy()]
        self.portfolio_value_history = [1.0]  # Start with 1.0 (100%)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Observation space includes returns, fundamentals, sentiment, and economic indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(len(self.all_feature_cols),),
            dtype=np.float32
        )
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng()

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def _calculate_fundamental_outperformance(self, weights):
        """
        Calculate how well the portfolio is positioned based on fundamental factors
        """
        # Get current date and fundamental data
        current_date = self.dates[self.current_step]
        current_data = self.data[self.data.index == current_date]
        
        # If no fundamental data is available, return neutral score
        if not self.fundamental_cols:
            return 0.0
            
        # For each sector, calculate average fundamental score (e.g., PE ratio, ROE)
        fundamental_scores = []
        
        for i, sector in enumerate(self.sector_names):
            sector_fundamental_cols = [col for col in self.fundamental_cols if sector in col]
            
            if sector_fundamental_cols:
                # Average the standardized fundamental metrics (higher is better)
                sector_score = current_data[sector_fundamental_cols].mean(axis=1).values[0]
                fundamental_scores.append(sector_score)
            else:
                fundamental_scores.append(0.0)
        
        # Calculate weighted score based on portfolio weights
        weighted_score = np.dot(weights, fundamental_scores)
        return weighted_score

    def _calculate_sentiment_alignment(self, weights):
        """
        Calculate how well the portfolio is positioned based on sentiment factors
        """
        # Get current date and sentiment data
        current_date = self.dates[self.current_step]
        current_data = self.data[self.data.index == current_date]
        
        # If no sentiment data is available, return neutral score
        if not self.sentiment_cols:
            return 0.0
            
        # For each sector, get the sentiment score
        sentiment_scores = []
        
        for i, sector in enumerate(self.sector_names):
            sector_sentiment_cols = [col for col in self.sentiment_cols if sector in col]
            
            if sector_sentiment_cols:
                # Average the sentiment metrics (higher is better)
                sector_score = current_data[sector_sentiment_cols].mean(axis=1).values[0]
                sentiment_scores.append(sector_score)
            else:
                sentiment_scores.append(0.0)
        
        # Calculate weighted score based on portfolio weights
        weighted_score = np.dot(weights, sentiment_scores)
        return weighted_score

    def _calculate_economic_alignment(self, weights):
        """
        Calculate how well the portfolio is positioned based on economic indicators
        """
        # Get current date and economic data
        current_date = self.dates[self.current_step]
        current_data = self.data[self.data.index == current_date]
        
        # If no economic data is available, return neutral score
        if not self.economic_cols:
            return 0.0
            
        # Get current economic indicators
        economic_indicators = current_data[self.economic_cols].values[0]
        
        # Define sector sensitivities to economic factors (simplified)
        # Each row is a sector, each column is an economic factor
        # These should ideally be learned from historical data
        if len(self.economic_cols) > 0:
            sensitivities = np.random.rand(self.num_assets, len(self.economic_cols)) * 2 - 1
            
            # Calculate how well each sector is positioned for current economic conditions
            sector_alignment = sensitivities @ economic_indicators
            
            # Normalize to [0, 1] range
            sector_alignment = (sector_alignment - sector_alignment.min()) / (sector_alignment.max() - sector_alignment.min() + 1e-10)
            
            # Calculate weighted alignment based on portfolio weights
            weighted_alignment = np.dot(weights, sector_alignment)
            return weighted_alignment
        else:
            return 0.0

    def calculate_reward(self, portfolio_return, weights, previous_weights):
        """
        Calculate composite reward based on multiple factors
        """
        # Transaction cost penalty
        transaction_cost_penalty = np.sum(np.abs(weights - previous_weights)) * self.transaction_cost
        
        # Diversification penalty
        equal_weights = np.ones(self.num_assets) / self.num_assets
        diversification_penalty = self.diversification_penalty * np.sum((weights - equal_weights) ** 2)
        
        # Volatility penalty (if we have enough history)
        volatility_penalty = 0
        if len(self.returns_history) > 1:
            returns_std = np.std(self.returns_history[-10:] if len(self.returns_history) >= 10 else self.returns_history)
            volatility_penalty = self.volatility_penalty * returns_std
        
        # Component rewards
        fundamental_component = self._calculate_fundamental_outperformance(weights)
        sentiment_component = self._calculate_sentiment_alignment(weights)
        economic_component = self._calculate_economic_alignment(weights)
        
        # Combine all components into final reward
        # Portfolio return is still the main component
        reward = portfolio_return - transaction_cost_penalty - diversification_penalty - volatility_penalty
        reward += 0.2 * fundamental_component + 0.2 * sentiment_component + 0.1 * economic_component
        
        reward_components = {
            'portfolio_return': portfolio_return,
            'transaction_cost': -transaction_cost_penalty,
            'diversification': -diversification_penalty,
            'volatility': -volatility_penalty,
            'fundamental': fundamental_component,
            'sentiment': sentiment_component,
            'economic': economic_component,
            'total_reward': reward
        }
        
        return reward, reward_components
    
    def step(self, action):
        """
        Execute one time step within the environment
        """
        # Ensure action is valid - normalize weights to sum to 1
        action = np.array(action).reshape(-1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.num_assets) / self.num_assets
            
        # Calculate transaction costs
        previous_weights = self.portfolio_weights.copy()
            
        # Update portfolio weights
        self.portfolio_weights = action
        self.weights_history.append(self.portfolio_weights.copy())
        
        # Get current observation data
        current_date = self.dates[self.current_step]
        current_data = self.data[self.data.index == current_date]
        
        # Extract returns for current period
        current_returns = current_data[self.return_cols].values[0]
        
        # Calculate portfolio return
        portfolio_return = np.dot(self.portfolio_weights, current_returns)
        
        # Handle invalid portfolio return
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            portfolio_return = 0.0
        
        # Update portfolio value
        self.portfolio_value_history.append(self.portfolio_value_history[-1] * (1 + portfolio_return))
        
        # Update returns history
        self.returns_history.append(portfolio_return)
        
        # Calculate reward
        reward, reward_components = self.calculate_reward(
            portfolio_return=portfolio_return,
            weights=self.portfolio_weights,
            previous_weights=previous_weights
        )
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        # Prepare next observation
        if not done:
            next_date = self.dates[self.current_step]
            next_data = self.data[self.data.index == next_date]
            next_obs = next_data[self.all_feature_cols].values[0]
        else:
            next_obs = np.zeros(len(self.all_feature_cols))
        
        # Check for dynamic rebalancing
        # Rebalance if there are two consecutive negative returns
        if len(self.returns_history) >= 2 and self.returns_history[-1] < 0 and self.returns_history[-2] < 0:
            self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        
        info = {
            'date': current_date,
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value_history[-1],
            'weights': self.portfolio_weights.copy(),
            'reward_components': reward_components
        }
        
        return next_obs, reward, done, info
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        self.returns_history = []
        self.weights_history = [self.portfolio_weights.copy()]
        self.portfolio_value_history = [1.0]
        
        # Get initial observation
        initial_date = self.dates[self.current_step]
        initial_data = self.data[self.data.index == initial_date]
        initial_obs = initial_data[self.all_feature_cols].values[0]
        
        # Handle invalid observations
        if np.isnan(initial_obs).any() or np.isinf(initial_obs).any():
            initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return initial_obs