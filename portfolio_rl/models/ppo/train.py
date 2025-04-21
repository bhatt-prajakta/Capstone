import numpy as np
import pandas as pd
from environment import PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
from utils import monte_carlo_portfolio_optimization, seed
from evaluate import evaluate_model
from visualizations import plot_portfolio_performance

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# WALK-FORWARD VALIDATION
def walk_forward_validation(data, initial_train_days=252, validation_days=30, test_days=30):
    """
    Implement walk-forward validation for time series data.
    
    Parameters:
        data (pd.DataFrame): DataFrame with 'Date' column
        initial_train_days (int): Number of days for initial training period
        validation_days (int): Number of days for validation
        test_days (int): Number of days for testing
        
    Returns:
        list: List of tuples (train_data, validation_data, test_data)
    """
    data = data.sort_index()
    dates = data.index.unique()
    
    if len(dates) < initial_train_days + validation_days + test_days:
        raise ValueError("Not enough data for the specified periods")
        
    splits = []
    
    # Start with initial training period
    current_idx = initial_train_days
    
    # Continue until we can't form a complete test period
    while current_idx + validation_days + test_days <= len(dates):
        # Define date ranges
        train_end_date = dates[current_idx - 1]
        validation_end_date = dates[current_idx + validation_days - 1]
        test_end_date = dates[current_idx + validation_days + test_days - 1]
        
        # Split data
        train_data = data[data.index <= train_end_date]
        validation_data = data[(data.index > train_end_date) & (data.index <= validation_end_date)]
        test_data = data[(data.index > validation_end_date) & (data.index <= test_end_date)]
        
        if not validation_data.empty and not test_data.empty:
            assert all(train_data.index < validation_data.index.min()), "Training data overlaps with validation data!"
            assert all(validation_data.index < test_data.index.min()), "Validation data overlaps with test data!"
        splits.append((train_data, validation_data, test_data))
        
        # Move forward by test period
        current_idx += test_days
        
    print(f"Created {len(splits)} walk-forward validation splits")
    return splits

# TRAINING AND EVALUATION FUNCTIONS
def train_portfolio_model(train_data, validation_data=None, 
                          initial_weights=None, hyperparams=None,
                          total_timesteps=20000, verbose=0):
    """
    Train a PPO model on the given training data
    
    Args:
        train_data (pd.DataFrame): Training data
        validation_data (pd.DataFrame): Validation data for early stopping
        initial_weights (np.array): Initial portfolio weights
        hyperparams (dict): Hyperparameters for the PPO model
        total_timesteps (int): Number of training timesteps
        verbose (int): Verbosity level
        
    Returns:
        tuple: (trained model, training environment)
    """
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 0.0001,
            'n_steps': 1024,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'ent_coef': 0.05,
            'clip_range': 0.2
        }
    
    # Create environment
    env = PortfolioEnv(data=train_data, initial_weights=initial_weights,
                       transaction_cost=0.001, diversification_penalty=0.1)
    
    # Set seed for reproducibility
    env.seed(SEED)
    
    # Wrap in DummyVecEnv for Stable Baselines compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Initialize model with hyperparameters
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=verbose,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        ent_coef=hyperparams['ent_coef'],
        clip_range=hyperparams['clip_range'],
        seed=SEED
    )
    
    # Implement callback for early stopping if validation data is provided
    if validation_data is not None:
        pass
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    return model, env

def train_and_evaluate_with_walk_forward(data, total_timesteps=20000, verbose=0):
    """
    Train and evaluate using walk-forward validation
    
    Args:
        data (pd.DataFrame): Full dataset
        total_timesteps (int): Number of training timesteps
        verbose (int): Verbosity level
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for each fold
    """
    # Create walk-forward splits
    splits = walk_forward_validation(data)
    
    # Store results for each split
    all_results = []
    
    for i, (train_data, validation_data, test_data) in enumerate(splits):
        print(f"\nProcessing split {i+1}/{len(splits)}")
        print(f"Train period: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Validation period: {validation_data.index.min()} to {validation_data.index.max()}")
        print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")
        
        # Get initial weights (either from Monte Carlo or equal weights)
        try:
            initial_weights = monte_carlo_portfolio_optimization(train_data)
            print("Using Monte Carlo optimized initial weights")
        except:
            # If Monte Carlo fails, use equal weights
            return_cols = [col for col in train_data.columns if '_Return' in col]
            num_assets = len(return_cols)
            initial_weights = np.ones(num_assets) / num_assets
            print("Using equal initial weights")
            
        # Train model
        model, train_env = train_portfolio_model(
            train_data=train_data,
            validation_data=validation_data,
            initial_weights=initial_weights,
            total_timesteps=total_timesteps,
            verbose=verbose
        )
        
        # Evaluate on test data
        test_results = evaluate_model(model, test_data, initial_weights)
        
        # Store results
        result = {
            'Split': i+1,
            'Train_Start': train_data.index.min(),
            'Train_End': train_data.index.max(),
            'Test_Start': test_data.index.min(),
            'Test_End': test_data.index.max(),
            'Total_Return': test_results['metrics']['total_return'],
            'Annualized_Return': test_results['metrics']['annualized_return'],
            'Sharpe_Ratio': test_results['metrics']['sharpe_ratio'],
            'Max_Drawdown': test_results['metrics']['max_drawdown'],
            'Win_Rate': test_results['metrics']['win_rate'],
            'Calmar_Ratio': test_results['metrics']['calmar_ratio'],
            'Sortino_Ratio': test_results['metrics']['sortino_ratio'],
        }
        
        all_results.append(result)

        # plot the test period results
        plot_portfolio_performance(
            dates=test_results['dates'],
            portfolio_values=test_results['cumulative_values'],
            weights_history=test_results['weights_history'],
            return_cols=[col.split('_')[0] for col in test_data.columns if '_Return' in col],
            title=f"Portfolio Performance - Split {i+1}"
        )
    
    # Combine all results into a dataframe
    results_df = pd.DataFrame(all_results)
    
    return results_df