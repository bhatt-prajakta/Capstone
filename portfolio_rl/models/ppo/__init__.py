import numpy as np
from data_preprocessing import load_data_sources, preprocess_data
from utils import monte_carlo_portfolio_optimization
from evaluate import evaluate_feature_importance, evaluate_model
from baseline_comparison import compare_model_with_baselines
from visualizations import plot_metrics_comparison, plot_risk_return_profile, plot_sharpe_and_return, plot_sharpe_ratios, plot_total_cumulative_return
from train import train_and_evaluate_with_walk_forward, train_portfolio_model, walk_forward_validation

# Load data sources
stock_returns, fundamental_data, sentiment_data, economic_data = load_data_sources()

# Preprocess and merge data
merged_data = preprocess_data(stock_returns, fundamental_data, sentiment_data, economic_data)

# Step 3: Perform walk-forward training and evaluation
results_df = train_and_evaluate_with_walk_forward(merged_data, total_timesteps=20000, verbose=1)
results_df

# Print and plot overall results
print("\nOverall Results:")
print("=" * 50)
print(results_df.describe())

plot_metrics_comparison(results_df)

#Select best model and create final portfolio allocation
best_split_idx = results_df['Sharpe_Ratio'].idxmax()
best_split = results_df.iloc[best_split_idx]['Split']

print(f"\nBest model from split {best_split} with Sharpe Ratio: {results_df.iloc[best_split_idx]['Sharpe_Ratio']:.4f}")
print(f"Total Return: {results_df.iloc[best_split_idx]['Total_Return']:.4f}")
print(f"Max Drawdown: {results_df.iloc[best_split_idx]['Max_Drawdown']:.4f}")
print(f"Win Rate: {results_df.iloc[best_split_idx]['Win_Rate']:.4f}")

# Get the appropriate test data and model for the best split
splits = walk_forward_validation(merged_data)
best_train_data, best_validation_data, best_test_data = splits[int(best_split) - 1]

# Create a new training for the best model so we have it in memory for analysis
try:
    initial_weights = monte_carlo_portfolio_optimization(best_train_data)
except:
    return_cols = [col for col in best_train_data.columns if '_Return' in col]
    num_assets = len(return_cols)
    initial_weights = np.ones(num_assets) / num_assets
    
# Train best model
best_model, train_env = train_portfolio_model(
    train_data=best_train_data,
    validation_data=best_validation_data,
    initial_weights=initial_weights,
    total_timesteps=20000,
    verbose=1
    )
# Evaluate on test data to get best model's test results
best_test_results = evaluate_model(best_model, best_test_data, initial_weights)

# Compare with baseline portfolios
print("\nComparing with baseline strategies...")
baseline_comparison, improvement_df = compare_model_with_baselines(
    best_test_results, 
    merged_data, 
    best_split_idx, 
    results_df
)

# Analyze feature importance
print("\nAnalyzing contribution of different data sources...")
feature_importance = evaluate_feature_importance(
    best_model,
    merged_data,
    best_split_idx,
    results_df
)

# Sharpe Ratio 
plot_sharpe_ratios(results_df)

# Cumulative reuturn
plot_total_cumulative_return(results_df)

# Sharpe Ratio vs Annualized Return
plot_sharpe_and_return(results_df)

# Risk vs Return
plot_risk_return_profile(results_df)

print("\nPortfolio Optimization complete!")