# Portfolio Optimization with Deep Q-Network (DQN)

This directory contains a complete implementation of a Deep Q-Network (DQN) reinforcement learning agent for portfolio optimization. The agent learns to allocate assets in a portfolio based on financial ratios, news sentiment, and economic indicators.

## Overview

The implementation consists of the following components:

1. **Environment** (`environment.py`): A custom OpenAI Gym environment that simulates a portfolio optimization problem.
2. **Agent** (`agent.py`): A DQN agent that learns to make investment decisions.
3. **Reward** (`reward.py`): Functions for calculating rewards based on portfolio performance.
4. **Training** (`train_dqn.py`): Script for training the DQN agent on historical data.
5. **Testing** (`test_dqn.py`): Script for evaluating the trained agent on test data.
6. **Environment Testing** (`env_test_script.py`): Script for testing the environment with random actions.

## Prerequisites

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Gym (OpenAI Gym)

## Data Structure

The implementation expects the following data structure:

```
data/
├── processed/
│   ├── ratios/
│   │   ├── AAPL_ratios.csv
│   │   ├── JPM_ratios.csv
│   │   └── ...
│   ├── economic_indicators_quarterly_2014_2024.csv
│   └── sector_sentiment_with_metrics_quarterly_2014_2024.csv
└── stock_prices/
    ├── AAPL_stock_prices.csv
    ├── JPM_stock_prices.csv
    └── ...
```

## Training the Model

To train a DQN agent on historical data, run:

```bash
python train_dqn.py --episodes 100 --seed 42 --save_path trained_models/dqn_agent.pth
```

Arguments:
- `--episodes`: Number of episodes to train for (default: 100)
- `--seed`: Random seed for reproducibility (default: None)
- `--save_path`: Path to save the trained model (default: 'trained_models/dqn_agent.pth')

The training script will:
1. Load financial, sentiment, and economic data
2. Create a portfolio environment
3. Train a DQN agent to make investment decisions
4. Save the trained model to the specified path
5. Generate a plot of training metrics (rewards, portfolio values, exploration rate)

## Testing the Model

To evaluate a trained DQN agent on test data, run:

```bash
python test_dqn.py --model_path trained_models/dqn_agent.pth --seed 42 --steps 252
```

Arguments:
- `--model_path`: Path to the trained model (default: 'trained_models/dqn_agent.pth')
- `--seed`: Random seed for reproducibility (default: None)
- `--steps`: Maximum number of steps to run (default: 252, approximately one trading year)

The testing script will:
1. Load the trained model
2. Load test data (from 2023 onwards by default)
3. Evaluate the agent's performance on the test data
4. Calculate performance metrics (total return, annualized return, Sharpe ratio, max drawdown)
5. Compare the agent's performance to a benchmark
6. Generate plots of test results (portfolio value, rewards, asset allocations)
7. Save detailed results to CSV

## Complete Pipeline

### Option 1: Single Command

The easiest way to run the complete pipeline is to use the provided `run_pipeline.py` script:

```bash
python run_pipeline.py --train_episodes 100 --test_steps 252 --seed 42
```

This script will:
1. Train the agent with the specified number of episodes
2. Test the trained agent for the specified number of steps
3. Save all results and provide a summary

### Option 2: Step by Step

Alternatively, you can run each step manually:

1. Ensure your data is in the correct structure (see Data Structure section)
2. Train the agent:
   ```bash
   python train_dqn.py --episodes 100
   ```
3. Test the agent:
   ```bash
   python test_dqn.py
   ```
4. Review the results in the `results` directory

## Customization

You can customize the agent's behavior by modifying the following:

- **Environment parameters** in `train_dqn.py` and `test_dqn.py`:
  - `window_size`: Number of past time steps to include in state
  - `max_steps`: Maximum number of steps per episode
  - `tickers`: List of stock tickers to consider

- **Agent parameters** in `train_dqn.py`:
  - `learning_rate`: Learning rate for the optimizer
  - `gamma`: Discount factor
  - `epsilon_start`, `epsilon_end`, `epsilon_decay`: Exploration parameters
  - `buffer_size`: Size of the replay buffer
  - `batch_size`: Batch size for training
  - `target_update_freq`: Frequency of target network updates

## Example Results

After training and testing, you'll get:

1. A trained model saved in the `trained_models` directory
2. Training metrics plot showing rewards, portfolio values, and exploration rate
3. Test results plot showing portfolio value, rewards, and asset allocations over time
4. Performance metrics including total return, annualized return, Sharpe ratio, and maximum drawdown
5. Comparison to a benchmark (equal-weight portfolio by default)
6. Detailed results saved to CSV for further analysis

## Troubleshooting

If you encounter issues:

1. Check that all required data files exist in the expected locations
2. Ensure you have the correct Python dependencies installed
3. Check the error messages for specific issues
4. Try running with a different random seed
5. Reduce the number of episodes or steps if you're running into memory issues
