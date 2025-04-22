# Capstone: Portfolio Optimization using Reinforcement Learning

## Overview
This project implements a data-driven approach to optimize stock portfolio weights using reinforcement learning techniques. By leveraging fundamental financial data, economic indicators, and market sentiment analysis, we train an RL model to dynamically adjust portfolio weights on a quarterly basis. Our goal is to outperform an evenly-weighted portfolio benchmark, with performance primarily measured using the Sharpe ratio.

## Project Objective
Our primary objective is to develop a reinforcement learning framework that can:
- Analyze diverse financial datasets to identify optimal portfolio allocations
- Dynamically adjust weights based on changing market conditions
- Generate higher risk-adjusted returns compared to benchmark strategies
- Provide insights into feature importance for portfolio optimization decisions

## Data Sources

- **Stock Price Data**: Historical price data for various companies
- **Financial Statements**: Quarterly (10Q) Balance sheets, income statements, and cash flow statements
- **News Sentiment**: Analyzed articles from sources like the New York Times (2014-2024)
- **Economic Indicators**: Quarterly economic data from the Federal Reserve Economic Data (FRED)

## Prerequisites (Data Access Statement)

To use this project, you will need:
- An API key from AlphaVantage to access financial data
- A Wharton Research Data Services (WRDS) account for additional financial information
- Python 3.9+ and dependencies listed in requirements.txt

## Repository Structure

```
├── config/                  # Configuration files
├── data/
│   ├── processed/           # Cleaned and transformed data
│   │   ├── economic_indicators_quarterly_2014_2024.csv
│   │   ├── sector_sentiment_finbert_quarterly_2014_2024.csv
│   │   ├── financial_statements/
│   │   └── ratios/
│   ├── raw/                 # Original unprocessed data
│   │   ├── news_data/       # News articles by quarter
│   │   ├── balance_sheets/
│   │   ├── cashflow_statements/
│   │   ├── income_statements/
│   │   └── stock_prices/
│   └── stock_prices/        # Stock price data
├── notebooks/
│   ├── exploratory/         # Data exploration notebooks
│   └── reports/             # Result analysis and visualizations
├── portfolio_rl/            # Core implementation
│   ├── extractors/          # Data extraction modules
│   ├── processors/          # Data processing and feature engineering
│   ├── models/              # RL model implementations
│   │   ├── environment.py   # Portfolio environment
│   │   ├── agent.py         # RL agent
│   │   ├── reward.py        # Reward functions
│   │   ├── env_test_script.py   # Test script for Portfolio environment
│   │   ├── run_pipeline.py  # Run both train and test scripts
│   │   ├── test_dqn.py      # Test DQN model
│   │   └── train_dqn.py     # Train DQN model
│   └── visualizations/      # Visualization components
├── scripts/                 # Utility scripts for news data
└── utils/                   # API key and other utilities
```

## Key Features

- **Sector-based Stock Selection**: Balanced portfolio across market sectors
- **Sentiment Analysis**: FinBERT-powered analysis of financial news for market sentiment
- **Economic Indicator Integration**: Incorporation of macroeconomic factors in decision-making
- **Financial Ratio Analysis**: Fundamental analysis based on company financials
- **Reinforcement Learning Optimization**: Dynamic portfolio weight adjustments using RL
- **Comprehensive Visualization**: Performance and allocation visualization tools

## Technologies Used

### Data Processing & Analysis
- **Python**: Primary programming language
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Feature engineering and preprocessing
- **FinBERT**: Financial sentiment analysis
- **NLTK**: Natural language processing for news data

### Machine Learning & Reinforcement Learning
- **PyTorch**: Deep learning framework
- **Gym**: Environment creation for RL training

### Visualization & Reporting
- **Matplotlib**: Statistical visualization and charts

## Setup and Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Create a `config.yaml` file with your AlphaVantage API key and WRDS credentials
4. Run data extraction scripts to populate the data directory

## Contributors
- Stephen John: Data Collection, Processing, Financial Analysis & Evaluation, and DQN Model
- Prajakta Bhatt: Market and News Sentiment Analysis, Feature Engineering, PPO model, and Visualization
- James Isioma: Rainbow DQN Reinforcement Learning Model Development and Implementation