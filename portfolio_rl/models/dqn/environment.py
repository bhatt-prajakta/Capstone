"""
Portfolio Environment for Reinforcement Learning.

This module implements a portfolio environment that uses financial ratios,
news sentiment, and economic indicators for portfolio optimization.
"""

import os
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import gym
from gym import spaces
from portfolio_rl.models.dqn.reward import RewardCalculator

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STOCK_PRICE_DIR = os.path.join(DATA_DIR, "stock_prices")

# Column name mappings for standardization
PRICE_COLUMN_MAP = {
    "timestamp": "date",
    "adjusted_close": "price",
    "close": "raw_close",
    "open": "raw_open",
    "high": "raw_high",
    "low": "raw_low",
    "volume": "volume",
    "dividend_amount": "dividend",
    "split_coefficient": "split",
}


class PortfolioEnv(gym.Env):
    """Environment for portfolio optimization using financial data."""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            financial_data: pd.DataFrame,
            sentiment_data: pd.DataFrame,
            economic_data: pd.DataFrame,
            tickers: list[str],
            initial_balance: float = 10000.0,
            transaction_cost: float = 0.01,
            window_size: int = 10,
            max_steps: int = 252,  # Approximately one trading year
            use_old_step_api: bool = False,  # For backward compatibility
    ):
        """
        Initialize the portfolio environment.

        Args:
            financial_data: DataFrame containing financial ratios
            sentiment_data: DataFrame containing news sentiment metrics
            economic_data: DataFrame containing economic indicators
            tickers: List of stock tickers to consider
            initial_balance: Initial portfolio balance
            transaction_cost: Cost per transaction as a fraction of trade value
            window_size: Number of past time steps to include in state
            max_steps: Maximum number of steps per episode
            use_old_step_api: If True, step() returns (obs, reward, done, info) instead of
                             (obs, reward, terminated, truncated, info)
        """
        super(PortfolioEnv, self).__init__()
        # Define ticker sector mapping
        self.ticker_sector_map = {
            "AAPL": "Technology",
            "AMD": "AI/Chip Manufacturing",
            "AMZN": "Technology",
            "AVGO": "AI/Chip Manufacturing",
            "AX": "Finance/Banking",
            "BAC": "Finance/Banking",
            "BAX": "Healthcare",
            "BLK": "Finance/Banking",
            "F": "Automotive",
            "GM": "Automotive",
            "GOOGL": "Technology",
            "GS": "Finance/Banking",
            "IBM": "Technology",
            "INTC": "AI/Chip Manufacturing",
            "JNJ": "Healthcare",
            "JPM": "Finance/Banking",
            "META": "Technology",
            "MRK": "Healthcare",
            "MRNA": "Healthcare",
            "MS": "Finance/Banking",
            "MSFT": "Technology",
            "NFLX": "Technology",
            "NVDA": "AI/Chip Manufacturing",
            "PFE": "Healthcare",
            "STLA": "Automotive",
            "TM": "Automotive",
            "TSLA": "Automotive",
            "TSMC": "AI/Chip Manufacturing",
            "WFC": "Finance/Banking",
        }

        # Store basic parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.max_steps = max_steps
        self.tickers = tickers
        self.n_assets = len(tickers)

        # Define financial ratio columns
        self.financial_ratio_columns = [
            "inventory_turnover",
            "days_of_inventory_onhand",
            "receivables_turnover",
            "days_of_sales_outstanding",
            "payables_turnover",
            "working_capital_turnover",
            "fixed_asset_turnover",
            "total_asset_turnover",
            "cash_conversion_cycle",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "defensive_internal_ratio",
            "debt_to_capital_ratio",
            "debt_to_equity_ratio",
            "financial_leverage_ratio",
            "debt_to_ebitda_ratio",
            "gross_profit_margin",
            "operating_profit_margin",
            "net_profit_margin",
            "operating_return_on_assets",
            "return_on_assets",
            "return_on_equity",
        ]

        # Process price data
        self.price_data = self._load_and_prepare_price_data()

        # Process financial data
        self.financial_data = self._prepare_financial_data(financial_data)

        # Process sentiment data
        self.sentiment_data = self._prepare_sentiment_data(sentiment_data)

        # Process economic data
        self.economic_data = self._prepare_economic_data(economic_data)

        # Align data date ranges
        # Get the earliest date from economic and sentiment data
        economic_min_date = self.economic_data["date"].min()
        sentiment_min_date = self.sentiment_data["date"].min()

        # Use the later of the two dates as the start date
        start_date = max(economic_min_date, sentiment_min_date)

        # Get the latest date from economic and sentiment data
        economic_max_date = self.economic_data["date"].max()
        sentiment_max_date = self.sentiment_data["date"].max()

        # Use the earlier of the two dates as the end date
        end_date = min(economic_max_date, sentiment_max_date)

        print(
            f"Financial data date range: {self.price_data['date'].min()} to {self.price_data['date'].max()}"
        )
        print(
            f"Economic data date range: {self.economic_data['date'].min()} to {self.economic_data['date'].max()}"
        )
        print(
            f"Sentiment data date range: {self.sentiment_data['date'].min()} to {self.sentiment_data['date'].max()}"
        )

        # Filter price data to match the date range of economic and sentiment data
        self.price_data = self.price_data[
            (self.price_data["date"] >= start_date) &
            (self.price_data["date"] <= end_date)
        ]

        print(
            f"Aligned data date range: {self.price_data['date'].min()} to {self.price_data['date'].max()}"
        )

        # Organize price data by ticker
        self.ticker_price_data = self._organize_price_by_ticker()

        # Get all unique dates from price data for step progression
        self.dates = sorted(self.price_data["date"].unique())

        # Calculate sector averages
        self.sector_metrics = self._calculate_sector_metrics()

        # Flag for backward compatibility with older gym API
        self._return_old_step_api = use_old_step_api

        # Define action and observation spaces
        # Action space: continuous allocation weights for each asset
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        # Calculate observation space dimension
        # For each ticker: price, volatility, momentum, financial features, sector one-hot, position
        sectors = sorted(set(self.ticker_sector_map.values()))
        ticker_feature_dim = 3 + len(self.financial_ratio_columns) + len(sectors) + 1

        # Sentiment features: 4 metrics per sector
        sentiment_feature_dim = len(sectors) * 4

        # Economic features
        econ_cols = [col for col in self.economic_data.columns if col not in ["date"]]
        economic_feature_dim = max(1, len(econ_cols))

        # Portfolio features: cash balance and portfolio value
        portfolio_feature_dim = 2

        # Total observation dimension
        obs_dim = (
                (ticker_feature_dim * self.n_assets)
                + sentiment_feature_dim
                + economic_feature_dim
                + portfolio_feature_dim
        )

        # Observation space: continuous state vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize environment
        self.reset()

    def _load_and_prepare_price_data(self) -> pd.DataFrame:
        """Load and consolidate price data from individual stock files."""
        all_price_data = []

        for ticker in self.tickers:
            file_path = os.path.join(STOCK_PRICE_DIR, f"{ticker}_stock_prices.csv")

            try:
                # Load the price data
                ticker_data = pd.read_csv(file_path)

                # Rename columns according to our mapping
                ticker_data.columns = ticker_data.columns.str.strip().str.lower()
                ticker_data = ticker_data.rename(columns=PRICE_COLUMN_MAP)

                # Add ticker column
                ticker_data["ticker"] = ticker

                # Convert date to datetime
                ticker_data["date"] = pd.to_datetime(ticker_data["date"])

                # Calculate returns
                ticker_data["returns"] = ticker_data["price"].pct_change()

                # Handle missing values
                ticker_data["returns"] = ticker_data["returns"].fillna(0)

                all_price_data.append(ticker_data)

            except FileNotFoundError:
                print(f"Warning: Price data file not found for {ticker}")
                continue

        if not all_price_data:
            raise ValueError("No price data files found for any ticker")

        # Combine all price data
        price_data = pd.concat(all_price_data, ignore_index=True)
        price_data = price_data.sort_values(["ticker", "date"])

        return price_data

    def _prepare_financial_data(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize financial data and convert types."""
        financial_data = financial_data.copy()

        # Convert date to datetime
        financial_data["date"] = pd.to_datetime(financial_data["date"])

        # Convert ratio columns to numeric
        for col in self.financial_ratio_columns:
            if col in financial_data.columns:
                financial_data[col] = pd.to_numeric(
                    financial_data[col], errors="coerce"
                )

        return financial_data

    def _prepare_sentiment_data(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize sentiment data column names and convert types."""
        sentiment_data = sentiment_data.copy()

        # Standardize column names
        if "Date" in sentiment_data.columns:
            sentiment_data = sentiment_data.rename(columns={"Date": "date"})

        # Convert date to datetime
        sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])

        return sentiment_data

    def _prepare_economic_data(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize economic data column names and convert types."""
        economic_data = economic_data.copy()

        # Standardize column names
        if "Date" in economic_data.columns:
            economic_data = economic_data.rename(columns={"Date": "date"})

        # Convert date to datetime
        economic_data["date"] = pd.to_datetime(economic_data["date"])

        return economic_data

    def _organize_price_by_ticker(self) -> dict:
        """Organize price data by ticker."""
        ticker_price_data = {}
        for ticker in self.tickers:
            ticker_data = self.price_data[
                self.price_data["ticker"] == ticker
                ].sort_values("date")
            ticker_price_data[ticker] = ticker_data
        return ticker_price_data

    def _calculate_sector_metrics(self):
        """
        Calculate sector-level metrics from ticker data.

        Returns:
            Dictionary with sector metrics
        """
        # Get unique sectors
        sectors = set(self.ticker_sector_map.values())

        # Financial metrics by sector
        sector_financial = {}

        for sector in sectors:
            # Get tickers in this sector
            sector_tickers = [
                t for t, s in self.ticker_sector_map.items() if s == sector
            ]

            # Filter financial data for this sector
            sector_financial_data = self.financial_data[
                self.financial_data["ticker"].isin(sector_tickers)
            ]

            # Group by date and calculate mean of metrics
            if not sector_financial_data.empty:
                # Clean numeric columns before averaging
                numeric_cols = sector_financial_data.select_dtypes(
                    include=[np.number]
                ).columns
                sector_avg = sector_financial_data.groupby("date")[numeric_cols].mean()

                # Fill NaN values with sector median or 0
                sector_avg = sector_avg.fillna(sector_avg.median())
                sector_financial[sector] = sector_avg
            else:
                print(f"Warning: No data found for sector {sector}")
                sector_financial[sector] = pd.DataFrame()

        return sector_financial

    def seed(self, seed=None):
        """Set random seed for reproducibility.

        Args:
            seed: Random seed

        Returns:
            List containing the seed
        """
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (initial state observation, info dictionary)
        """
        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.n_assets)  # Holdings of each asset
        self.portfolio_units = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_balance
        self.current_step = self.window_size

        # Ensure current_step doesn't exceed the available dates
        if self.current_step >= len(self.dates):
            self.current_step = len(self.dates) - 1

        self.current_date = self.dates[self.current_step]
        self.done = False

        # Portfolio history for analysis
        self.portfolio_history = [self.portfolio.copy()]
        self.value_history = [self.portfolio_value]
        self.date_history = [self.current_date]
        self.action_history = []

        # Get initial state
        initial_state = self._get_state()

        # Return state and info dictionary as per gym convention
        info = {
            "portfolio_value": self.portfolio_value,
            "date": self.current_date,
            "portfolio_allocation": self.portfolio.copy(),
        }

        # For backward compatibility with older code
        if self._return_old_step_api:
            return initial_state

        # Return in the format expected by newer gym versions
        return initial_state, info

    def _get_state(self):
        # Get the current date
        current_date = self.dates[self.current_step]

        # Ensure the date is in the correct format
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)

        # Get ticker features
        ticker_features = []
        for i, ticker in enumerate(self.tickers):
            # Get price features
            ticker_price = self.ticker_price_data[ticker]

            # Ensure date column is datetime for comparison
            if ticker_price["date"].dtype == "object":
                ticker_price["date"] = pd.to_datetime(ticker_price["date"])

            # Filter prices up to current date
            recent_prices = (
                ticker_price[ticker_price["date"] <= current_date]
                .sort_values("date")
                .tail(self.window_size)
            )

            if not recent_prices.empty:
                # Use DlyRet for returns-based features
                returns = (
                    recent_prices["DlyRet"].values
                    if "DlyRet" in recent_prices.columns
                    else np.zeros(len(recent_prices))
                )
                volatility = np.std(returns) if len(returns) > 1 else 0
                momentum = np.mean(returns) if len(returns) > 0 else 0

                # Use DlyCap as price for current value
                current_price = (
                    recent_prices["DlyCap"].values[-1]
                    if "DlyCap" in recent_prices.columns
                    else 0
                )
            else:
                volatility = 0
                momentum = 0
                current_price = 0

            # Get financial features
            ticker_financial = (
                self.financial_data[
                    (self.financial_data["ticker"] == ticker)
                    & (self.financial_data["date"] <= current_date)
                    ]
                .sort_values("date")
                .tail(1)
            )

            financial_features = []
            if not ticker_financial.empty:
                # Extract available financial ratios
                available_ratios = [
                    col
                    for col in self.financial_ratio_columns
                    if col in ticker_financial.columns
                ]

                if available_ratios:
                    financial_features = ticker_financial[
                        available_ratios
                    ].values.flatten()

                    # Handle NaN values by replacing with 0
                    financial_features = np.nan_to_num(financial_features, nan=0.0)

            # Ensure consistent feature size
            if len(financial_features) < len(self.financial_ratio_columns):
                financial_features = np.pad(
                    financial_features,
                    (0, len(self.financial_ratio_columns) - len(financial_features)),
                    "constant",
                    constant_values=0,
                )

            # Get sector information
            sector = self.ticker_sector_map.get(ticker, "Unknown")
            sectors = sorted(set(self.ticker_sector_map.values()))

            # Create one-hot encoding for sector
            sector_one_hot = np.zeros(len(sectors))
            if sector in sectors:
                sector_one_hot[sectors.index(sector)] = 1

            # Combine price and financial features
            ticker_feature_vector = np.concatenate(
                [
                    [current_price],  # Current price
                    [volatility],  # Price volatility
                    [momentum],  # Price momentum
                    financial_features,
                    sector_one_hot,
                    [self.portfolio[i]],  # Current position in this asset
                ]
            )

            ticker_features.append(ticker_feature_vector)

        # Combine all ticker features
        all_ticker_features = np.concatenate(ticker_features)

        # Get sentiment features
        sentiment_features = []
        recent_sentiment = (
            self.sentiment_data[self.sentiment_data["date"] <= current_date]
            .sort_values("date")
            .tail(self.window_size)
        )

        # Get sentiment columns for relevant sectors
        for sector in sorted(set(self.ticker_sector_map.values())):
            # Handle the sector name mapping
            sentiment_sector_name = sector
            if sector == "AI/Chip Manufacturing":
                sentiment_sector_name = "AI/Chip Manufacturing"
            elif sector == "Artificial Intelligence":
                sentiment_sector_name = "AI/Chip Manufacturing"
            elif (
                    sector == "Technology"
                    and "Technology" not in self.sentiment_data.columns
            ):
                sentiment_sector_name = "Technology"

            # Get all sentiment-related columns for this sector
            sector_sentiment_cols = [
                col
                for col in self.sentiment_data.columns
                if sentiment_sector_name in col
                   and any(
                    term in col.lower()
                    for term in ["sentiment", "momentum", "dispersion", "normalized"]
                )
            ]

            if not recent_sentiment.empty and sector_sentiment_cols:
                # Take the mean of sentiment features across the window
                sector_sentiment_metrics = (
                    recent_sentiment[sector_sentiment_cols].mean().values
                )
                sector_sentiment_metrics = np.nan_to_num(
                    sector_sentiment_metrics, nan=0.0
                )
                sentiment_features.extend(sector_sentiment_metrics)
            else:
                # If no sentiment data available, use zeros
                sentiment_features.extend(np.zeros(4))  # 4 types of sentiment metrics

        # Ensure sentiment_features is a proper numpy array
        sentiment_features = np.array(sentiment_features)

        # Get economic indicators
        economic_features = []
        recent_economic = (
            self.economic_data[self.economic_data["date"] <= current_date]
            .sort_values("date")
            .tail(self.window_size)
        )

        # Identify economic indicator columns
        econ_cols = [col for col in self.economic_data.columns if col not in ["date"]]

        if not recent_economic.empty and econ_cols:
            # Take the mean across time steps for a fixed-size state
            economic_indicators = recent_economic[econ_cols].mean().values
            economic_indicators = np.nan_to_num(economic_indicators, nan=0.0)
            economic_features.extend(economic_indicators)
        else:
            # Use zeros if no economic data available
            economic_features = np.zeros(max(1, len(econ_cols)))

        # Portfolio metrics
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized cash balance
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
        ]

        # Combine all features into state vector
        state = np.concatenate(
            [
                all_ticker_features,
                sentiment_features,
                economic_features,
                portfolio_features,
            ]
        )

        # Handle any NaN values in the state
        state = np.nan_to_num(state, nan=0.0)

        # Ensure state is in the correct format for the observation space
        state = state.astype(np.float32)

        # Verify state shape matches observation space
        if (
                hasattr(self, "observation_space")
                and state.shape != self.observation_space.shape
        ):
            print(
                f"Warning: State shape {state.shape} does not match observation space shape {self.observation_space.shape}"
            )
            # Pad or truncate state to match observation space shape
            if len(state) < self.observation_space.shape[0]:
                state = np.pad(
                    state, (0, self.observation_space.shape[0] - len(state)), "constant"
                )
            else:
                state = state[: self.observation_space.shape[0]]

        return state

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Action to take (portfolio allocation)

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Map action to portfolio allocation
        new_allocation = self._action_to_allocation(action)

        # Store old portfolio value for reward calculation
        old_portfolio_value = self.portfolio_value

        # Set costs
        base_cost_rate = 0.01  # 1% base transaction cost
        market_impact_factor = 5.0  # Market impact factor

        # Calculate one-way turnover
        turnover = abs(new_allocation - self.portfolio).sum() / 2  # Divide by 2 to get one-way turnover

        # Add market impact/slippage
        fixed_costs = base_cost_rate * turnover * self.portfolio_value

        # Market impact cost
        market_impact = market_impact_factor * base_cost_rate * np.sqrt(turnover) * self.portfolio_value

        # Total trading costs
        total_trading_cost = fixed_costs + market_impact

        # Apply transaction costs directly to portfolio value
        self.portfolio_value -= total_trading_cost

        # Update portfolio allocations
        self.portfolio = new_allocation

        # Store portfolio state
        self.portfolio_history.append(self.portfolio.copy())
        self.action_history.append(action)

        # Move to next time step
        self.current_step += 1
        if self.current_step < len(self.dates):
            self.current_date = self.dates[self.current_step]
        else:
            # Handle the case where we've reached the end of available dates
            self.current_date = self.dates[-1]

        # Store date
        self.date_history.append(self.current_date)

        # Calculate new portfolio value AFTER transaction costs and market movement
        market_updated_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value = market_updated_portfolio_value

        # Store portfolio value
        self.value_history.append(self.portfolio_value)

        # Calculate portfolio returns for volatility estimation
        if len(self.value_history) > 1:
            log_returns = np.log(
                np.array(self.value_history[1:]) / np.array(self.value_history[:-1])
            )
            portfolio_volatility = np.std(log_returns) if len(log_returns) > 1 else 0.0
        else:
            portfolio_volatility = 0.0

        # Calculate reward using risk-adjusted return with log returns
        if old_portfolio_value > 0:
            log_return = np.log(self.portfolio_value / old_portfolio_value)
            reward = RewardCalculator.risk_adjusted_return(
                self.portfolio_value,
                old_portfolio_value,
                portfolio_volatility,
                risk_free_rate=0.03,  # Approximate risk-free rate
                risk_aversion=0.7,  # Balance between return and risk
                num_days=63,
                periods_per_year=252
            )

            # Apply scaling to prevent exploding rewards
            reward = np.clip(reward, -3.0, 3.0)
        else:
            reward = 0.0  # Fallback if old_portfolio_value is zero

        # Check if episode is done
        terminated = self.current_step >= len(self.dates) - 1
        truncated = self.current_step >= self.window_size + self.max_steps
        self.done = terminated or truncated

        # Get next state
        next_state = self._get_state()

        # Additional info
        info = {
            "portfolio_value": self.portfolio_value,
            "transaction_cost": total_trading_cost,
            "date": self.current_date,
            "portfolio_allocation": self.portfolio.copy(),
            "old_portfolio_value": old_portfolio_value,
            "return": reward,
            "volatility": portfolio_volatility,
        }

        # For backward compatibility with older code
        if hasattr(self, "_return_old_step_api") and self._return_old_step_api:
            return next_state, reward, self.done, info

        # Return in the format expected by newer gym versions
        return next_state, reward, terminated, truncated, info

    def _action_to_allocation(self, action):
        """
        Convert action index to portfolio allocation.

        Args:
            action: Action value (can be index or direct allocation vector)

        Returns:
            Array of portfolio allocations
        """
        # Validate action against action space if available
        if hasattr(self, "action_space"):
            if isinstance(action, np.ndarray) and not self.action_space.contains(
                    action
            ):
                # Clip action to be within action space bounds
                action = np.clip(action, self.action_space.low, self.action_space.high)

        n_assets = len(self.tickers)
        max_allocation_per_asset = 0.4  # 40% maximum per position

        # If action is an index, convert to predefined allocation
        if isinstance(action, (int, np.integer)):
            # Generate predefined allocations
            if action == 0:
                # Equal allocation
                return np.ones(n_assets) / n_assets

            elif action < n_assets + 1:
                # All in one asset - but cap at max_allocation_per_asset
                allocation = np.ones(n_assets) * 0.05  # Base 5% for all assets
                primary_asset = action - 1
                secondary_asset = (primary_asset + 1) % n_assets  # Next asset in line
                third_asset = (primary_asset + 2) % n_assets  # Third asset in line

                # Primary gets 30%, secondary gets 15%, tertiary gets 10%, rest get 5%
                allocation[primary_asset] = 0.30
                allocation[secondary_asset] = 0.15
                allocation[third_asset] = 0.10

                # Normalize to ensure sum is 1.0
                allocation = allocation / allocation.sum()
                return allocation

            elif action == n_assets + 1:
                # 60/40 portfolio but ensure no position exceeds max_allocation_per_asset
                allocation = np.zeros(n_assets)
                mid_point = n_assets // 2

                if mid_point > 0:
                    per_asset_first_half = 0.6 / mid_point
                    if per_asset_first_half > max_allocation_per_asset:
                        # Adjust if it would exceed max per position
                        allocation[:mid_point] = max_allocation_per_asset
                        # Redistribute excess equally to second half
                        excess = 0.6 - (max_allocation_per_asset * mid_point)
                        if n_assets - mid_point > 0:
                            allocation[mid_point:] = (0.4 + excess) / (n_assets - mid_point)
                    else:
                        allocation[:mid_point] = per_asset_first_half
                        if n_assets - mid_point > 0:
                            allocation[mid_point:] = 0.4 / (n_assets - mid_point)
                else:
                    # Edge case: only one asset
                    allocation[0] = 1.0

                return allocation

            else:
                # Default: equal weights
                return np.ones(n_assets) / n_assets

        # If action is a vector, ensure it sums to 1 and cap individual positions
        elif isinstance(action, np.ndarray):
            # Convert to float32 for consistency
            action = action.astype(np.float32)

            # Ensure non-negative allocations
            action = np.maximum(action, 0)

            # First, normalize to sum to 1
            if np.sum(action) > 0:
                action = action / np.sum(action)
            else:
                # If all zeros, return equal allocation
                return np.ones(n_assets, dtype=np.float32) / n_assets

            # Now apply the maximum allocation cap
            # Find positions exceeding the cap
            excess_mask = action > max_allocation_per_asset
            excess_count = np.sum(excess_mask)

            if excess_count > 0:
                # Calculate total excess allocation
                total_excess = np.sum(action[excess_mask] - max_allocation_per_asset)

                # Cap the exceeding positions
                action[excess_mask] = max_allocation_per_asset

                # Redistribute excess to positions below the cap
                below_cap_mask = ~excess_mask
                below_cap_count = np.sum(below_cap_mask)

                if below_cap_count > 0:
                    # Distribute excess equally among positions below cap
                    action[below_cap_mask] += total_excess / below_cap_count

                    # Ensure redistribution doesn't push any position over the cap
                    over_after_redist = action > max_allocation_per_asset
                    if np.any(over_after_redist & below_cap_mask):
                        # Normalize again as a simple solution
                        action = np.minimum(action, max_allocation_per_asset)
                        if np.sum(action) > 0:
                            action = action / np.sum(action)
                        else:
                            return np.ones(n_assets, dtype=np.float32) / n_assets

            return action

        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def _calculate_transaction_cost(self, new_allocation):
        """
        Calculate transaction cost for portfolio reallocation.

        Args:
            new_allocation: New portfolio allocation

        Returns:
            Transaction cost
        """
        # Calculate the absolute difference between current and new allocations
        allocation_diff = np.abs(self.portfolio - new_allocation)

        # Calculate the total value being reallocated
        value_reallocated = np.sum(allocation_diff) * self.portfolio_value

        # Apply transaction cost
        return value_reallocated * self.transaction_cost

    def _calculate_portfolio_value(self):
        """
        Calculate the current portfolio value.

        Returns:
            Current portfolio value
        """
        # Get current date
        current_date = self.dates[min(self.current_step, len(self.dates) - 1)]

        # Get current prices for each ticker
        asset_values = []
        for i, ticker in enumerate(self.tickers):
            if ticker not in self.ticker_price_data:
                asset_values.append(0.0)
                continue

            ticker_price_data = self.ticker_price_data[ticker]
            current_price_data = ticker_price_data[
                ticker_price_data["date"] == current_date
                ]

            if not current_price_data.empty:
                price = current_price_data["price"].values[0]
            else:
                # Use the last available price if no data for current date
                recent_prices = ticker_price_data[
                    ticker_price_data["date"] < current_date
                    ].sort_values("date")
                if not recent_prices.empty:
                    price = recent_prices.tail(1)["price"].values[0]
                else:
                    # If no prior prices available, use a default value
                    price = 1.0

            # Calculate value of holdings in this asset
            asset_values.append(self.portfolio[i] * self.initial_balance * price)

        # Calculate portfolio value
        portfolio_value = self.balance + np.sum(asset_values)

        return portfolio_value

    def get_portfolio_history(self):
        """
        Get the history of portfolio values and allocations.

        Returns:
            Dictionary with portfolio history data
        """
        history = {
            "dates": self.date_history,
            "portfolio_values": self.value_history,
            "allocations": self.portfolio_history,
            "actions": self.action_history,
        }

        return history

    def render(self, mode="human"):
        """
        Render the environment (simplified).

        Args:
            mode: Rendering mode
        """
        print(f"Date: {self.current_date}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Cash Balance: ${self.balance:.2f}")
        print("Portfolio Allocation:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.portfolio[i]:.4f}")

    def get_sector_allocations(self):
        """
        Calculate current allocation by sector.

        Returns:
            Dictionary with sector allocations
        """
        sector_allocations = {}
        for i, ticker in enumerate(self.tickers):
            sector = self.ticker_sector_map.get(ticker, "Unknown")
            if sector not in sector_allocations:
                sector_allocations[sector] = 0
            sector_allocations[sector] += self.portfolio[i]

        return sector_allocations

    def check_rebalancing_trigger(
            self, ticker, ratio_name="return_on_equity", threshold=0.8
    ):
        """Check if a ticker underperforms relative to its sector."""
        current_date = self.dates[self.current_step]
        sector = self.ticker_sector_map.get(ticker, "Unknown")

        ticker_data = self.financial_data[
            (self.financial_data["ticker"] == ticker)
            & (self.financial_data["date"] == current_date)
            ]

        if not ticker_data.empty and ratio_name in ticker_data.columns:
            ticker_ratio = ticker_data[ratio_name].values[0]

            if (
                    sector in self.sector_metrics
                    and ratio_name in self.sector_metrics[sector].columns
            ):
                sector_ratio = self.sector_metrics[sector].loc[current_date, ratio_name]

                if ticker_ratio < (sector_ratio * threshold):
                    return True
        return False
