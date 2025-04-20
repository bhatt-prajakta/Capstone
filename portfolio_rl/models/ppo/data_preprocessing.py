import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data_sources():
    """
    Load all data sources for the portfolio optimization project.
    
    Returns:
        tuple: (stock_returns, fundamental_data, sentiment_data, economic_data)
    """
    try:
        # Load stock returns, sentiment and economic indicators data
        stock_returns = pd.read_csv('../data/processed/stock_return_data.csv')
        sentiment_data = pd.read_csv('../data/processed/sector_sentiment_with_metrics_quarterly_2014_2024.csv')
        economic_data = pd.read_csv('../data/processed/economic_indicators_quarterly_2014_2024.csv')

        # Load sector mapping from file
        global sector_mapping
        if 'sector_mapping' not in globals():
            sector_mapping = pd.read_csv('../data/sector_company_mapping.csv')

        # Load fundamental ratios
        fundamental_data = pd.DataFrame()
        for ticker in sector_mapping['ticker']:
            file_path = f'../data/processed/ratios/{ticker}_ratios.csv'
            if os.path.exists(file_path):
                sector_data = pd.read_csv(file_path)
                sector_data['sector'] = sector_mapping.loc[sector_mapping['ticker'] == ticker, 'sector'].values[0]
                fundamental_data = pd.concat([fundamental_data, sector_data])
            else:
                print(f"Warning: File not found for ticker {ticker}. Skipping.")

        # Load economic indicators
        economic_data['Date'] = pd.to_datetime(economic_data['Date'])
        economic_data.reset_index()

        # Rename date columns to 'Date' for consistency
        stock_returns.rename(columns={'DlyCalDt': 'Date'}, inplace=True)
        fundamental_data.rename(columns={'date': 'Date'}, inplace=True)

        # Datetime conversion
        stock_returns['Date'] = pd.to_datetime(stock_returns['Date'])
        fundamental_data['Date'] = pd.to_datetime(fundamental_data['Date'])
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

        print("All data sources loaded successfully")
        return stock_returns, fundamental_data, sentiment_data, economic_data
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure all required data files are in the correct directories")
        return None, None, None, None

def preprocess_data(stock_returns, fundamental_data, sentiment_data, economic_data):
    """    Preprocess and align all data sources to daily frequency.        
    Args:        
        stock_returns: DataFrame with daily stock returns        
        fundamental_data: DataFrame with quarterly fundamental ratios        
        sentiment_data: DataFrame with daily or periodic sentiment scores        
        economic_data: DataFrame with quarterly economic indicators           
    Returns:        
        DataFrame: Merged dataset with all features aligned to daily frequency    
    
    """
    # Ensure all dates are in datetime format    
    for df in [stock_returns, fundamental_data, sentiment_data, economic_data]:
        if df is not None and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Quarter'] = df['Date'].dt.quarter

    # Create a complete daily date range    
    min_date = stock_returns['Date'].min()
    max_date = stock_returns['Date'].max()
    all_dates = pd.DataFrame({'Date': pd.date_range(min_date, max_date, freq='D')})
    all_dates['Year'] = all_dates['Date'].dt.year
    all_dates['Quarter'] = all_dates['Date'].dt.quarter
    
    # Process stock returns    
    if 'Log_Return' not in stock_returns.columns:
        stock_returns['Log_Return'] = np.log(1 + stock_returns['DlyRet'].clip(-0.99, np.inf))

    stock_returns.rename(columns={'Ticker': 'ticker'}, inplace=True)
    stock_returns = pd.merge(stock_returns, sector_mapping, on='ticker', how='left')

    # Create sector-day panel for returns
    sector_returns = stock_returns.pivot_table(index='Date', columns='sector', values='Log_Return')
    sector_returns = sector_returns.add_suffix('_Return')
    sector_returns = sector_returns.reset_index()
    
    # Process fundamental data
    if fundamental_data is not None and not fundamental_data.empty:
        fundamental_features = [col for col in fundamental_data.columns if col not in ['Date', 'Year', 'Quarter', 'sector']]
        
        # Clean data
        for col in fundamental_features:
            if col in fundamental_data.columns:
                fundamental_data[col] = fundamental_data[col].replace([np.inf, -np.inf], np.nan)
                fundamental_data[col] = fundamental_data[col].clip(-1e6, 1e6)
        
        # Use train_end_date instead of current_date to prevent look-ahead bias
        fundamental_pivot = fundamental_data.pivot_table(
            index=['Year', 'Quarter'],
            columns='sector',
            values=fundamental_features,
            aggfunc='mean')
        
        fundamental_pivot = fundamental_pivot.fillna(method='ffill')
        
        if isinstance(fundamental_pivot.columns, pd.MultiIndex):
            fundamental_pivot.columns = ['_'.join([str(col[1]), str(col[0])]) for col in fundamental_pivot.columns]
            fundamental_pivot = fundamental_pivot.add_prefix('fund_')
            fundamental_pivot = fundamental_pivot.reset_index()
    else:
        fundamental_pivot = pd.DataFrame({'Year': [], 'Quarter': []})

    # Process sentiment data
    if sentiment_data is not None and not sentiment_data.empty:
    # Convert Date to datetime
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

        # Add Year and Quarter
        sentiment_data['Year'] = sentiment_data['Date'].dt.year
        sentiment_data['Quarter'] = sentiment_data['Date'].dt.quarter

        # Identify sentiment-related feature columns
        sentiment_features = [col for col in sentiment_data.columns if col not in ['Date', 'Year', 'Quarter']]

        # Clean and clip values
        for col in sentiment_features:
            if col in sentiment_data.columns:
                sentiment_data[col] = sentiment_data[col].replace([np.inf, -np.inf], np.nan)
                sentiment_data[col] = sentiment_data[col].clip(-10, 10)

        # Add prefix for clarity
        sentiment_data = sentiment_data.add_prefix('sent_')

        # Restore Date, Year, Quarter column names after prefixing
        sentiment_data = sentiment_data.rename(columns={
            'sent_Date': 'Date',
            'sent_Year': 'Year',
            'sent_Quarter': 'Quarter'
        })

        # Set the final pivoted/processed DataFrame
        sentiment_pivot = sentiment_data.copy()

    else:
        sentiment_pivot = pd.DataFrame({'Date': []})
          
    # Process economic indicators
    if economic_data is not None and not economic_data.empty:
        economic_features = [col for col in economic_data.columns if col not in ['Date', 'Year', 'Quarter']]
        
        for col in economic_features:
            economic_data[col] = pd.to_numeric(economic_data[col], errors='coerce')
            economic_data[col] = economic_data[col].replace([np.inf, -np.inf], np.nan)
            if pd.api.types.is_numeric_dtype(economic_data[col]):
                economic_data[col] = economic_data[col].clip(-1e6, 1e6)
        
        economic_pivot = economic_data.set_index(['Year', 'Quarter'])
        economic_features = [col for col in economic_pivot.columns if col not in ['Date']]
        economic_pivot = economic_pivot[economic_features]
        economic_pivot = economic_pivot.add_prefix('econ_')
        economic_pivot = economic_pivot.reset_index()
    
    else:
        economic_pivot = pd.DataFrame({'Year': [], 'Quarter': []})
        
    # Merge all datasets
    merged_data = pd.merge(all_dates, sector_returns, on='Date', how='left')

    if not fundamental_pivot.empty and 'Year' in fundamental_pivot.columns:
        merged_data = pd.merge(merged_data, fundamental_pivot, on=['Year', 'Quarter'], how='left')
    
    if not economic_pivot.empty and 'Year' in economic_pivot.columns:
        merged_data = pd.merge(merged_data, economic_pivot, on=['Year', 'Quarter'], how='left')
    
    #if not sentiment_pivot.empty and 'Year' in sentiment_pivot.columns:
        #merged_data = pd.merge(merged_data, sentiment_pivot, on=['Year', 'Quarter'], how='left')
    
    if 'Year' in sentiment_pivot.columns:
        sentiment_pivot = sentiment_pivot.drop(columns=['Date'], errors='ignore')
        merged_data = pd.merge(merged_data, sentiment_pivot, on=['Year', 'Quarter'], how='left')
    elif 'Date' in sentiment_pivot.columns:
        sentiment_pivot = sentiment_pivot.drop(columns=['Year', 'Quarter'], errors='ignore')
        merged_data = pd.merge(merged_data, sentiment_pivot, on='Date', how='left')
        
    # Drop non-numeric columns except Date, Year, Quarter
    non_numeric_cols = [col for col in merged_data.columns
                        if not pd.api.types.is_numeric_dtype(merged_data[col])
                        and col not in ['Date', 'Year', 'Quarter']]
    merged_data = merged_data.drop(columns=non_numeric_cols)
    merged_data.set_index('Date', inplace=True)

    # Handle missing values
    merged_data = merged_data.fillna(method='ffill')
    merged_data = merged_data.fillna(merged_data.mean())
    merged_data = merged_data.fillna(0)
    
    # Clean any remaining issues
    for col in merged_data.columns:
        if pd.api.types.is_numeric_dtype(merged_data[col]):
            merged_data[col] = merged_data[col].replace([np.inf, -np.inf], 0)
            
    # Standardize features
    feature_columns = [col for col in merged_data.columns
                       if any(prefix in col for prefix in ['fund_', 'sent_', 'econ_'])]
    
    if feature_columns:
        try:
            scaler = StandardScaler()
            merged_data[feature_columns] = scaler.fit_transform(merged_data[feature_columns])
            
        except Exception:
            # Manual standardization
            for col in feature_columns:
                mean = merged_data[col].mean()
                std = merged_data[col].std()
                if std > 0:
                    merged_data[col] = (merged_data[col] - mean) / std
                else:
                    merged_data[col] = 0
                merged_data[col] = merged_data[col].clip(-5, 5)
    return merged_data