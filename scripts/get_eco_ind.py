import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

# Define the economic indicators and their FRED codes
indicators = {
    "RGDP": "GDPC1",  # Real Gross Domestic Product (Chained 2012 Dollars)
    "NGDP": "GDP",  # Nominal Gross Domestic Product
    "RDPI": "DSPIC96",  # Real Disposable Personal Income
    "NDPI": "DSPI",  # Nominal Disposable Personal Income
    "UR": "UNRATE",  # Unemployment Rate
    "UC": "CPIAUCSL",  # Consumer Price Index, Urban Consumers
    "X3MT": "DGS3MO",  # 3-Month Treasury Yield
    "X5MT": "DGS5",  # 5-Year Treasury Yield
    "X10YT": "DGS10",  # 10-Year Treasury Yield
    "BBB": "BAMLC0A4CBBBEY",  # ICE BofA 7-10 Year BBB Corporate Index
    "X30YC": "MORTGAGE30US",  # 30-Year Fixed Mortgage Commitment Rate
    "PR": "DPRIME",  # Bank Prime Loan Rate
    "DJI": "SP500",  # Dow Jones US Total Stock Market Index
    "ORE": "CSUSHPISA",  # Home Price Index, Owner-Occupied Real Estate
    "CREPI": "BOGZ1FL075035503Q",  # Commercial Real Estate Price Index
    "VIX": "VIXCLS",  # Volatility Index (VIX)
    "X103S": "T10Y3M",  # 10-Year Treasury Yield - 3-Month Treasury Yield
}

# Define the time period
start_date = datetime(2014, 1, 1)
end_date = datetime(2024, 12, 31)

# Dictionary to store fetched data
data_frames = {}

# Fetch data and apply appropriate resampling
for name, series_id in indicators.items():
    print(f"Fetching data for {name} ({series_id})...")
    try:
        df = web.DataReader(series_id, "fred", start_date, end_date)
        
        # Resampling rules based on data frequency
        if name in ["RGDP", "NGDP", "CREPI"]:  # Quarterly data
            df = df.resample("Q").mean()
        elif name in ["RDPI", "NDPI", "UR", "UC", "ORE"]:  # Monthly data
            df = df.resample("Q").mean()  # Convert to Quarterly
        elif name in ["X30YC"]:  # Weekly data
            df = df.resample("Q").mean()
        else:  # Daily data (Treasury yields, VIX, DJI, PR, etc.)
            df = df.resample("Q").mean()
        
        df.dropna(inplace=True)  # Drop missing values
        data_frames[name] = df

    except Exception as e:
        print(f"Error fetching {name}: {e}")

# Combine all data into a single DataFrame
final_df = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())

# Flatten the column names to avoid multi-level headers
final_df.columns = [col[0] for col in final_df.columns]

# Convert index to the last date of each quarter (e.g., '3/31/2014')
final_df.index = final_df.index.to_period('Q').dt.end_time

# Format the datetime index to 'month/day/year' (e.g., '3/31/2014')
final_df.index = final_df.index.strftime('%m/%d/%Y')

# Round all numeric columns to 4 decimal places
final_df = final_df.round(4)

# Save to CSV with the proper format (quarters as rows)
final_df.to_csv("../data/processed/economic_indicators_quarterly_2014_2024.csv")

print("Data successfully saved to CSV!")