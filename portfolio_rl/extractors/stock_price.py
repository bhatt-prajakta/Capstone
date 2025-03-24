from tkinter.filedialog import SaveAs

"""
Stock price data extraction from Alpha Vantage API.

This module provides functionality for fetching and processing
daily adjusted stock price data from Alpha Vantage. It handles API interactions,
data conversion, and error handling for time series data.

Functions:
    fetch_stock_prices: Fetch daily adjusted stock price data
    process_daily_adjusted_prices: Process raw stock price data into a structured format
    fetch_and_process_stock_prices: Combined function to fetch, process and save stock price data

Usage:
    from .stock_price import fetch_and_process_stock_prices
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from .common import (
    fetch_alpha_vantage_data,
    load_api_key,
    safe_float_convert,
    save_to_file,
    save_csv_to_file,
    logger,
)


def fetch_stock_prices(
    symbol: str, api_key: str, outputsize: str = "full", datatype: str = "csv"
) -> Optional[Dict[str, Any]]:
    """
    Fetch daily adjusted stock price data for a specific symbol from Alpha Vantage.

    Args:
        symbol (str): The stock symbol
        api_key (str): Alpha Vantage API key
        outputsize (str): 'compact' for most recent 100 datapoints, 'full' for 20+ years of data

    Returns:
        Optional[Dict[str, Any]]: Daily adjusted stock price data or None if failed
    """
    # For TIME_SERIES_DAILY we need to include the outputsize parameter
    function = "TIME_SERIES_DAILY"

    # Use the common fetch function with the full URL
    return fetch_alpha_vantage_data(
        function, symbol, api_key, outputsize=outputsize, datatype=datatype
    )


def process_csv_stock_prices(csv_data: str, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Process stock price data received in CSV format.

    Args:
        csv_data (str): Raw CSV stock price data from Alpha Vantage
        symbol (str): The stock symbol

    Returns:
        dict: Processed stock price data in consistent format
    """
    if not csv_data or not isinstance(csv_data, str) or len(csv_data) < 10:
        logger.warning("No CSV data or invalid CSV provided.")
        return None

    processed_data = {
        "symbol": symbol,
        "last_refreshed": datetime.now().strftime("%Y-%m-%d"),
        "output_size": "full",
        "time_zone": "US/Eastern",
        "daily_prices": [],
    }

    try:
        # Split lines and process
        lines = csv_data.strip().split("\n")
        headers = lines[0].split(",")

        daily_prices = []

        # Process all data rows (skip header)
        for line in lines[1:]:
            if not line.strip():
                continue

            values = line.split(",")
            if len(values) < len(headers):
                continue

            row_data = dict(zip(headers, values))

            processed_day = {
                "date": row_data.get("timestamp", ""),
                "open": safe_float_convert(row_data.get("open")),
                "high": safe_float_convert(row_data.get("high")),
                "low": safe_float_convert(row_data.get("low")),
                "close": safe_float_convert(row_data.get("close")),
                "volume": safe_float_convert(row_data.get("volume")),
                # Note: The free tier CSV doesn't include these fields
                "adjusted_close": None,
                "dividend_amount": None,
                "split_coefficient": None,
            }

            daily_prices.append(processed_day)

        # After processing all days, sort chronologically (oldest first)
        sorted_prices = sorted(
            daily_prices, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d")
        )

        # Assign sorted prices to processed_data
        processed_data["daily_prices"] = sorted_prices

        # Add metadata
        processed_data["total_days"] = len(processed_data["daily_prices"])
        if processed_data["daily_prices"]:
            processed_data["date_range"] = {
                "start": processed_data["daily_prices"][0]["date"],
                "end": processed_data["daily_prices"][-1]["date"],
            }
        else:
            processed_data["date_range"] = {"start": None, "end": None}

    except Exception as e:
        logger.error("Error processing CSV stock price data: %s", str(e))
        return None

    return processed_data


def fetch_and_process_stock_prices(
    symbol: str,
    api_key: str,
    outputsize: str = "full",
    datatype: str = "csv",
    output_dir: str = "data/stock_prices",
) -> Optional[Dict[str, Any]]:
    """
    Fetch and process daily stock price data for a symbol.

    Args:
        symbol (str): Stock symbol
        api_key (str): Alpha Vantage API key
        outputsize (str): 'compact' for most recent 100 datapoints, 'full' for 20+ years of data
        datatype (str): Format of data to retrieve ('json' or 'csv')
        output_dir (str): Directory to save the processed data

    Returns:
        Optional[Dict[str, Any]]: Processed stock price data
    """
    # Fetch stock price data
    stock_prices = fetch_stock_prices(symbol, api_key, outputsize, datatype)

    if not stock_prices:
        logger.warning("Failed to fetch stock prices for %s", symbol)
        return None

    # Process the data based on datatype
    if datatype.lower() == "csv" and isinstance(stock_prices, str):
        processed_stock_prices = process_csv_stock_prices(stock_prices, symbol)

        # Also save the raw CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = os.path.join(output_dir, f"{symbol}_stock_prices.csv")
        with open(csv_filename, "w", encoding="utf-8") as f:
            f.write(stock_prices)
            logger.info("Saved raw CSV data to %s", csv_filename)
    else:
        processed_stock_prices = process_daily_adjusted_prices(stock_prices)

    if processed_stock_prices:
        # Save the processed data as JSON
        filename = os.path.join(output_dir, f"{symbol}_stock_prices.json")
        save_to_file(processed_stock_prices, filename)

    return processed_stock_prices


if __name__ == "__main__":
    # Load API key
    api_key = load_api_key()

    # CSV fetching
    symbol = "AX"
    output_dir = "data/stock_prices"  # Define the output directory

    # Fetch the CSV data
    csv_data = fetch_stock_prices(symbol, api_key, outputsize="full", datatype="csv")

    if csv_data:
        # Make sure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create the correct filename with path
        csv_filename = os.path.join(output_dir, f"{symbol}_stock_prices.csv")

        # Save the CSV data with the full path
        if save_csv_to_file(csv_data, csv_filename):
            print(f"Successfully saved CSV data for {symbol} to {csv_filename}")
        else:
            print(f"Failed to save CSV data for {symbol}")
    else:
        print(f"Failed to fetch data for {symbol}")
