"""
Common utilities for financial data extraction from Alpha Vantage API.

This module provides shared functionality for fetching and processing
financial data across different financial statement types (balance sheet,
income statement, cash flow). It handles API interactions, data conversion,
and error handling.

Functions:
    load_api_key: Load Alpha Vantage API key from config file
    fetch_alpha_vantage_data: Generic function to fetch data from Alpha Vantage API
    safe_float_convert: Safely convert values to float type

Usage:
    from .common import fetch_alpha_vantage_data, safe_float_convert, logger
"""

import os
import logging
import time
import json
from typing import Any, Dict, Optional, Union
from pandas.tseries import api
import requests
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_api_key(config_file: str = "config/config.yaml"):
    """Load Alpha Vantage API key from config file"""
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if "alpha_vantage" not in config or "api_key" not in config["alpha_vantage"]:
            raise KeyError("Alpha Vantage API key not found in config file")
        return config["alpha_vantage"]["api_key"]
    except Exception as e:
        logger.error("Error loading API key: %s", str(e))
        raise


def fetch_alpha_vantage_data(
    function: str,
    symbol: str,
    api_key: str,
    max_retries: int = 3,
    outputsize: str = "full",
    datatype: str = "json",
) -> Optional[Union[Dict[str, Any], str]]:
    """Generic function to fetch data from Alpha Vantage API"""
    # Get config data
    config_file = "config/config.yaml"
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    contact_email = config["alpha_vantage"]["contact_email"]

    # Base URL
    base_url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}"

    # Additional parameters for stock price data
    if function == "TIME_SERIES_DAILY":
        url = f"{base_url}&outputsize={outputsize}&datatype={datatype}"
    else:
        url = base_url

    headers = {"User-Agent": f"PortfolioProphets-UMichCapstone/1.0 ({contact_email})"}

    retries = 0
    while retries < max_retries:
        try:
            logger.info("Fetching %s data for %s", function, symbol)
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            if datatype.lower() == "csv" and function == "TIME_SERIES_DAILY":
                return response.text
            else:
                data = response.json()

                # Check if the API returned an error message
                if "Error Message" in data:
                    logger.warning("Alpha Vantage API error: %s", data["Error Message"])
                    raise ValueError("Alpha Vantage API error")

                # Check for API limit message
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning("Alpha Vantage API rate limit exceeded")
                    time.sleep(60)  # Sleep for 60 seconds before retrying
                    retries += 1
                    continue

                return data

        except requests.exceptions.RequestException as e:
            logger.error("Alpha Vantage API request error: %s", str(e))
            retries += 1
            time.sleep(5)  # Short delay before retry
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            return None

    logger.error(
        "Failed to fetch %s data for %s after %d retries", function, symbol, max_retries
    )
    return None


def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def save_to_file(data: Dict[str, Any], filename: str) -> bool:
    """Save processed data to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Successfully saved data to %s", filename)
        return True
    except Exception as e:
        logger.error("Error saving data to %s: %s", filename, str(e))
        return False


def save_csv_to_file(csv_data: str, filename: str) -> bool:
    """Save raw CSV data to a file.

    Args:
        csv_data (str): Raw CSV data as string
        filename (str): Output filename

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Write the CSV data directly to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(csv_data)

        logger.info("Successfully saved CSV data to %s", filename)
        return True
    except Exception as e:
        logger.error("Error saving CSV data to %s: %s", filename, str(e))
        return False
