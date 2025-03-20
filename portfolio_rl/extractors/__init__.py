"""
Financial data extractors for Alpha Vantage API.

This package provides modules for fetching and processing financial data
from Alpha Vantage, organized by financial statement type.

Modules:
    common: Shared utilities and helper functions
    balance_sheet: Balance sheet data extraction and processing
    income_statement: Income statement data extraction and processing
    cashflow_statement: Cashflow statement data extraction and processing

Each module provides functions to fetch raw data from Alpha Vantage,
process it into a standardized format, and save it to files.
"""

# Import key functions to make them available directly from the extractors package
from .common import load_api_key, safe_float_convert
from .balance_sheet import (
    fetch_balance_sheet,
    process_quarterly_balance_sheet,
    fetch_and_process_balance_sheet,
)
from .income_statement import (
    fetch_income_statement,
    process_quarterly_income_statement,
    fetch_and_process_income_statement,
)

from .cashflow_statement import (
    fetch_cashflow_statement,
    process_quarterly_cashflow_statement,
    fetch_and_process_cashflow_statement,
)

# Define what gets imported with `from extractors import *`
__all__ = [
    "load_api_key",
    "safe_float_convert",
    "fetch_balance_sheet",
    "process_quarterly_balance_sheet",
    "fetch_and_process_balance_sheet",
    "fetch_income_statement",
    "process_quarterly_income_statement",
    "fetch_and_process_income_statement",
    "fetch_cashflow_statement",
    "process_quarterly_cashflow_statement",
    "fetch_and_process_cashflow_statement",
]
