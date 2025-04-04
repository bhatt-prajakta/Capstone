"""
Financial Statement Analysis Module (processors/financial_statement.py)

This module provides comprehensive functionality for loading, processing, and analyzing
financial statements from publicly traded companies. It consolidates balance sheet,
income statement, and cash flow statement data into a single dataframe, and offers
analytical tools to calculate key financial ratios across several categories:

- Activity ratios: Measure operational efficiency (inventory turnover, receivables turnover, etc.)
- Liquidity ratios: Assess short-term payment capability (current ratio, quick ratio, etc.)
- Solvency ratios: Evaluate long-term debt obligations (debt-to-equity, leverage, etc.)
- Profitability ratios: Analyze earning capability (profit margins, ROA, ROE, etc.)

Usage:
    To analyze a specific company:
        python portfolio_rl/processors/financial_statement.py AAPL

    Additional options:
        --output_dir: Specify output directory (default: current directory)

Data Assumptions:
    - Quarterly financial data is used in calculations
    - Average values use current and previous quarter
    - First row may contain NaN for ratios requiring averages

Returns:
    For each ticker symbol, the module generates:
    - A combined financial statement CSV file
    - A comprehensive financial ratios CSV file with all calculated metrics
"""

import os
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


# Constants
DAYS_IN_QUARTER = 90
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Define subdirectories
BALANCE_SHEETS_DIR = os.path.join(RAW_DATA_DIR, "balance_sheets")
INCOME_STATEMENTS_DIR = os.path.join(RAW_DATA_DIR, "income_statements")
CASHFLOW_STATEMENTS_DIR = os.path.join(RAW_DATA_DIR, "cashflow_statements")


def load_financial_statements(ticker: str) -> pd.DataFrame:
    """
    Load balance sheet, income statement, and cashflow statement data for a given ticker
    and return a combined dataframe.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
        pd.DataFrame: Combined financial statements data for the given ticker
    """
    # Validate path exists
    for path in [BALANCE_SHEETS_DIR, INCOME_STATEMENTS_DIR, CASHFLOW_STATEMENTS_DIR]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

    balance_sheet_path = os.path.join(
        BALANCE_SHEETS_DIR, f"{ticker}_balance_sheet.json"
    )
    income_statement_path = os.path.join(
        INCOME_STATEMENTS_DIR, f"{ticker}_income_statement.json"
    )
    cashflow_statement_path = os.path.join(
        CASHFLOW_STATEMENTS_DIR, f"{ticker}_cashflow_statement.json"
    )

    try:
        # Load JSON data
        with open(balance_sheet_path, "r", encoding="utf-8") as f:
            balance_sheet_data = json.load(f)
        with open(income_statement_path, "r", encoding="utf-8") as f:
            income_statement_data = json.load(f)
        with open(cashflow_statement_path, "r", encoding="utf-8") as f:
            cashflow_statement_data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Financial statement file not found: {e.filename}"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(
            "Invalid JSON format in one of the financial statement files"
        ) from e

    try:
        # Convert JSON data to DataFrames
        balance_sheet_df = pd.json_normalize(balance_sheet_data["quarterly_reports"])
        income_statement_df = pd.json_normalize(
            income_statement_data["quarterly_reports"]
        )
        cashflow_statement_df = pd.json_normalize(
            cashflow_statement_data["quarterly_reports"]
        )
    except KeyError as e:
        raise ValueError(f"Missing expected key in financial data: {str(e)}") from e

    # Convert date strings to datetime objects before merging
    balance_sheet_df["fiscal_date_ending"] = pd.to_datetime(
        balance_sheet_df["fiscal_date_ending"]
    )
    income_statement_df["fiscal_date_ending"] = pd.to_datetime(
        income_statement_df["fiscal_date_ending"]
    )
    cashflow_statement_df["fiscal_date_ending"] = pd.to_datetime(
        cashflow_statement_df["fiscal_date_ending"]
    )

    # Drop reported_currency columns, only calculating fundamental ratios
    balance_sheet_df.drop(columns=["reported_currency"], inplace=True)
    income_statement_df.drop(columns=["reported_currency"], inplace=True)
    cashflow_statement_df.drop(columns=["reported_currency"], inplace=True)

    # Merged DataFrames on fiscal_date_ending
    merged_df = pd.merge(
        balance_sheet_df, income_statement_df, on="fiscal_date_ending", how="left"
    )

    # Merge with cashflow_statement_df on fiscal_date_ending
    merged_df = pd.merge(
        merged_df, cashflow_statement_df, on="fiscal_date_ending", how="left"
    )

    # Sort by fiscal_date_ending (most recent last)
    merged_df = merged_df.sort_values(by="fiscal_date_ending", ascending=True)

    # Handle potential duplicate columns from merging
    if "net_income_x" in merged_df.columns and "net_income_y" in merged_df.columns:
        merged_df.rename(columns={"net_income_x": "net_income"}, inplace=True)
        merged_df.drop(columns=["net_income_y"], inplace=True)

    # Fill missing values as NaN
    merged_df.fillna(value=np.nan, inplace=True)

    # Reset index and drop the index column
    merged_df = merged_df.reset_index(drop=True)

    return merged_df


def calculate_activity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate activity ratios based on the provided financial statements dataframe.

    This function currently calculates activity ratios including:
    - Inventory turnover and days of inventory on hand
    - Receivables turnover and days of sales outstanding
    - Payables turnover and number of days of payables
    - Working capital turnover
    - Fixed asset turnover
    - Total asset turnover

    Args:
        df (pd.DataFrame): Combined financial statements dataframe from load_financial_statements()

    Returns:
        pd.DataFrame: Dataframe containing activity financial ratios

    Notes:
        - This function assumes quarterly financial data
        - Average values are calculated using current and previous quarter values
        - First row will contain NaN values for ratios requiring average calculations
    """

    activity_df = pd.DataFrame()
    activity_df["date"] = df["fiscal_date_ending"]

    # Calculate average values and intermediate values
    activity_df["average_inventory"] = (df["inventory"] + df["inventory"].shift(1)) / 2
    activity_df["average_receivables"] = (
        df["current_net_receivables"] + df["current_net_receivables"].shift(1)
    ) / 2
    activity_df["average_payables"] = (
        df["current_accounts_payable"] + df["current_accounts_payable"].shift(1)
    ) / 2

    activity_df["working_capital"] = (
        df["total_current_assets"] - df["total_current_liabilities"]
    )
    activity_df["average_working_capital"] = (
        activity_df["working_capital"] + activity_df["working_capital"].shift(1)
    ) / 2

    activity_df["net_fixed_assets"] = df["property_plant_equipment"] - np.where(
        df["accumulated_depreciation_amortization_ppe"].isnull(),
        0,
        df["accumulated_depreciation_amortization_ppe"],
    )
    activity_df["average_net_fixed_assets"] = (
        activity_df["net_fixed_assets"] + activity_df["net_fixed_assets"].shift(1)
    ) / 2

    activity_df["average_total_assets"] = (
        df["total_assets"] + df["total_assets"].shift(1)
    ) / 2

    # Calculate activity ratios
    # Measures the efficiency of a company's operations
    activity_df["inventory_turnover"] = (
        df["cost_of_goods_and_services_sold"] / activity_df["average_inventory"]
    )
    activity_df["days_of_inventory_onhand"] = (
        DAYS_IN_QUARTER / activity_df["inventory_turnover"]
    )
    activity_df["receivables_turnover"] = (
        df["total_revenue"] / activity_df["average_receivables"]
    )

    activity_df["days_of_sales_outstanding"] = (
        DAYS_IN_QUARTER / activity_df["receivables_turnover"]
    )

    activity_df["payables_turnover"] = (
        df["cost_of_goods_and_services_sold"] / activity_df["average_payables"]
    )
    activity_df["number_of_days_of_payables"] = (
        DAYS_IN_QUARTER / activity_df["payables_turnover"]
    )
    activity_df["working_capital_turnover"] = (
        df["total_revenue"] / activity_df["average_working_capital"]
    )
    activity_df["fixed_asset_turnover"] = (
        df["total_revenue"] / activity_df["average_net_fixed_assets"]
    )

    activity_df["total_asset_turnover"] = (
        df["total_revenue"] / activity_df["average_total_assets"]
    )

    activity_df["cash_conversion_cycle"] = (
        activity_df["days_of_inventory_onhand"]
        + activity_df["days_of_sales_outstanding"]
        - activity_df["number_of_days_of_payables"]
    )

    # Return a subset of the activity ratios DataFrame
    return activity_df[
        [
            "date",
            "inventory_turnover",
            "days_of_inventory_onhand",
            "receivables_turnover",
            "days_of_sales_outstanding",
            "payables_turnover",
            "number_of_days_of_payables",
            "working_capital_turnover",
            "fixed_asset_turnover",
            "total_asset_turnover",
            "cash_conversion_cycle",
        ]
    ]


def calculate_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate liquidity ratios based on the provided financial statements dataframe.
    The liquidity ratios measure the company's ability to meet its short-term obligations.

    Args:
        df (pd.DataFrame): Combined financial statements dataframe from load_financial_statements()

    Returns:
        pd.DataFrame: Dataframe containing liquidity ratios
    """
    liquidity_df = pd.DataFrame()
    liquidity_df["date"] = df["fiscal_date_ending"]

    # Calculate intermediate values
    liquidity_df["daily_cash_expenditures"] = (
        df["cost_of_goods_and_services_sold"]
        + df["selling_general_and_administrative"]
        + df["research_and_development"]
        - df["depreciation_and_amortization"]
    ) / DAYS_IN_QUARTER

    liquidity_df["current_ratio"] = (
        df["total_current_assets"] / df["total_current_liabilities"]
    )

    liquidity_df["quick_ratio"] = (
        df["cash_and_short_term_investments"] + df["current_net_receivables"]
    ) / df["total_current_liabilities"]

    liquidity_df["cash_ratio"] = (
        df["cash_and_short_term_investments"] / df["total_current_liabilities"]
    )

    liquidity_df["defensive_internal_ratio"] = (
        df["cash_and_short_term_investments"] / liquidity_df["daily_cash_expenditures"]
    )

    return liquidity_df[
        [
            "date",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "defensive_internal_ratio",
        ]
    ]


def calculate_solvency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate solvency ratios based on the provided financial statements dataframe.
    The solvency ratios measures the company's ability to meet its long-term obligations

    Args:
        df (pd.DataFrame): Financial statements dataframe

    Returns:
        pd.DataFrame: Solvency ratios dataframe
    """
    solvency_df = pd.DataFrame()
    solvency_df["date"] = df["fiscal_date_ending"]

    # Calculate average values
    solvency_df["average_total_assets"] = (
        df["total_assets"] + df["total_assets"].shift(1)
    ) / 2

    solvency_df["average_total_equity"] = (
        df["total_shareholder_equity"] + df["total_shareholder_equity"].shift(1)
    ) / 2

    solvency_df["debt_to_assets_ratio"] = (
        df["short_long_term_debt_total"] / df["total_assets"]
    )

    solvency_df["debt_to_capital_ratio"] = df["short_long_term_debt_total"] / (
        df["short_long_term_debt_total"] + df["total_shareholder_equity"]
    )

    solvency_df["debt_to_equity_ratio"] = (
        df["short_long_term_debt_total"] / df["total_shareholder_equity"]
    )

    solvency_df["financial_leverage_ratio"] = (
        solvency_df["average_total_assets"] / solvency_df["average_total_equity"]
    )

    solvency_df["debt_to_ebitda_ratio"] = (
        df["short_long_term_debt_total"] / df["ebitda"]
    )

    return solvency_df[
        [
            "date",
            "debt_to_capital_ratio",
            "debt_to_equity_ratio",
            "financial_leverage_ratio",
            "debt_to_ebitda_ratio",
        ]
    ]


def calculate_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate profitability ratios based on the provided financial statements dataframe.
    The profiabilitiy ratios measures the company's ability to generate profits from
    its resources or sales.

    Ratios calculated:
        - Gross Profit Margin: Measures the amount of gross profit generated for each dollar of revenue
        - Operating Profit Margin: Measures operating efficiency
        - Net Profit Margin: Overall profitability after all expenses
        - Operating Return on Assets: Operating efficiency of assets
        - Return on Assets: Overall return generated by assets
        - Return on Equity: Return generated on shareholders' investment

    Args:
        df (pd.DataFrame): Combined financial statements dataframe from load_financial_statements()

    Returns:
        pd.DataFrame: Dataframe containing profitability ratios
    """
    profitability_df = pd.DataFrame()
    profitability_df["date"] = df["fiscal_date_ending"]

    profitability_df["average_total_assets"] = (
        df["total_assets"] + df["total_assets"].shift(1)
    ) / 2
    profitability_df["average_total_equity"] = (
        df["total_shareholder_equity"] + df["total_shareholder_equity"].shift(1)
    ) / 2

    # Return on Sales
    profitability_df["gross_profit_margin"] = df["gross_profit"] / df["total_revenue"]
    profitability_df["operating_profit_margin"] = (
        df["operating_income"] / df["total_revenue"]
    )

    profitability_df["net_profit_margin"] = df["net_income"] / df["total_revenue"]

    # Return on Investment
    profitability_df["operating_return_on_assets"] = (
        df["operating_income"] / profitability_df["average_total_assets"]
    )
    profitability_df["return_on_assets"] = (
        df["net_income"] / profitability_df["average_total_assets"]
    )

    profitability_df["return_on_equity"] = (
        df["net_income"] / profitability_df["average_total_equity"]
    )

    # Return only the calculation results, filtering out intermediate columns
    return profitability_df[
        [
            "date",
            "gross_profit_margin",
            "operating_profit_margin",
            "net_profit_margin",
            "operating_return_on_assets",
            "return_on_assets",
            "return_on_equity",
        ]
    ]


def fetch_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch other data from the financial statements like deferred revenue and net interest income

    Args:
        df (pd.DataFrame): Combined financial statements dataframe from load_financial_statements()

    Returns:
        pd.DataFrame: Dataframe containing additional metrics
    """
    additional_metrics_df = pd.DataFrame()
    additional_metrics_df["date"] = df["fiscal_date_ending"]
    additional_metrics_df["deferred_revenue"] = df["deferred_revenue"]
    additional_metrics_df["net_interest_income"] = df["net_interest_income"]

    return additional_metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process financial statement data")
    parser.add_argument("symbol", type=str, help="Stock symbol to process")
    args = parser.parse_args()

    symbol = args.symbol.upper()

    # Create output directories if they don't exist
    financial_statements_dir = os.path.join("data", "processed", "financial_statements")
    ratios_dir = os.path.join("data", "processed", "ratios")

    os.makedirs(financial_statements_dir, exist_ok=True)
    os.makedirs(ratios_dir, exist_ok=True)

    print(f"Processing financial data for {symbol}...")

    # Load financial statements
    financial_data = load_financial_statements(symbol)

    # Save combined financial statement as CSV in the financial_statements directory
    financial_statement_path = os.path.join(
        financial_statements_dir, f"{symbol}_financial_statement.csv"
    )
    financial_data.to_csv(financial_statement_path, index=False)
    print(f"Financial statement saved to {financial_statement_path}")

    # Calculate ratios
    print(f"Calculating financial ratios for {symbol}")
    activity_ratios = calculate_activity_ratios(financial_data)
    liquidity_ratios = calculate_liquidity_ratios(financial_data)
    solvency_ratios = calculate_solvency_ratios(financial_data)
    profitability_ratios = calculate_profitability_ratios(financial_data)
    additional_metrics = fetch_additional_metrics(financial_data)

    # Merge the ratio dataframes
    if (
        "date" in activity_ratios.columns
        and "date" in liquidity_ratios.columns
        and "date" in solvency_ratios.columns
        and "date" in profitability_ratios.columns
        and "date" in additional_metrics.columns
    ):
        # Set date as index in each dataframe to avoid duplicating it
        liquidity_ratios = liquidity_ratios.set_index("date")
        solvency_ratios = solvency_ratios.set_index("date")
        profitability_ratios = profitability_ratios.set_index("date")
        additional_metrics = additional_metrics.set_index("date")

        # Merge the dataframes
        ratios = (
            activity_ratios.set_index("date")
            .join(liquidity_ratios)
            .join(solvency_ratios)
            .join(profitability_ratios)
            .join(additional_metrics)
        )
        # Reset index to keep date as a column
        ratios = ratios.reset_index()
    else:
        # If date column not present or has different name, just concat
        ratios = pd.concat(
            [
                activity_ratios,
                liquidity_ratios,
                solvency_ratios,
                profitability_ratios,
                additional_metrics,
            ],
            axis=1,
        )

    # Save ratios as CSV in the ratios directory
    ratios_path = os.path.join(ratios_dir, f"{symbol}_ratios.csv")
    ratios.to_csv(ratios_path, index=False)
    print(f"Financial ratios saved to {ratios_path}")

    # Display sample results
    print("\nSample of financial statement data:")
    print(financial_data.head(3))

    print("\nSample of calculated ratios:")
    print(ratios.head(3))

    print(f"Processing completed successfully for {symbol}")
