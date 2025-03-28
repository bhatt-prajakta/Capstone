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
        --base_path: Specify data directory (default: 'data')
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
import pandas as pd
import numpy as np
import json


# Constants
DAYS_IN_QUARTER = 90


def load_financial_statements(symbol: str, base_path: str = "data") -> pd.DataFrame:
    """
    Load balance sheet, income statement, and cashflow statement data for a given symbol
    and return a combined dataframe.

    Parameters:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
        base_path (str): Base directory where financial data is stored

    Returns:
        pd.DataFrame: Combined financial statements data for the given symbol
    """
    raw_path = os.path.join(base_path, "raw")

    balance_sheet_path = os.path.join(
        raw_path, f"balance_sheets/{symbol}_balance_sheet.json"
    )
    income_statement_path = os.path.join(
        raw_path, f"income_statements/{symbol}_income_statement.json"
    )
    cashflow_statement_path = os.path.join(
        raw_path, f"cashflow_statements/{symbol}_cashflow_statement.json"
    )

    try:
        # Load JSON data
        with open(balance_sheet_path, "r") as f:
            balance_sheet_data = json.load(f)
        with open(income_statement_path, "r") as f:
            income_statement_data = json.load(f)
        with open(cashflow_statement_path, "r") as f:
            cashflow_statement_data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Financial statement file not found: {e.filename}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in one of the financial statement files")

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
        raise ValueError(f"Missing expected key in financial data: {e}")

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
    Calculate financial ratios based on the provided financial statements dataframe.

    This function currently calculates activity ratios including:
    - Inventory turnover and days of inventory on hand
    - Receivables turnover and days of sales outstanding
    - Payables turnover and number of days of payables
    - Working capital turnover
    - Fixed asset turnover
    - Total asset turnover

    Parameters:
        df (pd.DataFrame): Combined financial statements dataframe from load_financial_statements()

    Returns:
        pd.DataFrame: Dataframe containing calculated financial ratios

    Notes:
        - This function assumes quarterly financial data
        - Average values are calculated using current and previous quarter values
        - First row will contain NaN values for ratios requiring average calculations
    """

    calculated_df = pd.DataFrame()
    calculated_df["date"] = df["fiscal_date_ending"]

    # Calculate average values and intermediate values
    calculated_df["average_inventory"] = (
        df["inventory"] + df["inventory"].shift(1)
    ) / 2
    calculated_df["average_receivables"] = (
        df["current_net_receivables"] + df["current_net_receivables"].shift(1)
    ) / 2
    calculated_df["average_payables"] = (
        df["current_accounts_payable"] + df["current_accounts_payable"].shift(1)
    ) / 2

    calculated_df["working_capital"] = (
        df["total_current_assets"] - df["total_current_liabilities"]
    )
    calculated_df["average_working_capital"] = (
        calculated_df["working_capital"] + calculated_df["working_capital"].shift(1)
    ) / 2

    calculated_df["net_fixed_assets"] = df["property_plant_equipment"] - np.where(
        df["accumulated_depreciation_amortization_ppe"].isnull(),
        0,
        df["accumulated_depreciation_amortization_ppe"],
    )
    calculated_df["average_net_fixed_assets"] = (
        calculated_df["net_fixed_assets"] + calculated_df["net_fixed_assets"].shift(1)
    ) / 2

    calculated_df["average_total_assets"] = (
        df["total_assets"] + df["total_assets"].shift(1)
    ) / 2

    # Calculate activity ratios
    # Measures the efficiency of a company's operations
    calculated_df["inventory_turnover"] = (
        df["cost_of_goods_and_services_sold"] / calculated_df["average_inventory"]
    )
    calculated_df["days_of_inventory_onhand"] = (
        DAYS_IN_QUARTER / calculated_df["inventory_turnover"]
    )
    calculated_df["receivables_turnover"] = (
        df["total_revenue"] / calculated_df["average_receivables"]
    )

    calculated_df["days_of_sales_outstanding"] = (
        DAYS_IN_QUARTER / calculated_df["receivables_turnover"]
    )

    calculated_df["payables_turnover"] = (
        df["cost_of_goods_and_services_sold"] / calculated_df["average_payables"]
    )
    calculated_df["number_of_days_of_payables"] = (
        DAYS_IN_QUARTER / calculated_df["payables_turnover"]
    )
    calculated_df["working_capital_turnover"] = (
        df["total_revenue"] / calculated_df["average_working_capital"]
    )
    calculated_df["fixed_asset_turnover"] = (
        df["total_revenue"] / calculated_df["average_net_fixed_assets"]
    )

    calculated_df["total_asset_turnover"] = (
        df["total_revenue"] / calculated_df["average_total_assets"]
    )

    calculated_df["cash_conversion_cycle"] = (
        calculated_df["days_of_inventory_onhand"]
        + calculated_df["days_of_sales_outstanding"]
        - calculated_df["number_of_days_of_payables"]
    )

    # Return a subset of the calculated DataFrame
    return calculated_df[
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
    The profiabilitiy ratios easures the company's ability to generate profits from its resources or sales.
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
    # profitability_df["pretax_margin"] =
    profitability_df["net_profit_margin"] = df["net_income"] / df["total_revenue"]

    # Return on Investment
    profitability_df["operating_return_on_assets"] = (
        df["operating_income"] / profitability_df["average_total_assets"]
    )
    profitability_df["return_on_assets"] = (
        df["net_income"] / profitability_df["average_total_assets"]
    )
    # profitability_df["return_on_invested_capital"]
    profitability_df["return_on_equity"] = (
        df["net_income"] / profitability_df["average_total_equity"]
    )

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
    # Fetch other data from the financial statements like deferred revenue and net interest income
    #
    additional_metrics_df = pd.DataFrame()
    additional_metrics_df["date"] = df["fiscal_date_ending"]
    additional_metrics_df["deferred_revenue"] = df["deferred_revenue"]
    additional_metrics_df["net_interest_income"] = df["net_interest_income"]

    return additional_metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process financial statement data")
    parser.add_argument("symbol", type=str, help="Stock symbol to process")
    parser.add_argument(
        "--base_path",
        type=str,
        default="data",
        help="Base directory where financial data is stored",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    base_path = args.base_path

    # Create output directories if they don't exist
    financial_statements_dir = os.path.join("data", "processed", "financial_statements")
    ratios_dir = os.path.join("data", "processed", "ratios")

    os.makedirs(financial_statements_dir, exist_ok=True)
    os.makedirs(ratios_dir, exist_ok=True)

    print(f"Processing financial data for {symbol}...")

    # Load financial statements
    df = load_financial_statements(symbol, base_path)

    # Save combined financial statement as CSV in the financial_statements directory
    financial_statement_path = os.path.join(
        financial_statements_dir, f"{symbol}_financial_statement.csv"
    )
    df.to_csv(financial_statement_path, index=False)
    print(f"Financial statement saved to {financial_statement_path}")

    # Calculate ratios
    print(f"Calculating financial ratios for {symbol}")
    activity_ratios = calculate_activity_ratios(df)
    liquidity_ratios = calculate_liquidity_ratios(df)
    solvency_ratios = calculate_solvency_ratios(df)
    profitability_ratios = calculate_profitability_ratios(df)
    additional_metrics = fetch_additional_metrics(df)

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
    print(df.head(3))

    print("\nSample of calculated ratios:")
    print(ratios.head(3))

    print(f"Processing completed successfully for {symbol}")
