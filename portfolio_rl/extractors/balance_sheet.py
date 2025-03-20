import os
from datetime import datetime
from typing import Dict, Any, Optional
from .common import fetch_alpha_vantage_data, safe_float_convert, save_to_file, logger


def fetch_balance_sheet(symbol: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch balance sheet data for a specific symbol from Alpha Vantage.

    Args:
        symbol (str): The stock symbol
        api_key (str): Alpha Vantage API key

    Returns:
        Optional[Dict[str, Any]]: Balance sheet data or None if failed
    """
    return fetch_alpha_vantage_data("BALANCE_SHEET", symbol, api_key)


def process_quarterly_balance_sheet(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes quarterly (10Q) balance sheet data.

    Args:
        data (Dict[str, Any]): Raw balance sheet data from Alpha Vantage

    Returns:
        dict: Processed balance sheet data if successful.
        None otherwise.
    """
    if not data or "quarterlyReports" not in data:
        logger.warning("No data or quarterly reports provided.")
        return None

    # Initialize processed data dictionary
    processed_data: Dict[str, Any] = {
        "symbol": data.get("symbol", "Unknown"),
        "quarterly_reports": [],
    }

    try:
        # Process all quarterly reports
        quarterly_reports = []

        for quarter in data["quarterlyReports"]:
            fiscal_date = quarter["fiscalDateEnding"]

            # Extract relevant data from each quarter
            processed_quarter = {
                "fiscal_date_ending": fiscal_date,
                "reported_currency": quarter["reportedCurrency"],
                # Assets
                "total_assets": safe_float_convert(quarter.get("totalAssets")),
                "total_current_assets": safe_float_convert(
                    quarter.get("totalCurrentAssets")
                ),
                "cash_and_cash_equivalents": safe_float_convert(
                    quarter.get("cashAndCashEquivalentsAtCarryingValue")
                ),
                "cash_and_short_term_investments": safe_float_convert(
                    quarter.get("cashAndShortTermInvestments")
                ),
                "inventory": safe_float_convert(quarter.get("inventory")),
                "current_net_receivables": safe_float_convert(
                    quarter.get("currentNetReceivables")
                ),
                "total_non_current_assets": safe_float_convert(
                    quarter.get("totalNonCurrentAssets")
                ),
                "property_plant_equipment": safe_float_convert(
                    quarter.get("propertyPlantEquipment")
                ),
                "accumulated_depreciation_amortization_ppe": safe_float_convert(
                    quarter.get("accumulatedDepreciationAmortizationPPE")
                ),
                "intangible_assets": safe_float_convert(
                    quarter.get("intangibleAssets")
                ),
                "intangible_assets_excluding_goodwill": safe_float_convert(
                    quarter.get("intangibleAssetsExcludingGoodwill")
                ),
                "goodwill": safe_float_convert(quarter.get("goodwill")),
                "investments": safe_float_convert(quarter.get("investments")),
                "long_term_investments": safe_float_convert(
                    quarter.get("longTermInvestments")
                ),
                "short_term_investments": safe_float_convert(
                    quarter.get("shortTermInvestments")
                ),
                "other_current_assets": safe_float_convert(
                    quarter.get("otherCurrentAssets")
                ),
                "other_non_current_assets": safe_float_convert(
                    quarter.get("otherNonCurrentAssets")
                ),
                # Liabilities
                "total_liabilities": safe_float_convert(
                    quarter.get("totalLiabilities")
                ),
                "total_current_liabilities": safe_float_convert(
                    quarter.get("totalCurrentLiabilities")
                ),
                "current_accounts_payable": safe_float_convert(
                    quarter.get("currentAccountsPayable")
                ),
                "deferred_revenue": safe_float_convert(quarter.get("deferredRevenue")),
                "current_debt": safe_float_convert(quarter.get("currentDebt")),
                "short_term_debt": safe_float_convert(quarter.get("shortTermDebt")),
                "total_non_current_liabilities": safe_float_convert(
                    quarter.get("totalNonCurrentLiabilities")
                ),
                "capital_lease_obligations": safe_float_convert(
                    quarter.get("capitalLeaseObligations")
                ),
                "long_term_debt": safe_float_convert(quarter.get("longTermDebt")),
                "current_long_term_debt": safe_float_convert(
                    quarter.get("currentLongTermDebt")
                ),
                "long_term_debt_non_current": safe_float_convert(
                    quarter.get("longTermDebtNoncurrent")
                ),
                "short_long_term_debt_total": safe_float_convert(
                    quarter.get("shortLongTermDebtTotal")
                ),
                "other_current_liabilities": safe_float_convert(
                    quarter.get("otherCurrentLiabilities")
                ),
                "other_non_current_liabilities": safe_float_convert(
                    quarter.get("otherNonCurrentLiabilities")
                ),
                # Equity
                "total_shareholder_equity": safe_float_convert(
                    quarter.get("totalShareholderEquity")
                ),
                "treasury_stock": safe_float_convert(quarter.get("treasuryStock")),
                "retained_earnings": safe_float_convert(
                    quarter.get("retainedEarnings")
                ),
                "common_stock": safe_float_convert(quarter.get("commonStock")),
                "common_stock_shares_outstanding": safe_float_convert(
                    quarter.get("commonStockSharesOutstanding")
                ),
            }

            quarterly_reports.append(processed_quarter)

        # After processing all quarters, sort chronologically (oldest first)
        sorted_reports = sorted(
            quarterly_reports,
            key=lambda x: datetime.strptime(x["fiscal_date_ending"], "%Y-%m-%d"),
        )

        # Assign sorted reports to processed_data
        processed_data["quarterly_reports"] = sorted_reports

        # Add metadata
        processed_data["total_quarters"] = len(processed_data["quarterly_reports"])
        if processed_data["quarterly_reports"]:
            processed_data["date_range"] = {
                "start": processed_data["quarterly_reports"][0]["fiscal_date_ending"],
                "end": processed_data["quarterly_reports"][-1]["fiscal_date_ending"],
            }
        else:
            processed_data["date_range"] = {"start": None, "end": None}
    except KeyError as e:
        logger.error("Missing key in data: %s", str(e))
        return None
    except ValueError as e:
        logger.error("Error converting value: %s", str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error processing balance sheet: %s", str(e))
        return None

    return processed_data


def fetch_and_process_balance_sheet(
    symbol: str,
    api_key: str,
    output_dir: str = "data/balance_sheets",
) -> Optional[Dict[str, Any]]:
    """
    Fetch and process balance sheet data for a symbol.

    Args:
        symbol (str): Stock symbol
        api_key (str): Alpha Vantage API key
        output_dir (str): Directory to save the processed data

    Returns:
        Optional[Dict[str, Any]]: Processed balance sheet data
    """
    # Fetch balance sheet data
    balance_sheet = fetch_balance_sheet(symbol, api_key)

    if not balance_sheet:
        logger.warning("Failed to fetch balance sheet for %s", symbol)
        return None

    # Process the data
    processed_balance_sheet = process_quarterly_balance_sheet(balance_sheet)

    if processed_balance_sheet:
        # Save to file
        filename = os.path.join(output_dir, f"{symbol}_balance_sheet.json")
        save_to_file(processed_balance_sheet, filename)

    return processed_balance_sheet
