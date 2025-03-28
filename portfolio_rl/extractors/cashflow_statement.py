import os
from datetime import datetime
from typing import Dict, Any, Optional
from .common import fetch_alpha_vantage_data, safe_float_convert, save_to_file, logger


def fetch_cashflow_statement(symbol: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch cash flow statement data for a specific symbol from Alpha Vantage.

    Args:
        symbol (str): The stock symbol
        api_key (str): Alpha Vantage API key

    Returns:
        Optional[Dict[str, Any]]: Cash flow statement data or None if failed
    """
    result = fetch_alpha_vantage_data("CASH_FLOW", symbol, api_key)
    if isinstance(result, dict):
        return result
    return None


def process_quarterly_cashflow_statement(
    data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Processes quarterly cash flow statement data.

    Args:
        data (Dict[str, Any]): Raw cash flow statement data from Alpha Vantage

    Returns:
        dict: Processed cash flow statement data if successful.
        None otherwise.
    """
    if not data or "quarterlyReports" not in data:
        logger.warning("No data or quarterly reports provided.")
        return None

    # Initialize processed data dictionary
    processed_data = {
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
                # Operating cash flow items
                "operating_cashflow": safe_float_convert(
                    quarter.get("operatingCashflow")
                ),
                "payments_for_operating_activities": safe_float_convert(
                    quarter.get("paymentsForOperatingActivities")
                ),
                "proceeds_from_operating_activities": safe_float_convert(
                    quarter.get("proceedsFromOperatingActivities")
                ),
                "change_in_operating_liabilities": safe_float_convert(
                    quarter.get("changeInOperatingLiabilities")
                ),
                "change_in_operating_assets": safe_float_convert(
                    quarter.get("changeInOperatingAssets")
                ),
                "depreciation_depletion_and_amortization": safe_float_convert(
                    quarter.get("depreciationDepletionAndAmortization")
                ),
                "capital_expenditures": safe_float_convert(
                    quarter.get("capitalExpenditures")
                ),
                "change_in_receivables": safe_float_convert(
                    quarter.get("changeInReceivables")
                ),
                "change_in_inventory": safe_float_convert(
                    quarter.get("changeInInventory")
                ),
                "profit_loss": safe_float_convert(quarter.get("profitLoss")),
                # Investment and financing cash flows
                "cashflow_from_investment": safe_float_convert(
                    quarter.get("cashflowFromInvestment")
                ),
                "cashflow_from_financing": safe_float_convert(
                    quarter.get("cashflowFromFinancing")
                ),
                "proceeds_from_repayments_of_short_term_debt": safe_float_convert(
                    quarter.get("proceedsFromRepaymentsOfShortTermDebt")
                ),
                "payments_for_repurchase_of_common_stock": safe_float_convert(
                    quarter.get("paymentsForRepurchaseOfCommonStock")
                ),
                "payments_for_repurchase_of_equity": safe_float_convert(
                    quarter.get("paymentsForRepurchaseOfEquity")
                ),
                "payments_for_repurchase_of_preferred_stock": safe_float_convert(
                    quarter.get("paymentsForRepurchaseOfPreferredStock")
                ),
                "dividend_payout": safe_float_convert(quarter.get("dividendPayout")),
                "dividend_payout_common_stock": safe_float_convert(
                    quarter.get("dividendPayoutCommonStock")
                ),
                "dividend_payout_preferred_stock": safe_float_convert(
                    quarter.get("dividendPayoutPreferredStock")
                ),
                "proceeds_from_issuance_of_common_stock": safe_float_convert(
                    quarter.get("proceedsFromIssuanceOfCommonStock")
                ),
                "proceeds_from_issuance_of_long_term_debt_and_capital_securities_net": safe_float_convert(
                    quarter.get(
                        "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet"
                    )
                ),
                "proceeds_from_issuance_of_preferred_stock": safe_float_convert(
                    quarter.get("proceedsFromIssuanceOfPreferredStock")
                ),
                "proceeds_from_repurchase_of_equity": safe_float_convert(
                    quarter.get("proceedsFromRepurchaseOfEquity")
                ),
                "proceeds_from_sale_of_treasury_stock": safe_float_convert(
                    quarter.get("proceedsFromSaleOfTreasuryStock")
                ),
                "change_in_cash_and_cash_equivalents": safe_float_convert(
                    quarter.get("changeInCashAndCashEquivalents")
                ),
                "change_in_exchange_rate": safe_float_convert(
                    quarter.get("changeInExchangeRate")
                ),
                "net_income": safe_float_convert(quarter.get("netIncome")),
            }

            quarterly_reports.append(processed_quarter)

        # Sort the quarterly reports by date
        sorted_reports = sorted(
            quarterly_reports,
            key=lambda x: datetime.strptime(x["fiscal_date_ending"], "%Y-%m-%d"),
        )

        # Assign sorted reports to processed_data
        processed_data["quarterly_reports"] = sorted_reports

        # Add metadata
        processed_data["total_quarters"] = len(sorted_reports)
        if sorted_reports:
            processed_data["date_range"] = {
                "start": sorted_reports[0]["fiscal_date_ending"],
                "end": sorted_reports[-1]["fiscal_date_ending"],
            }
        else:
            processed_data["date_range"] = {"start": None, "end": None}
            logger.error("No data found")
            return None
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


def fetch_and_process_cashflow_statement(
    symbol: str,
    api_key: str,
    output_dir: str = "data/raw/cashflow_statements",
) -> Optional[Dict[str, Any]]:
    """
    Fetch and process cash flow statement data for a symbol.

    Args:
        symbol (str): Stock symbol
        api_key (str): Alpha Vantage API key
        output_dir (str): Directory to save the processed data

    Returns:
        Optional[Dict[str, Any]]: Processed cash flow statement data
    """
    # Fetch cash flow statement data
    cashflow_statement = fetch_cashflow_statement(symbol, api_key)

    if not cashflow_statement:
        logger.warning("Failed to fetch cash flow statement for %s", symbol)
        return None

    # Process the data
    processed_cashflow_statement = process_quarterly_cashflow_statement(
        cashflow_statement
    )

    if processed_cashflow_statement:
        # Save to file
        filename = os.path.join(output_dir, f"{symbol}_cashflow_statement.json")
        save_to_file(processed_cashflow_statement, filename)

    return processed_cashflow_statement
