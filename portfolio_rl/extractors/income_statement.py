import os
from datetime import datetime
from typing import Optional, Dict, Any
from .common import fetch_alpha_vantage_data, safe_float_convert, save_to_file, logger


def fetch_income_statement(symbol: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch income statement data for a specific symbol from Alpha Vantage.

    Args:
        symbol (str): The stock symbol
        api_key (str): Alpha Vantage API key

    Returns:
        Optional[Dict[str, Any]]: Income statement data or None if failed
    """
    return fetch_alpha_vantage_data("INCOME_STATEMENT", symbol, api_key)


def process_quarterly_income_statement(
    data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Process quarterly income statement data.

    Args:
        data (Dict[str, Any]): Raw income statement data from Alpha Vantage

    Returns:
        Optional[Dict[str, Any]]: Processed income statement data
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
        quarterly_reports = []

        for quarter in data["quarterlyReports"]:
            fiscal_date = quarter["fiscalDateEnding"]
            # Extract relevant data from each quarter
            processed_quarter = {
                "fiscal_date_ending": fiscal_date,
                "reported_currency": quarter["reportedCurrency"],
                # Revenue and profit
                "gross_profit": safe_float_convert(quarter.get("grossProfit")),
                "total_revenue": safe_float_convert(quarter.get("totalRevenue")),
                "cost_of_revenue": safe_float_convert(quarter.get("costOfRevenue")),
                "cost_of_goods_and_services_sold": safe_float_convert(
                    quarter.get("costofGoodsAndServicesSold")
                ),
                # Operating metrics
                "operating_income": safe_float_convert(quarter.get("operatingIncome")),
                "selling_general_and_administrative": safe_float_convert(
                    quarter.get("sellingGeneralAndAdministrative")
                ),
                "research_and_development": safe_float_convert(
                    quarter.get("researchAndDevelopment")
                ),
                "operating_expenses": safe_float_convert(
                    quarter.get("operatingExpenses")
                ),
                "investment_income_net": safe_float_convert(
                    quarter.get("investmentIncomeNet")
                ),
                "net_interest_income": safe_float_convert(
                    quarter.get("netInterestIncome")
                ),
                "interest_income": safe_float_convert(quarter.get("interestIncome")),
                "interest_expense": safe_float_convert(quarter.get("interestExpense")),
                "non_interest_income": safe_float_convert(
                    quarter.get("nonInterestIncome")
                ),
                "other_non_operating_income": safe_float_convert(
                    quarter.get("otherNonOperatingIncome")
                ),
                "depreciation": safe_float_convert(quarter.get("depreciation")),
                "depreciation_and_amortization": safe_float_convert(
                    quarter.get("depreciationAndAmortization")
                ),
                # Income and tax
                "income_before_tax": safe_float_convert(quarter.get("incomeBeforeTax")),
                "income_tax_expense": safe_float_convert(
                    quarter.get("incomeTaxExpense")
                ),
                "interest_and_debt_expense": safe_float_convert(
                    quarter.get("interestAndDebtExpense")
                ),
                "net_income_from_continuing_operations": safe_float_convert(
                    quarter.get("netIncomeFromContinuingOperations")
                ),
                "comprehensive_income_net_of_tax": safe_float_convert(
                    quarter.get("comprehensiveIncomeNetOfTax")
                ),
                # EBIT and EBITDA
                "ebit": safe_float_convert(quarter.get("ebit")),
                "ebitda": safe_float_convert(quarter.get("ebitda")),
                "net_income": safe_float_convert(quarter.get("netIncome")),
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
                    "start": processed_data["quarterly_reports"][0][
                        "fiscal_date_ending"
                    ],
                    "end": processed_data["quarterly_reports"][-1][
                        "fiscal_date_ending"
                    ],
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
        logger.error("Unexpected error processing income statement: %s", str(e))
        return None

    return processed_data


def fetch_and_process_income_statement(
    symbol: str,
    api_key: str,
    output_dir: str = "data/income_statements",
) -> Optional[Dict[str, Any]]:
    """
    Fetch and process income statement data for a symbol.

    Args:
        symbol (str): Stock symbol
        api_key (str): Alpha Vantage API key
        output_dir (str): Directory to save the processed data

    Returns:
        Optional[Dict[str, Any]]: Processed income statement data
    """
    # Fetch income statement data
    income_statement = fetch_income_statement(symbol, api_key)

    if not income_statement:
        logger.warning("Failed to fetch income statement for %s", symbol)
        return None

    # Process the data
    processed_income_statement = process_quarterly_income_statement(income_statement)

    if processed_income_statement:
        # Save to file
        filename = os.path.join(output_dir, f"{symbol}_income_statement.json")
        save_to_file(processed_income_statement, filename)

    return processed_income_statement
