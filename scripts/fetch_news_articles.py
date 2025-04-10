import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# NYT API setup
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Define the sectors and their companies
SECTORS = {
    "Automotive": ["Ford Motor Company", "General Motors", "Tesla Inc", "Stellantis", "Toyota Motor"],
    "Technology": ["Meta Platforms", "Netflix", "Apple Inc", "Microsoft", "Amazon"],
    "Finance/Banking": ["JPMorgan Chase", "Goldman Sachs", "Morgan Stanley", "Bank of America", "BlackRock"],
    "AI/Chip Manufacturing": ["NVIDIA", "Advanced Micro Devices", "Intel", "Broadcom", "Taiwan Semiconductor"],
    "Healthcare": ["Johnson & Johnson", "Pfizer", "Moderna", "Baxter International", "Merck & Co."]
}

# Function to get API key
def get_api_key(filename="../utils/nyt_api_key.txt"):
    """
    Reads the NYT API key from a text file.

    Parameters:
        filename (str): Path to the file containing the API key.

    Returns:
        str: The API key as a string.

    Raises:
        SystemExit: If the file is not found.
    """
    try:
        with open(filename, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please create the file and add your API key.")
        exit(1)

# Generate quarterly date ranges from start to end year
def generate_quarters(start_year=2014, end_year=2024):
    """
    Generates quarterly date ranges from the start year to the end year.

    Parameters:
        start_year (int): Starting year.
        end_year (int): Ending year.

    Returns:
        list of tuples: Each tuple contains (start_date, end_date, quarter_label).
    """
    quarters = []
    for year in range(start_year, end_year + 1):
        for q, month in enumerate([1, 4, 7, 10], start=1):  # Q1, Q2, Q3, Q4
            start_date = datetime(year, month, 1).strftime("%Y-%m-%d")
            end_month = month + 2
            end_date = (datetime(year, end_month, 1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
            label = f"{year}_Q{q}"
            quarters.append((start_date, end_date.strftime("%Y-%m-%d"), label))
    return quarters

# Fetch NYT articles
def fetch_nyt_articles(query, start_date, end_date, api_key, max_articles=100):
    """
    Fetches NYT articles for a given query within a date range using the NYT API.

    Parameters:
        query (str): Search term (e.g., company name).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str): NYT API key.
        max_articles (int): Maximum number of articles to fetch.

    Returns:
        list: A list of article metadata dictionaries.
    """
    all_articles = []
    pages = max_articles // 10

    for page in range(pages):
        params = {
            'q': query,
            'begin_date': start_date.replace("-", ""),
            'end_date': end_date.replace("-", ""),
            'api-key': api_key,
            'page': page
        }
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            #articles = response.json().get('response', {}).get('docs', [])
            articles = response.json().get('response', {}).get('docs') or []
            all_articles.extend(articles)
        elif response.status_code == 429:
            print("Rate limit hit. Waiting 60 seconds before retrying...")
            time.sleep(60)
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                articles = response.json().get('response', {}).get('docs', [])
                all_articles.extend(articles)
        else:
            print(f"Error fetching data: {response.status_code}")
            break

    return all_articles

# Collect NYT article data for all companies in all quarters
def collect_nyt_data_all_quarters(sectors, quarters, api_key, data_folder="../data/raw/news_data"):
    """
    Collects NYT article data for each company in all defined sectors and quarters.

    Parameters:
        sectors (dict): Dictionary mapping sectors to lists of company names.
        quarters (list): List of quarterly date tuples as returned by `generate_quarters`.
        api_key (str): NYT API key.
        data_folder (str): Path to save the collected CSV files.

    Saves:
        CSV files for each quarter with article data.
    """
    os.makedirs(data_folder, exist_ok=True)

    for start_date, end_date, label in quarters:
        print(f"\nCollecting data for {start_date} to {end_date}...")
        quarter_data = []

        for sector, companies in sectors.items():
            for company_name in companies:
                print(f"Fetching data for {company_name}...")
                articles = fetch_nyt_articles(company_name, start_date, end_date, api_key)

                for article in articles:
                    data = {
                        'company': company_name,
                        'sector': sector,
                        'headline': article.get('headline', {}).get('main', ''),
                        'snippet': article.get('snippet', ''),
                        'pub_date': article.get('pub_date', ''),
                        'url': article.get('web_url', ''),
                        'quarter': f"{start_date} to {end_date}"
                    }
                    quarter_data.append(data)

        df = pd.DataFrame(quarter_data)
        if df.empty:
            df = pd.DataFrame([{
                'company': 'N/A', 'sector': 'N/A',
                'headline': 'No articles found', 'snippet': '',
                'pub_date': '', 'url': '', 'quarter': f"{start_date} to {end_date}"
            }])

        csv_filename = f"{data_folder}/nyt_articles_{label}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")

# Main execution
if __name__ == "__main__":
    API_KEY = get_api_key()
    quarters = generate_quarters(start_year=2014, end_year=2024)
    collect_nyt_data_all_quarters(SECTORS, quarters, API_KEY)