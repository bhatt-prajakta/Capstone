"""
Sentiment Analysis Pipeline using FinBERT for financial news articles.

This script:
- Loads quarterly CSVs of news data.
- Cleans and processes article text.
- Applies FinBERT to compute sentiment scores.
- Derives article-level features like relative sentiment, momentum, dispersion.
- Aggregates metrics by sector per quarter.
- Saves quarterly sector-level metrics to a processed CSV file.

Requirements:
- transformers
- torch
- pandas
- numpy
- glob
- re

Output:
- data/processed/sector_sentiment_with_metrics_quarterly_2014_2024.csv
"""
import pandas as pd
import glob
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Initialize FinBERT Sentiment Analyzer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Preprocessing Function
def preprocess_text(text):
    """
    Cleans and preprocesses a text string by:
    - Lowercasing
    - Removing URLs, HTML tags, special characters, and extra whitespace
    
    Args:
        text (str): Raw text snippet.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Get FinBERT Sentiment Score
def get_finbert_sentiment(text):
    """
    Applies FinBERT model to classify the sentiment of input text.

    Args:
        text (str): Cleaned text string.

    Returns:
        int: Sentiment score (0=Negative, 1=Neutral, 2=Positive).
    """
    if text:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment = torch.argmax(outputs.logits).item()  # 0: Negative, 1: Neutral, 2: Positive
        return sentiment
    return 1  # Default to Neutral

# Calculate Relative Sentiment (Article Level)
def calculate_relative_sentiment(df):
    """
    Calculates relative sentiment by subtracting mean sentiment from each article.

    Args:
        df (DataFrame): DataFrame with 'finbert_sentiment' column.

    Returns:
        DataFrame: Updated with 'relative_sentiment' column.
    """
    df['relative_sentiment'] = df['finbert_sentiment'] - df['finbert_sentiment'].mean()
    return df

# Calculate Momentum (Article Level)
def calculate_momentum(df):
    """
    Calculates sentiment momentum using first-order difference.

    Args:
        df (DataFrame): DataFrame with 'finbert_sentiment' column.

    Returns:
        DataFrame: Updated with 'momentum' column.
    """
    df['momentum'] = df['finbert_sentiment'].diff()
    return df

# Calculate Dispersion (Article Level)
def calculate_dispersion(df, window=5):
    """
    Calculates dispersion (volatility) in sentiment using rolling standard deviation.

    Args:
        df (DataFrame): DataFrame with 'finbert_sentiment' column.
        window (int): Rolling window size.

    Returns:
        DataFrame: Updated with 'dispersion' column.
    """
    df['dispersion'] = df['finbert_sentiment'].rolling(window=window, min_periods=1).std()
    return df

# Normalize Sentiment Scores (Article Level)
def normalize_sentiment(df):
    """
    Normalizes sentiment scores to [0, 1] range.

    Args:
        df (DataFrame): DataFrame with 'finbert_sentiment' column.

    Returns:
        DataFrame: Updated with 'normalized_sentiment' column.
    """
    min_sentiment = df['finbert_sentiment'].min()
    max_sentiment = df['finbert_sentiment'].max()
    df['normalized_sentiment'] = (df['finbert_sentiment'] - min_sentiment) / (max_sentiment - min_sentiment)
    return df

def analyze_sector_properly_weighted_sentiment(df):
    """
    Aggregates sentiment-related metrics at the sector level with proper weighting.

    Args:
        df (DataFrame): Article-level DataFrame with sector and sentiment metrics.

    Returns:
        DataFrame: Sector-level aggregated sentiment metrics.
    """
    if df.empty or 'sector' not in df.columns or 'finbert_sentiment' not in df.columns:
        return pd.DataFrame()

    # Count articles per sector
    sector_counts = df.groupby('sector')['finbert_sentiment'].count().reset_index()
    sector_counts.rename(columns={'finbert_sentiment': 'article_count'}, inplace=True)

    # Compute sector-wise aggregated metrics
    sector_sentiment = df.groupby('sector').agg(
        weighted_sentiment=('finbert_sentiment', lambda x: (x * len(x)).sum() / df.shape[0]),
        avg_relative_sentiment=('relative_sentiment', 'mean'),
        avg_momentum=('momentum', 'mean'),
        sentiment_dispersion=('dispersion', 'mean'),
        avg_normalized_sentiment=('normalized_sentiment', 'mean')
    ).reset_index()

    return sector_sentiment

# Load and Process All CSV Files
def load_and_process_data(file_paths):
    """
    Loads and processes all article CSV files and computes sentiment metrics.

    Args:
        file_paths (list): List of CSV file paths.

    Returns:
        dict: Dictionary with quarter as key and sector-level DataFrame as value.
    """
    all_results = {}

    for file_path in file_paths:
        match = re.search(r'(\d{4})_Q([1-4])', file_path)
        if not match:
            continue  # Skip files with invalid format
        year, quarter = match.groups()
        quarter_index = f"{year}_Q{quarter}"  

        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        if 'sector' not in df.columns:
            print(f"Skipping {file_path} - 'sector' column missing")
            continue

        # Apply text preprocessing and sentiment analysis
        df['snippet'] = df['snippet'].fillna('').apply(preprocess_text)
        df['finbert_sentiment'] = df['snippet'].apply(get_finbert_sentiment)

        # Apply sentiment-based transformations BEFORE grouping by sector
        df = calculate_relative_sentiment(df)
        df = calculate_momentum(df)
        df = calculate_dispersion(df)
        df = normalize_sentiment(df)

        # Compute sector-wise weighted sentiment and additional metrics
        sector_weighted_sentiment = analyze_sector_properly_weighted_sentiment(df)

        if not sector_weighted_sentiment.empty:
            all_results[quarter_index] = sector_weighted_sentiment

    return all_results

# Combine Results and Save to CSV
def save_results(all_results):
    """
    Combines all sector-level results and saves to a CSV file.

    Args:
        all_results (dict): Dictionary of DataFrames indexed by quarter.
    """
    # Convert dictionary of DataFrames into one DataFrame
    final_df = pd.concat(all_results, names=['quarter'], axis=0).reset_index()
    final_df.rename(columns={"level_0": "quarter"}, inplace=True)
    final_df = final_df.pivot(index="quarter", columns="sector")

    # Flatten multi-index column names 
    final_df.columns = ['_'.join(col).strip() for col in final_df.columns.values]
    final_df = final_df.reindex(sorted(final_df.index, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1][1]))))
    final_df = final_df.round(4)

    # Save to CSV
    final_df.to_csv('../data/processed/sector_sentiment_with_metrics_quarterly_2014_2024.csv')

    print("Sentiment scores with metrics per sector saved to 'sector_sentiment_with_metrics_quarterly_2014_2024.csv'.")

# Main function
def main():
    file_paths = glob.glob('../data/raw/news_data/nyt_articles_*.csv')
    all_results = load_and_process_data(file_paths)
    save_results(all_results)

if __name__ == "__main__":
    main()