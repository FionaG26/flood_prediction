import pandas as pd
import os

def load_data(filepath):
    """Load flood data from a CSV file."""
    return pd.read_csv(filepath, encoding='ISO-8859-1')

def clean_data(df):
    """Clean and preprocess the flood data."""
    # Convert 'flood' column to datetime
    df['flood'] = pd.to_datetime(df['flood'], errors='coerce')  # Ensure errors are handled
    df.dropna(subset=['flood'], inplace=True)  # Drop rows with invalid dates

    # Rename columns to remove whitespace and make them easier to access
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Handle missing values, if any (example: fill with 0 or drop)
    df.fillna(0, inplace=True)  # Replace missing values with 0 (customize as needed)

    return df

def save_cleaned_data(df, filepath):
    """Save cleaned data to a new CSV file."""
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Update file paths to use relative paths
    raw_filepath = '../data/raw/sorted_flood_data.csv'  # Path to raw data
    cleaned_filepath = '../data/processed/cleaned_flood_data.csv'  # Path to save cleaned data

    # Debugging: Check current working directory
    print("Current Working Directory:", os.getcwd())

    df = load_data(raw_filepath)
    cleaned_df = clean_data(df)
    save_cleaned_data(cleaned_df, cleaned_filepath)
    print(f'Cleaned data saved to {cleaned_filepath}')

