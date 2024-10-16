import pandas as pd
import os

def load_data(filepath):
    """Load flood data from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess the flood data."""
    # Convert 'flood' column to datetime
    df['flood'] = pd.to_datetime(df['flood'], errors='coerce')  # Ensure errors are handled
    df.dropna(subset=['flood'], inplace=True)  # Drop rows with invalid dates

    # Rename columns to remove whitespace and make them easier to access
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Remove commas from numeric columns and convert them to numeric types
    numeric_columns = ['exposed_(mn)', 'displaced_(k)', 'killed', 'duration(days)']
    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace(',', '').str.strip()
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Drop rows with NaN values in these columns
    df.dropna(subset=numeric_columns, inplace=True)

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

