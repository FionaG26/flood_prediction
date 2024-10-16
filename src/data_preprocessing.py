import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and dropping irrelevant columns."""
    # Example: Drop irrelevant columns
    df.drop(columns=['cause', 'Areas_Affected'], inplace=True)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables."""
    df['Contributing_factors'] = df['Contributing_factors'].astype('category').cat.codes
    return df

def preprocess_data(file_path):
    """Load and preprocess the data."""
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_categorical_variables(df)
    
    # Split the data into features and target variable
    X = df.drop(columns=['exposed_mn'])
    y = df['exposed_mn']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

