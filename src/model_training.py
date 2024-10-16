import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def load_cleaned_data(filepath):
    """Load cleaned data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data to convert columns to numeric."""
    
    # Clean and convert numeric columns again (if necessary)
    numeric_columns = ['exposed_(mn)', 'displaced_(k)', 'killed', 'duration(days)']
    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace(',', '').str.strip()
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with NaN values in these columns
    df.dropna(subset=numeric_columns, inplace=True)

    return df

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save the trained model to a .pkl file."""
    joblib.dump(model, filename)

if __name__ == "__main__":
    cleaned_filepath = '../data/processed/cleaned_flood_data.csv'
    
    # Load the cleaned data
    df = load_cleaned_data(cleaned_filepath)

    # Preprocess the data
    df = preprocess_data(df)

    # Prepare the features and target variable
    X = df[['exposed_(mn)', 'displaced_(k)', 'killed', 'duration(days)']]
    y = df['displaced_(k)']  # Using displaced_(k) as target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Specify the name of the pickle file
    filename = '../models/model.pkl'  # Adjust the path if necessary

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the model
    save_model(model, filename)

    print(f"Model saved to {filename}")

