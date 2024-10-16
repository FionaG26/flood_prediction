import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_model(filepath):
    """Load the trained model from a .pkl file."""
    return joblib.load(filepath)

def preprocess_test_data(df):
    """Preprocess the test data to convert columns to numeric."""
    numeric_columns = ['exposed_(mn)', 'displaced_(k)', 'killed', 'duration(days)']

    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace(',', '').str.strip()
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df.dropna(subset=numeric_columns, inplace=True)

    return df

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2

if __name__ == "__main__":
    model_filepath = '../models/model.pkl'
    test_data_filepath = '../data/processed/cleaned_flood_data.csv'

    # Load the trained model
    model = load_model(model_filepath)

    # Load and preprocess test data
    df_test = pd.read_csv(test_data_filepath)
    df_test = preprocess_test_data(df_test)

    # Print the first few rows to inspect the data
    print("Test DataFrame Head:")
    print(df_test.head())

    # Define features and target variable for test data
    X_test = df_test[['exposed_(mn)', 'displaced_(k)', 'killed', 'duration(days)']]  # Features
    y_test = df_test['displaced_(k)']  # Using displaced_(k) as target variable

    # Ensure y_test is numeric before proceeding
    if not pd.api.types.is_numeric_dtype(y_test):
        raise ValueError("Target variable is not numeric!")

    # Evaluate the model
    mae, mse, r2 = evaluate_model(model, X_test, y_test)

    # Print evaluation metrics
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

