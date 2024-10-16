from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save the trained model to a .pkl file."""
    joblib.dump(model, filename)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

# Main script
if __name__ == "__main__":
    # Assuming you have already preprocessed your data
    X_train, X_test, y_train, y_test = preprocess_data('data/processed/flood_data.csv')
    
    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse, y_pred = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

    # Create the models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Specify the name of the pickle file
    filename = 'models/model.pkl'
    
    # Save the model to the specified .pkl file
    save_model(model, filename)

