import json

def save_model(model, filename):
    """Save the trained model to a file."""
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    """Load a model from a file."""
    import joblib
    return joblib.load(filename)

def save_metrics(metrics, filename):
    """Save model metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def load_metrics(filename):
    """Load metrics from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

