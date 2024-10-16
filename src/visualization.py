import matplotlib.pyplot as plt
import seaborn as sns

def plot_exposed_over_time(df):
    """Plot the exposed population over time."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='exposed_mn', data=df)
    plt.title('Exposed Population Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposed Population (mn)')
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_matrix(df):
    """Plot the correlation matrix."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_predictions(y_test, y_pred):
    """Visualize true vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect predictions
    plt.show()

