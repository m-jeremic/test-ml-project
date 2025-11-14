import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Model Training and Evaluation Functions
def train_linear_regression(X_train, y_train):
    """
    Trains a linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        
    Returns:
        LinearRegression: The trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model
    
    

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns key metrics.
    
    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return metrics



def plot_predictions(y_true, y_pred, figsize=(10, 6)):
    """Plots true values vs. predicted values."""
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True Values vs. Predicted Values')
    plt.show()
    
    