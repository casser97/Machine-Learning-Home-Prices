import sys
import os
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the project root directory to sys.path to resolve module imports dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess import load_and_clean_data, split_and_scale_data

def train_and_evaluate():
    # Load and preprocess the data
    data = load_and_clean_data('data/house_prices.csv')
    X_train, X_test, y_train, y_test = split_and_scale_data(data)
    
    # Initialize TPOT Regressor
    tpot = TPOTRegressor(verbosity=10, generations=20, population_size=30, random_state=42)
    
    # Fit TPOT to the training data
    tpot.fit(X_train, y_train)
    
    # Predict on the test set using the best model found by TPOT
    y_pred_log = tpot.predict(X_test)
    
    # Reverse log transformation to get the original price scale
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_original, y_pred)
    print(f"Test Set Mean Squared Error: {mse}")
    
    # Visualize Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred, color='blue')
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], '--r', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.show()

    # Residuals Plot
    residuals = y_test_original - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='blue')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    
    # Extract the fitted pipeline from TPOT
    best_pipeline = tpot.fitted_pipeline_

    # Check if the best model supports feature importances (tree-based models like RandomForest)
    if hasattr(best_pipeline.steps[-1][1], 'feature_importances_'):
        importances = best_pipeline.steps[-1][1].feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]  # Update with your actual feature names
        
        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance plot not available for this model.")
    
    # Export the best pipeline found by TPOT
    tpot.export('best_pipeline.py')

if __name__ == '__main__':
    train_and_evaluate()
