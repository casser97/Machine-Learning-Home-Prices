import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_clean_data(filepath):
    """
    Load and preprocess the housing price data.
    """
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Handle missing values by filling numerical columns with the median
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Handle missing values in categorical columns (if any)
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna('None')  # Replace missing categories with a placeholder
    
    # Feature Engineering: Add a new feature 'price_per_sqft' if 'price' and 'area' are present
    if 'price' in data.columns and 'area' in data.columns:
        data['price_per_sqft'] = data['price'] / data['area']

    # Feature Engineering: Add a new feature 'house_age' if 'year_built' is present
    if 'year_built' in data.columns:
        data['house_age'] = 2024 - data['year_built']  # Assuming the current year is 2024
    
    # Remove outliers in the target variable 'price'
    upper_bound = data['price'].quantile(0.99)  # Remove top 1% of the data
    data = data[data['price'] <= upper_bound]
    
    # Log transform the target variable 'price'
    data['log_price'] = np.log(data['price'])
    
    # Convert categorical columns to dummy variables (One-Hot Encoding)
    data = pd.get_dummies(data, drop_first=True)

    return data

def split_and_scale_data(data):
    """
    Split the data into train and test sets and scale the features.
    """
    # Split the data into features (X) and the log-transformed target (log_price)
    X = data.drop(['price', 'log_price'], axis=1)
    y = data['log_price']  # Use log-transformed target variable
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
