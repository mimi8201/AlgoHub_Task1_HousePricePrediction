"""
Data Preparation and Model Training Module.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def prepare_data(input_path: str):
    """Loads data, selects specific features, and splits it for training and testing."""
    print("--- Starting Data Preparation ---")
    
    if not os.path.exists(input_path):
        print(f"[Error] '{input_path}' not found.")
        return None, None, None, None

    df = pd.read_csv(input_path)

    # Define features and target variable
    selected_features = ['beds', 'baths', 'size', 'zip_code_freq', 'beds_baths_total']
    target = 'price'

    X = df[selected_features]
    y = df[target]
    
    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Trains a Linear Regression model and a Random Forest Regressor.
    
    Args:
        X_train (DataFrame): The training input features.
        y_train (Series): The training target values (prices).
        
    Returns:
        tuple: (trained_linear_model, trained_random_forest_model)
    """
    print("\n--- Starting Model Training ---")
    
    # 1. Train Linear Regression
    print("[*] Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("    -> Linear Regression training complete.")

    # 2. Train Random Forest Regressor
    # n_estimators=100 creates 100 distinct decision trees
    print("[*] Training Random Forest Regressor (100 trees)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("    -> Random Forest training complete.")
    
    return lr_model, rf_model

if __name__ == "__main__":
    from evaluate import compare_models
    import joblib
    import json
    
    input_file = 'data/house_data_features.csv'
    
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(input_file)
    
    if X_train is not None:
        # Train the models
        lr_model, rf_model = train_models(X_train, y_train)
        
        trained_models = {
            "Linear Regression": lr_model,
            "Random Forest Regressor": rf_model
        }
        
        # Evaluate models and get the winner
        results_table, best_model_name = compare_models(trained_models, X_test, y_test)
        
        # --- Artifact Serialization ---
        print("\n--- Saving Project Artifacts ---")
        os.makedirs('models', exist_ok=True)
        
        # 1. Save the best model
        best_model = trained_models[best_model_name]
        joblib.dump(best_model, 'models/model.pkl')
        print(f"[*] Saved winning model ({best_model_name}) to 'models/model.pkl'")
        
        # 2. Save the metrics
        metrics_dict = results_table.to_dict(orient='records')
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print("[*] Saved evaluation metrics to 'models/metrics.json'")