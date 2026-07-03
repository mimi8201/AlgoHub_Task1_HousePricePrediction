"""
Model Evaluation Module.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model_name: str, y_true, y_pred):
    """Calculates standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }

def compare_models(models_dict: dict, X_test, y_test):
    """Evaluates models and determines the best performer based on R2."""
    print("\n--- Starting Model Evaluation ---")
    results = []
    
    for name, model in models_dict.items():
        print(f"[*] Evaluating {name}...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    
    print("\n--- Evaluation Results ---")
    print(results_df.to_string(index=False))
    
    # Identify the winner based on the highest R-Squared Score
    best_model_idx = results_df['R2 Score'].idxmax()
    best_model_row = results_df.iloc[best_model_idx]
    best_model_name = best_model_row['Model']
    
    print(f"\n[*] BEST MODEL: {best_model_name}")
    print(f"    Reason: Highest R² Score ({best_model_row['R2 Score']:.4f}).")
    
    return results_df, best_model_name