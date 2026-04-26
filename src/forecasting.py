import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import time
from src.config import DATA_PROCESSED, OUTPUTS_FIGS, MODELS_DIR

def perform_forecasting():
    print("\n--- PHASE 6: SALES FORECASTING MODELS ---")
    df = pd.read_csv(DATA_PROCESSED / 'encoded_df.csv')
    
    # PART A - Feature Engineering
    target = 'order_revenue'
    features = [
        'purchase_month', 'purchase_year', 'purchase_dow', 'purchase_hour',
        'item_count', 'total_freight', 'category_encoded', 'payment_type_encoded',
        'delivery_days_actual', 'state_encoded'
    ]
    
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    
    # PART B - Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    results = []
    
    # PART C - Linear Regression
    print("Training Linear Regression...")
    start = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    train_time = time.time() - start
    
    results.append({
        'Model': 'Linear Regression',
        'R2 Score': r2_score(y_test, preds_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds_lr)),
        'MAE': mean_absolute_error(y_test, preds_lr),
        'Time': train_time
    })
    
    # PART D - Random Forest
    print("Training Random Forest...")
    start = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    train_time = time.time() - start
    
    results.append({
        'Model': 'Random Forest',
        'R2 Score': r2_score(y_test, preds_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds_rf)),
        'MAE': mean_absolute_error(y_test, preds_rf),
        'Time': train_time
    })
    
    # PART E - XGBoost
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        start = time.time()
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        xgb.fit(X_train, y_train)
        preds_xgb = xgb.predict(X_test)
        train_time = time.time() - start
        
        results.append({
            'Model': 'XGBoost',
            'R2 Score': r2_score(y_test, preds_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds_xgb)),
            'MAE': mean_absolute_error(y_test, preds_xgb),
            'Time': train_time
        })
        joblib.dump(xgb, MODELS_DIR / 'xgboost_model.pkl')
    else:
        print("[WARN] XGBoost not available, skipping...")
    
    # PART F - Comparison
    res_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(res_df)
    
    # PART G - Visualization
    plt.figure(figsize=(12, 6))
    best_preds = preds_xgb if XGBOOST_AVAILABLE else preds_rf
    label = 'Predicted (XGB)' if XGBOOST_AVAILABLE else 'Predicted (RF)'
    plt.plot(y_test.values[:50], label='Actual', marker='o')
    plt.plot(best_preds[:50], label=label, marker='x')
    plt.title('Actual vs Predicted Revenue (First 50 samples)')
    plt.legend()
    plt.savefig(OUTPUTS_FIGS / 'fig_11_actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # PART H - Save Models
    joblib.dump(lr, MODELS_DIR / 'linear_regression.pkl')
    joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
    
    print("[DONE] Phase 6 Complete. Models saved.")

if __name__ == "__main__":
    perform_forecasting()
