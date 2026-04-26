import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from src.config import DATA_PROCESSED, MODELS_DIR

def clean_data():
    print("\n--- PHASE 3: DATA CLEANING & PREPROCESSING ---")
    df = pd.read_csv(DATA_PROCESSED / 'master_df.csv')
    orig_rows = len(df)
    
    # 1. NULL HANDLING
    print("Handling nulls...")
    df = df[df['order_status'] == 'delivered'].copy()
    print(f"Kept only 'delivered' status. Rows: {len(df)}")
    
    df = df.dropna(subset=['total_payment_value'])
    df['review_score'] = df['review_score'].fillna(df['review_score'].median())
    df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')
    df['delivery_delay'] = df['delivery_delay'].fillna(0)
    
    # 2. DUPLICATE REMOVAL
    print("Removing duplicates...")
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['order_id'])
    print(f"Duplicates removed. Rows: {len(df)}")
    
    # 3. OUTLIER TREATMENT
    print("Treating outliers...")
    def cap_outliers(series):
        Q1 = series.quantile(0.01)
        Q3 = series.quantile(0.99)
        return series.clip(Q1, Q3)

    df['total_payment_value'] = cap_outliers(df['total_payment_value'])
    df = df[df['delivery_days_actual'] <= 120].copy()
    df['item_count'] = cap_outliers(df['item_count'])
    print(f"Outliers treated. Rows: {len(df)}")
    
    # 4. FEATURE ENCODING
    print("Encoding features...")
    le = LabelEncoder()
    df['payment_type_encoded'] = le.fit_transform(df['payment_type'].astype(str))
    df['category_encoded'] = le.fit_transform(df['product_category_name_english'].astype(str))
    df['state_encoded'] = le.fit_transform(df['customer_state'].astype(str))
    
    # 5. FEATURE SCALING
    print("Scaling features...")
    scaling_cols = ['total_payment_value', 'delivery_days_actual', 'review_score', 'item_count']
    scaler = StandardScaler()
    df_scaled_vals = scaler.fit_transform(df[scaling_cols])
    joblib.dump(scaler, MODELS_DIR / 'rfm_scaler.pkl')
    
    # 7. EXPORT
    df.to_csv(DATA_PROCESSED / 'clean_df.csv', index=False)
    encoded_cols = ['payment_type_encoded', 'category_encoded', 'state_encoded']
    df.to_csv(DATA_PROCESSED / 'encoded_df.csv', index=False)
    
    print(f"[DONE] Phase 3 Complete. Final Rows: {len(df)}")
    return df

if __name__ == "__main__":
    clean_data()
