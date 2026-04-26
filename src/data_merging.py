import pandas as pd
import numpy as np
import os
from src.config import DATA_RAW, DATA_PROCESSED

def load_and_merge():
    print("--- PHASE 2: DATA LOADING & MERGING ---")
    
    # 1. Load tables
    print("Loading tables...")
    orders = pd.read_csv(DATA_RAW / 'olist_orders_dataset.csv')
    customers = pd.read_csv(DATA_RAW / 'olist_customers_dataset.csv')
    items = pd.read_csv(DATA_RAW / 'olist_order_items_dataset.csv')
    payments = pd.read_csv(DATA_RAW / 'olist_order_payments_dataset.csv')
    reviews = pd.read_csv(DATA_RAW / 'olist_order_reviews_dataset.csv')
    products = pd.read_csv(DATA_RAW / 'olist_products_dataset.csv')
    sellers = pd.read_csv(DATA_RAW / 'olist_sellers_dataset.csv')
    geo = pd.read_csv(DATA_RAW / 'olist_geolocation_dataset.csv')
    translation = pd.read_csv(DATA_RAW / 'product_category_name_translation.csv')
    
    print(f"Total orders: {len(orders)}")
    
    # 1. Aggregate payments
    print("Aggregating payments...")
    def get_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    payments_agg = payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max',
        'payment_type': get_mode
    }).rename(columns={'payment_value': 'total_payment_value'}).reset_index()
    
    # 2. Deduplicate reviews
    print("Deduplicating reviews...")
    reviews = reviews.sort_values('review_answer_timestamp', ascending=False).drop_duplicates('order_id')
    
    # 3. Translate categories
    print("Translating categories...")
    products = products.merge(translation, on='product_category_name', how='left')
    products['product_category_name_english'] = products['product_category_name_english'].fillna('unknown')
    
    # 4. Aggregate items
    print("Aggregating items...")
    items_agg = items.groupby('order_id').agg({
        'order_item_id': 'count',
        'price': 'sum',
        'freight_value': 'sum',
        'product_id': get_mode
    }).rename(columns={
        'order_item_id': 'item_count',
        'price': 'total_item_price',
        'freight_value': 'total_freight'
    }).reset_index()
    
    # Merge items with products to get category
    items_agg = items_agg.merge(products[['product_id', 'product_category_name_english']], on='product_id', how='left')
    
    # 5. Geolocation
    print("Deduplicating geolocation...")
    geo_clean = geo.drop_duplicates('geolocation_zip_code_prefix')
    customers = customers.merge(geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    
    # 6. Final Merge
    print("Performing final merge...")
    master_df = orders.merge(customers, on='customer_id', how='left')
    master_df = master_df.merge(items_agg, on='order_id', how='left')
    master_df = master_df.merge(payments_agg, on='order_id', how='left')
    master_df = master_df.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
    
    # 7. Derived Columns
    print("Adding derived columns...")
    master_df['order_revenue'] = master_df['total_item_price'] + master_df['total_freight']
    
    time_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for col in time_cols:
        master_df[col] = pd.to_datetime(master_df[col])
        
    master_df['purchase_month'] = master_df['order_purchase_timestamp'].dt.month
    master_df['purchase_year'] = master_df['order_purchase_timestamp'].dt.year
    master_df['purchase_dow'] = master_df['order_purchase_timestamp'].dt.dayofweek
    master_df['purchase_hour'] = master_df['order_purchase_timestamp'].dt.hour
    
    master_df['delivery_days_actual'] = (master_df['order_delivered_customer_date'] - master_df['order_purchase_timestamp']).dt.days
    master_df['delivery_delay'] = (master_df['order_delivered_customer_date'] - master_df['order_estimated_delivery_date']).dt.days
    
    # Export
    master_df.to_csv(DATA_PROCESSED / 'master_df.csv', index=False)
    print(f"[DONE] Phase 2 Complete. Master DF shape: {master_df.shape}")
    return master_df

if __name__ == "__main__":
    load_and_merge()
