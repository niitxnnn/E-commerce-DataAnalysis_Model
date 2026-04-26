import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_mock_data():
    print("Generating complete mock Olist dataset (9 CSVs)...")
    
    # Create directory structure if not exists
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    np.random.seed(42)
    num_orders = 1000
    num_customers = 800
    num_products = 50
    num_sellers = 20
    
    # 1. Customers
    cust_ids = [f'cust_{i}' for i in range(num_customers)]
    unique_cust_ids = [f'u_cust_{i}' for i in range(num_customers)]
    customers = pd.DataFrame({
        'customer_id': cust_ids,
        'customer_unique_id': unique_cust_ids,
        'customer_zip_code_prefix': np.random.randint(1000, 9999, num_customers),
        'customer_city': np.random.choice(['Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador'], num_customers),
        'customer_state': np.random.choice(['SP', 'RJ', 'MG', 'BA'], num_customers)
    })
    customers.to_csv(f'{raw_dir}/olist_customers_dataset.csv', index=False)
    
    # 2. Orders
    order_ids = [f'ord_{i}' for i in range(num_orders)]
    order_dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(num_orders)]
    orders = pd.DataFrame({
        'order_id': order_ids,
        'customer_id': np.random.choice(cust_ids, num_orders),
        'order_status': 'delivered',
        'order_purchase_timestamp': order_dates,
        'order_approved_at': [d + timedelta(hours=np.random.randint(1, 24)) for d in order_dates],
        'order_delivered_carrier_date': [d + timedelta(days=np.random.randint(1, 3)) for d in order_dates],
        'order_delivered_customer_date': [d + timedelta(days=np.random.randint(2, 15)) for d in order_dates],
        'order_estimated_delivery_date': [d + timedelta(days=10) for d in order_dates]
    })
    orders.to_csv(f'{raw_dir}/olist_orders_dataset.csv', index=False)
    
    # 3. Products
    prod_ids = [f'prod_{i}' for i in range(num_products)]
    categories = ['beleza_saude', 'informatica_acessorios', 'relogios_presentes', 'cama_mesa_banho', 'esporte_lazer']
    products = pd.DataFrame({
        'product_id': prod_ids,
        'product_category_name': np.random.choice(categories, num_products),
        'product_weight_g': np.random.randint(100, 5000, num_products),
        'product_length_cm': np.random.randint(10, 50, num_products),
        'product_height_cm': np.random.randint(5, 30, num_products),
        'product_width_cm': np.random.randint(10, 40, num_products)
    })
    products.to_csv(f'{raw_dir}/olist_products_dataset.csv', index=False)
    
    # 4. Sellers
    seller_ids = [f'sell_{i}' for i in range(num_sellers)]
    sellers = pd.DataFrame({
        'seller_id': seller_ids,
        'seller_zip_code_prefix': np.random.randint(1000, 9999, num_sellers),
        'seller_city': np.random.choice(['Sao Paulo', 'Rio de Janeiro', 'Curitiba'], num_sellers),
        'seller_state': np.random.choice(['SP', 'RJ', 'PR'], num_sellers)
    })
    sellers.to_csv(f'{raw_dir}/olist_sellers_dataset.csv', index=False)
    
    # 5. Order Items
    order_items = pd.DataFrame({
        'order_id': np.random.choice(order_ids, num_orders),
        'order_item_id': 1,
        'product_id': np.random.choice(prod_ids, num_orders),
        'seller_id': np.random.choice(seller_ids, num_orders),
        'price': np.random.uniform(20, 500, num_orders),
        'freight_value': np.random.uniform(5, 50, num_orders),
        'shipping_limit_date': [d + timedelta(days=5) for d in order_dates]
    })
    order_items.to_csv(f'{raw_dir}/olist_order_items_dataset.csv', index=False)
    
    # 6. Order Payments
    payments = pd.DataFrame({
        'order_id': order_ids,
        'payment_sequential': 1,
        'payment_type': np.random.choice(['credit_card', 'boleto', 'voucher', 'debit_card'], num_orders),
        'payment_installments': np.random.randint(1, 10, num_orders),
        'payment_value': order_items.groupby('order_id')['price'].transform('sum') + order_items.groupby('order_id')['freight_value'].transform('sum')
    })
    payments.to_csv(f'{raw_dir}/olist_order_payments_dataset.csv', index=False)
    
    # 7. Order Reviews
    reviews = pd.DataFrame({
        'review_id': [f'rev_{i}' for i in range(num_orders)],
        'order_id': order_ids,
        'review_score': np.random.randint(1, 6, num_orders).astype(float),
        'review_comment_title': 'N/A',
        'review_comment_message': 'N/A',
        'review_creation_date': order_dates,
        'review_answer_timestamp': [d + timedelta(days=2) for d in order_dates]
    })
    reviews.to_csv(f'{raw_dir}/olist_order_reviews_dataset.csv', index=False)
    
    # 8. Geolocation
    zip_codes = np.unique(np.concatenate([customers['customer_zip_code_prefix'], sellers['seller_zip_code_prefix']]))
    geo = pd.DataFrame({
        'geolocation_zip_code_prefix': zip_codes,
        'geolocation_lat': np.random.uniform(-23, -22, len(zip_codes)),
        'geolocation_lng': np.random.uniform(-47, -46, len(zip_codes)),
        'geolocation_city': 'Unknown',
        'geolocation_state': 'Unknown'
    })
    geo.to_csv(f'{raw_dir}/olist_geolocation_dataset.csv', index=False)
    
    # 9. Translation
    trans = pd.DataFrame({
        'product_category_name': categories,
        'product_category_name_english': ['health_beauty', 'computers_accessories', 'watches_gifts', 'bed_bath_table', 'sports_leisure']
    })
    trans.to_csv(f'{raw_dir}/product_category_name_translation.csv', index=False)
    
    print(f"✅ All 9 CSVs generated successfully in '{raw_dir}'.")

if __name__ == "__main__":
    generate_mock_data()
