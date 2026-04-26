import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_PROCESSED, OUTPUTS_FIGS

def perform_eda():
    print("\n--- PHASE 4: EXPLORATORY DATA ANALYSIS (EDA) ---")
    df = pd.read_csv(DATA_PROCESSED / 'clean_df.csv')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    brand_colors = ["#FF3E6C", "#282C3F"] # Myntra colors
    
    # 1. Monthly Revenue Trend
    print("Generating Chart 1...")
    plt.figure(figsize=(12, 6))
    monthly_rev = df.groupby(['purchase_year', 'purchase_month'])['order_revenue'].sum().reset_index()
    monthly_rev['label'] = monthly_rev['purchase_year'].astype(str) + '-' + monthly_rev['purchase_month'].astype(str)
    sns.lineplot(data=monthly_rev, x='label', y='order_revenue', marker='o', color=brand_colors[0])
    plt.title('Monthly Revenue Trend', fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUTS_FIGS / 'fig_01_monthly_revenue.png', dpi=150)
    plt.close()

    # 2. Category Revenue Bar
    print("Generating Chart 2...")
    plt.figure(figsize=(12, 6))
    cat_rev = df.groupby('product_category_name_english')['order_revenue'].sum().sort_values(ascending=False).head(15)
    cat_rev.plot(kind='barh', color=sns.color_palette("rocket", 15))
    plt.title('Top 15 Categories by Revenue', fontsize=15)
    plt.xlabel('Revenue')
    plt.tight_layout()
    plt.savefig(OUTPUTS_FIGS / 'fig_02_category_revenue.png', dpi=150)
    plt.close()

    # 3. Payment Methods
    print("Generating Chart 3...")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=sns.color_palette("pastel"))
        df['payment_type'].value_counts().plot.bar(ax=ax2, color=brand_colors[1])
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGS / 'fig_03_payment_methods.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in Chart 3: {e}")

    # 4. Delivery Delay vs Review Score
    print("Generating Chart 4...")
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='delivery_delay', y='review_score', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Delivery Delay vs Review Score', fontsize=15)
    plt.savefig(OUTPUTS_FIGS / 'fig_04_delivery_vs_rating.png', dpi=150)
    plt.close()

    # 5. Seasonal Heatmap
    print("Generating Chart 5...")
    plt.figure(figsize=(12, 6))
    ct = pd.crosstab(df['purchase_month'], df['purchase_dow'])
    sns.heatmap(ct, cmap="YlGnBu", annot=True, fmt='d')
    plt.title('Order Volume by Month and Day of Week', fontsize=15)
    plt.savefig(OUTPUTS_FIGS / 'fig_05_order_heatmap.png', dpi=150)
    plt.close()

    # 6. Review Score Distribution
    print("Generating Chart 6...")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['review_score'], kde=True, color=brand_colors[0])
    plt.axvline(df['review_score'].mean(), color='blue', linestyle='--')
    plt.title('Review Score Distribution', fontsize=15)
    plt.savefig(OUTPUTS_FIGS / 'fig_06_review_distribution.png', dpi=150)
    plt.close()

    # 7. Top States
    print("Generating Chart 7...")
    plt.figure(figsize=(12, 6))
    df.groupby('customer_state')['order_revenue'].sum().sort_values(ascending=False).plot(kind='bar', color=brand_colors[1])
    plt.title('Revenue by State', fontsize=15)
    plt.savefig(OUTPUTS_FIGS / 'fig_07_state_revenue.png', dpi=150)
    plt.close()

    # 8. Correlation Heatmap
    print("Generating Chart 8...")
    plt.figure(figsize=(10, 8))
    cols = ['total_payment_value', 'item_count', 'delivery_days_actual', 'review_score', 'total_freight', 'delivery_delay']
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap', fontsize=15)
    plt.savefig(OUTPUTS_FIGS / 'fig_08_correlation_heatmap.png', dpi=150)
    plt.close()

    print("[DONE] Phase 4 Complete. 8 charts saved to outputs/figures/.")

if __name__ == "__main__":
    perform_eda()
