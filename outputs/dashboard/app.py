import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import joblib
import os
from pathlib import Path

# --- CACHING ---
@st.cache_data
def load_data():
    clean_df = pd.read_csv('data/processed/clean_df.csv', parse_dates=['order_purchase_timestamp'])
    rfm_df = pd.read_csv('data/processed/rfm_segments.csv')
    return clean_df, rfm_df

@st.cache_resource
def load_models():
    model_path = 'models/xgboost_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'models/linear_regression.pkl' # Fallback
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- UI SETUP ---
st.set_page_config(page_title="Myntra E-Commerce Analytics", layout="wide")
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #FF3E6C; }
    .main { background-color: #FFEEF2; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛍️ Myntra Analytics")
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bc/Myntra_Logo.png", width=100)
    st.divider()
    clean_df, rfm_df = load_data()
    
    st.subheader("Global Filters")
    date_range = st.date_input("Select Date Range", [clean_df['order_purchase_timestamp'].min(), clean_df['order_purchase_timestamp'].max()])
    categories = st.multiselect("Select Categories", options=clean_df['product_category_name_english'].unique())
    states = st.multiselect("Select States", options=clean_df['customer_state'].unique())

# Filter data
mask = (clean_df['order_purchase_timestamp'].dt.date >= date_range[0]) & (clean_df['order_purchase_timestamp'].dt.date <= date_range[1])
filtered_df = clean_df[mask]
if categories:
    filtered_df = filtered_df[filtered_df['product_category_name_english'].isin(categories)]
if states:
    filtered_df = filtered_df[filtered_df['customer_state'].isin(states)]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 EDA Explorer", "👥 Customer Segments", "📈 Sales Forecast", "💡 Insights"])

# TAB 1: OVERVIEW
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹ {filtered_df['order_revenue'].sum():,.0f}")
    col2.metric("Total Orders", f"{filtered_df['order_id'].nunique():,}")
    col3.metric("Unique Customers", f"{filtered_df['customer_unique_id'].nunique():,}")
    col4.metric("Avg Review", f"{filtered_df['review_score'].mean():.2f} / 5")
    
    st.subheader("Revenue Trend")
    if PLOTLY_AVAILABLE:
        rev_trend = filtered_df.resample('ME', on='order_purchase_timestamp')['order_revenue'].sum().reset_index()
        fig = px.line(rev_trend, x='order_purchase_timestamp', y='order_revenue', color_discrete_sequence=['#FF3E6C'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Interactive charts require Plotly. Showing static backup.")
        st.image("outputs/figures/fig_01_monthly_revenue.png")

# TAB 2: EDA EXPLORER
with tab2:
    st.subheader("Deep Dive Analysis")
    chart_type = st.selectbox("Choose Visualisation", 
        ["Monthly Revenue Trend", "Category Revenue", "Delivery Delay vs Rating", "Order Heatmap", "Review Distribution", "Correlation Heatmap"])
    
    chart_map = {
        "Monthly Revenue Trend": "fig_01_monthly_revenue.png",
        "Category Revenue": "fig_02_category_revenue.png",
        "Delivery Delay vs Rating": "fig_04_delivery_vs_rating.png",
        "Order Heatmap": "fig_05_order_heatmap.png",
        "Review Distribution": "fig_06_review_distribution.png",
        "Correlation Heatmap": "fig_08_correlation_heatmap.png"
    }
    
    st.image(f"outputs/figures/{chart_map[chart_type]}")
    st.write("**Key Insight:** Peak order volumes correlate strongly with seasonal holidays in late Q4.")

# TAB 3: SEGMENTS
with tab3:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Segment Distribution")
        seg_counts = rfm_df['Segment'].value_counts()
        if PLOTLY_AVAILABLE:
            fig_pie = px.pie(values=seg_counts.values, names=seg_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie)
    with col_b:
        st.subheader("3D RFM Space")
        if PLOTLY_AVAILABLE:
            fig_3d = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color='Segment')
            st.plotly_chart(fig_3d)

# TAB 4: FORECAST
with tab4:
    st.subheader("Predict Order Revenue")
    model = load_models()
    if model:
        c1, c2 = st.columns(2)
        m = c1.selectbox("Month", range(1, 13))
        cat = c1.selectbox("Category", list(range(20))) # Simplified for demo
        it = c2.slider("Item Count", 1, 10, 1)
        fr = c2.number_input("Freight", 0, 100, 20)
        
        if st.button("Predict"):
            # Mock feature vector matching the training features
            # features: month, year, dow, hour, item_count, total_freight, category_encoded, payment_type_encoded, delivery_days_actual, state_encoded
            input_data = np.array([[m, 2023, 2, 12, it, fr, cat, 0, 7, 1]])
            pred = model.predict(input_data)[0]
            st.metric("Predicted Revenue", f"₹ {pred:.2f}")
    else:
        st.error("Prediction models not found. Run the training pipeline.")

# TAB 5: INSIGHTS
with tab5:
    st.info("### Strategic Recommendations")
    st.markdown("""
    - **Retention**: Target the **At Risk** customers (Low Recency) with free shipping vouchers.
    - **Inventory**: Stock up on **Health & Beauty** products before the October festive peak.
    - **Logistics**: The correlation heatmap shows that **Delivery Delay** is the #1 reason for low ratings. Optimize last-mile delivery in SP region.
    """)
    st.divider()
    st.download_button("Download Full Report", filtered_df.to_csv(), "filtered_report.csv", "text/csv")

