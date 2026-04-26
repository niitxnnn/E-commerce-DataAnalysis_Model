import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PROCESSED, OUTPUTS_FIGS, MODELS_DIR

def perform_segmentation():
    print("\n--- PHASE 5: RFM ANALYSIS & K-MEANS CLUSTERING ---")
    df = pd.read_csv(DATA_PROCESSED / 'clean_df.csv')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # PART A - RFM
    print("Computing RFM...")
    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'total_payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'total_payment_value': 'Monetary'
    })
    
    # PART B - Normalization
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    joblib.dump(scaler, MODELS_DIR / 'rfm_clustering_scaler.pkl')
    
    # PART C - Optimal K
    print("Finding optimal K...")
    distortions = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        distortions.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.axvline(x=4, color='red', linestyle='--')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(OUTPUTS_FIGS / 'fig_09_elbow_curve.png', dpi=150)
    plt.close()
    
    # PART D - Clustering (k=4)
    print("Fitting KMeans (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Map segments based on Monrtary (sorting clusters by Monetary value)
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
    # 0: Low Engagement, 1: At Risk, 2: Loyal, 3: High Value (rough heuristic)
    mapping = {cluster_means.index[0]: 'Low Engagement', 
               cluster_means.index[1]: 'At Risk', 
               cluster_means.index[2]: 'Loyal', 
               cluster_means.index[3]: 'High Value'}
    rfm['Segment'] = rfm['Cluster'].map(mapping)
    
    # PART E - Visualisation
    if PLOTLY_AVAILABLE:
        print("Saving 3D scatter...")
        fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Segment', title="Customer Segments 3D")
        fig.write_html(OUTPUTS_FIGS / 'fig_09_3d_segments.html')
    else:
        print("[WARN] Plotly not available, skipping 3D chart...")
    
    # PART F - Export
    rfm.to_csv(DATA_PROCESSED / 'rfm_segments.csv')
    print("[DONE] Phase 5 Complete. Results saved.")
    return rfm

if __name__ == "__main__":
    perform_segmentation()
