import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data.loader import load_feature_vector_data

# Read the CSV file
def analyze_token_clusters(n_clusters=3):
    # Read data
    df = load_feature_vector_data()
    
    # Select features for clustering
    features = ['trade_frequency', 'average_trade_size', 'unique_traders', 
                'max_market_cap', 'initial_liquidity', 'price_volatility', 
                'total_trades', 'lifetime_seconds']
    
    # Create feature matrix
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Reduce dimensionality for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], 
                         cmap='viridis', alpha=0.6)
    plt.title('Token Clusters Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    
    # Add cluster analysis
    cluster_stats = []
    for i in range(n_clusters):
        cluster_tokens = df[df['Cluster'] == i]
        stats = {
            'Cluster': i,
            'Size': len(cluster_tokens),
            'Avg Trade Frequency': cluster_tokens['trade_frequency'].mean(),
            'Avg Trade Size': cluster_tokens['average_trade_size'].mean(),
            'Avg Unique Traders': cluster_tokens['unique_traders'].mean()
        }
        cluster_stats.append(stats)
    
    cluster_summary = pd.DataFrame(cluster_stats)
    
    return df, cluster_summary, plt

# Example usage
if __name__ == "__main__":
    df_with_clusters, cluster_summary, plot = analyze_token_clusters()
    print("\nCluster Summary:")
    print(cluster_summary)
    plot.show()