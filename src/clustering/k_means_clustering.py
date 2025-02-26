import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.data.loader import load_feature_vector_data

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
   
    # Add comprehensive cluster analysis
    cluster_stats = []
    for i in range(n_clusters):
        cluster_tokens = df[df['Cluster'] == i]
        stats = {
            'Cluster': i,
            'Size': len(cluster_tokens),
            # Include all features in statistics
            'Avg Trade Frequency': cluster_tokens['trade_frequency'].mean(),
            'Avg Trade Size': cluster_tokens['average_trade_size'].mean(),
            'Avg Unique Traders': cluster_tokens['unique_traders'].mean(),
            'Avg Max Market Cap': cluster_tokens['max_market_cap'].mean(),
            'Avg Initial Liquidity': cluster_tokens['initial_liquidity'].mean(),
            'Avg Price Volatility': cluster_tokens['price_volatility'].mean(),
            'Avg Total Trades': cluster_tokens['total_trades'].mean(),
            'Avg Lifetime (seconds)': cluster_tokens['lifetime_seconds'].mean(),
            'Median Trade Frequency': cluster_tokens['trade_frequency'].median(),
            'Median Trade Size': cluster_tokens['average_trade_size'].median(),
            'Median Unique Traders': cluster_tokens['unique_traders'].median(),
            'Std Dev Trade Frequency': cluster_tokens['trade_frequency'].std(),
            'Min Trade Frequency': cluster_tokens['trade_frequency'].min(),
            'Max Trade Frequency': cluster_tokens['trade_frequency'].max()
        }
        cluster_stats.append(stats)
   
    cluster_summary = pd.DataFrame(cluster_stats)
    
    # Calculate feature importance for each cluster
    feature_importance = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )
    
    return df, cluster_summary, feature_importance, plt

# Example usage
if __name__ == "__main__":
    df_with_clusters, cluster_summary, feature_importance, plot = analyze_token_clusters(n_clusters=2)
    
    print("\nCluster Summary:")
    print(cluster_summary)
    
    print("\nFeature Importance by Cluster:")
    print(feature_importance)
    
    # Additional visualizations
    # Plot feature distributions by cluster
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(df_with_clusters[['trade_frequency', 'average_trade_size', 
                                                'unique_traders', 'max_market_cap', 
                                                'initial_liquidity', 'price_volatility', 
                                                'total_trades', 'lifetime_seconds']].columns):
        for cluster in df_with_clusters['Cluster'].unique():
            subset = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            axes[i].hist(subset[feature], alpha=0.5, label=f'Cluster {cluster}')
        
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].legend()
    
    plt.tight_layout()
    
    # Show cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.title('Number of Tokens per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(cluster_counts.index)
    
    plot.show()