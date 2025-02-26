import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.data.loader import load_feature_vector_data
from sklearn.neighbors import NearestNeighbors

def analyze_token_clusters(eps=0.5, min_samples=5):
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
   
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(X_scaled)
    
    # Number of clusters (excluding noise points with label -1)
    n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
    n_noise = list(df['Cluster']).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')
   
    # Reduce dimensionality for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
   
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'],
                         cmap='viridis', alpha=0.6)
    plt.title(f'Token Clusters Visualization (DBSCAN: eps={eps}, min_samples={min_samples})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
   
    # Add comprehensive cluster analysis
    cluster_stats = []
    for i in sorted(df['Cluster'].unique()):
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
    
    # For DBSCAN, we don't have cluster centers like in K-means
    # Instead, we can compute the mean feature values for each cluster
    feature_importance = pd.DataFrame(columns=['Cluster'] + features)
    
    for i in sorted(df['Cluster'].unique()):
        cluster_tokens = df[df['Cluster'] == i]
        row = {'Cluster': i}
        
        for feature in features:
            row[feature] = cluster_tokens[feature].mean()
            
        feature_importance = pd.concat([feature_importance, pd.DataFrame([row])], ignore_index=True)
    
    return df, cluster_summary, feature_importance, plt

def find_optimal_eps(X_scaled, k=5):
    """
    Use the elbow method to find a good eps parameter for DBSCAN
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # Sort distances in ascending order
    distances = np.sort(distances[:, k-1])
    
    # Calculate the rate of change (derivative)
    derivative = np.diff(distances)
    
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Distance plot
    plt.subplot(1, 2, 1)
    plt.plot(distances)
    plt.grid(True)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {k}th nearest neighbor')
    plt.title('K-Distance Plot (Elbow Method)')
    
    # Plot 2: Derivative plot to help identify the elbow
    plt.subplot(1, 2, 2)
    plt.plot(derivative)
    plt.grid(True)
    plt.xlabel('Points')
    plt.ylabel('Rate of change in distance')
    plt.title('Derivative of K-Distance')
    
    # Find potential elbow points (where derivative peaks)
    elbow_candidates = np.argsort(-derivative)[:3]  # Top 3 peaks
    
    for candidate in elbow_candidates:
        suggested_eps = distances[candidate]
        print(f"Suggested eps value at point {candidate}: {suggested_eps:.3f}")
    
    plt.tight_layout()
    return plt, distances

# Example usage
if __name__ == "__main__":
    # Load and scale data first for eps estimation
    df_temp = load_feature_vector_data()
    features = ['trade_frequency', 'average_trade_size', 'unique_traders',
              'max_market_cap', 'initial_liquidity', 'price_volatility',
              'total_trades', 'lifetime_seconds']
    X = df_temp[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal eps value
    eps_plot = find_optimal_eps(X_scaled)
    
    # Set eps and min_samples based on the elbow plot
    # You should adjust these values after seeing the elbow plot
    eps = 1.5  # Starting value, adjust based on elbow plot
    min_samples = 50  # Smaller values detect more clusters
    
    df_with_clusters, cluster_summary, feature_importance, plot = analyze_token_clusters(
        eps=eps, min_samples=min_samples
    )
    
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
        for cluster in sorted(df_with_clusters['Cluster'].unique()):
            subset = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            cluster_label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
            axes[i].hist(subset[feature], alpha=0.5, label=cluster_label)
        
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
    
    # Add a silhouette score analysis
    from sklearn.metrics import silhouette_score
    
    # Exclude noise points for silhouette score
    if -1 in df_with_clusters['Cluster'].values:
        mask = df_with_clusters['Cluster'] != -1
        if mask.sum() > 1 and len(set(df_with_clusters.loc[mask, 'Cluster'])) > 1:
            silhouette_avg = silhouette_score(X_scaled[mask], df_with_clusters.loc[mask, 'Cluster'])
            print(f"\nSilhouette Score (excluding noise): {silhouette_avg:.3f}")
        else:
            print("\nCannot calculate silhouette score: not enough clusters or samples")
    
    plot.show()