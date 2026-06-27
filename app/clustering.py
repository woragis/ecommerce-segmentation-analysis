import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Tuple, List


def prepare_features_for_clustering(df: pd.DataFrame,
                                   features: List[str] = None) -> np.ndarray:
    
    if features is None:
        features = ['recency', 'frequency', 'monetary']
    
    feature_data = df[features].copy()
    
    feature_data = feature_data.fillna(0)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    return scaled_features, scaler


def find_optimal_clusters(features: np.ndarray,
                         max_clusters: int = 10,
                         random_state: int = 42) -> Tuple[int, dict]:
    
    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'calinski_harabasz_scores': []
    }
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        
        results['k_values'].append(k)
        results['inertias'].append(kmeans.inertia_)
        results['silhouette_scores'].append(silhouette_score(features, labels))
        results['davies_bouldin_scores'].append(davies_bouldin_score(features, labels))
        results['calinski_harabasz_scores'].append(calinski_harabasz_score(features, labels))
    
    silhouette_scores = np.array(results['silhouette_scores'])
    optimal_k = results['k_values'][np.argmax(silhouette_scores)]
    
    return optimal_k, results


def perform_kmeans_clustering(df: pd.DataFrame,
                              n_clusters: int = 5,
                              features: List[str] = None,
                              random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    
    if features is None:
        features = ['recency', 'frequency', 'monetary']
    
    scaled_features, scaler = prepare_features_for_clustering(df, features)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    return df_clustered, kmeans, scaler


def analyze_clusters(df: pd.DataFrame,
                    cluster_col: str = 'cluster') -> pd.DataFrame:
    
    cluster_analysis = df.groupby(cluster_col).agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'ltv': 'mean',
        'avg_transaction_value': 'mean',
        'customer_lifetime': 'mean'
    }).reset_index()
    
    cluster_analysis.columns = [
        'cluster',
        'customer_count',
        'avg_recency',
        'avg_frequency',
        'avg_monetary',
        'avg_ltv',
        'avg_transaction_value',
        'avg_customer_lifetime'
    ]
    
    cluster_analysis['customer_percentage'] = (
        cluster_analysis['customer_count'] / cluster_analysis['customer_count'].sum() * 100
    )
    
    cluster_analysis = cluster_analysis.sort_values('avg_ltv', ascending=False)
    
    return cluster_analysis


def assign_persona_names(df: pd.DataFrame,
                         cluster_col: str = 'cluster',
                         ltv_col: str = 'ltv',
                         recency_col: str = 'recency',
                         frequency_col: str = 'frequency') -> pd.DataFrame:
    
    df = df.copy()
    
    cluster_stats = df.groupby(cluster_col).agg({
        ltv_col: 'mean',
        recency_col: 'mean',
        frequency_col: 'mean',
        'monetary': 'mean'
    }).reset_index()
    
    cluster_stats = cluster_stats.sort_values(ltv_col, ascending=False)
    
    persona_names = {
        0: 'Champions',
        1: 'Loyal Customers',
        2: 'Potential Loyalists',
        3: 'At Risk',
        4: 'Lost Customers'
    }
    
    for idx, row in cluster_stats.iterrows():
        cluster_num = int(row[cluster_col])
        if cluster_num < len(persona_names):
            persona_names[cluster_num] = list(persona_names.values())[idx]
    
    reverse_mapping = {}
    for old_cluster, new_name in enumerate(cluster_stats[cluster_col]):
        reverse_mapping[int(new_name)] = list(persona_names.values())[old_cluster]
    
    df['persona'] = df[cluster_col].map(reverse_mapping)
    
    return df


def evaluate_clustering_quality(features: np.ndarray,
                               labels: np.ndarray) -> dict:
    
    metrics = {
        'silhouette_score': silhouette_score(features, labels),
        'davies_bouldin_score': davies_bouldin_score(features, labels),
        'calinski_harabasz_score': calinski_harabasz_score(features, labels),
        'n_clusters': len(np.unique(labels)),
        'n_samples': len(labels)
    }
    
    return metrics

