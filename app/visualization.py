import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import os


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_rfm_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(df['recency'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Recency Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df['frequency'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title('Frequency Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Transactions')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(df['monetary'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[2].set_title('Monetary Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Total Spending')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rfm_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    rfm_heatmap = df.pivot_table(
        values='monetary',
        index='R_Score',
        columns='F_Score',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(rfm_heatmap, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Average Monetary Value'})
    plt.title('RFM Heatmap: Average Monetary Value by R and F Scores', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Frequency Score', fontsize=12)
    plt.ylabel('Recency Score', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_segment_distribution(df: pd.DataFrame, segment_col: str = 'RFM_Segment', save_path: Optional[str] = None):
    segment_counts = df[segment_col].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("husl", len(segment_counts))
    
    ax1.barh(segment_counts.index, segment_counts.values, color=colors)
    ax1.set_xlabel('Number of Customers', fontsize=12)
    ax1.set_title('Customer Count by Segment', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Segment Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ltv_comparison(df: pd.DataFrame, segment_col: str = 'RFM_Segment', save_path: Optional[str] = None):
    segment_ltv = df.groupby(segment_col)['ltv'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(segment_ltv)), segment_ltv.values, color=sns.color_palette("viridis", len(segment_ltv)))
    plt.xticks(range(len(segment_ltv)), segment_ltv.index, rotation=45, ha='right')
    plt.ylabel('Average Lifetime Value ($)', fontsize=12)
    plt.title('Average Lifetime Value by Segment', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, segment_ltv.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_scatter(df: pd.DataFrame, 
                        cluster_col: str = 'cluster',
                        save_path: Optional[str] = None):
    
    features = df[['recency', 'frequency', 'monetary']].values
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    df_2d = pd.DataFrame(features_2d, columns=['PC1', 'PC2'])
    df_2d[cluster_col] = df[cluster_col].values
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_2d['PC1'], df_2d['PC2'], c=df_2d[cluster_col], 
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title('Customer Clusters (2D PCA Projection)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_elbow_curve(k_values: list, inertias: list, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=16, fontweight='bold', pad=20)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_silhouette_analysis(k_values: list, silhouette_scores: list, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Analysis for Optimal k', fontsize=16, fontweight='bold', pad=20)
    plt.grid(alpha=0.3)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k = {optimal_k}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_characteristics(df: pd.DataFrame,
                                cluster_col: str = 'cluster',
                                save_path: Optional[str] = None):
    
    cluster_stats = df.groupby(cluster_col).agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'ltv': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['recency', 'frequency', 'monetary', 'ltv']
    titles = ['Average Recency', 'Average Frequency', 'Average Monetary', 'Average LTV']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(range(len(cluster_stats)), cluster_stats[metric], 
                     color=sns.color_palette("husl", len(cluster_stats)))
        ax.set_xticks(range(len(cluster_stats)))
        ax.set_xticklabels(cluster_stats[cluster_col], rotation=45, ha='right')
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, cluster_stats[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Cluster Characteristics Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_segment_comparison(rfm_summary: pd.DataFrame, 
                           cluster_summary: pd.DataFrame,
                           save_path: Optional[str] = None):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    rfm_ltv = rfm_summary.sort_values('avg_ltv', ascending=False)
    cluster_ltv = cluster_summary.sort_values('avg_ltv', ascending=False)
    
    axes[0].barh(rfm_ltv['segment'], rfm_ltv['avg_ltv'], color='steelblue')
    axes[0].set_xlabel('Average Lifetime Value ($)', fontsize=12)
    axes[0].set_title('RFM Segments - Average LTV', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    axes[1].barh(cluster_ltv['cluster'].astype(str), cluster_ltv['avg_ltv'], color='coral')
    axes[1].set_xlabel('Average Lifetime Value ($)', fontsize=12)
    axes[1].set_title('K-means Clusters - Average LTV', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(df: pd.DataFrame,
                             rfm_summary: pd.DataFrame,
                             cluster_summary: pd.DataFrame,
                             clustering_results: dict = None,
                             output_dir: str = 'results/visualizations'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_rfm_distribution(df, os.path.join(output_dir, 'rfm_distribution.png'))
    plot_rfm_heatmap(df, os.path.join(output_dir, 'rfm_heatmap.png'))
    plot_segment_distribution(df, 'RFM_Segment', os.path.join(output_dir, 'rfm_segment_distribution.png'))
    plot_ltv_comparison(df, 'RFM_Segment', os.path.join(output_dir, 'ltv_by_rfm_segment.png'))
    
    if 'cluster' in df.columns:
        plot_cluster_scatter(df, os.path.join(output_dir, 'cluster_scatter.png'))
        plot_cluster_characteristics(df, os.path.join(output_dir, 'cluster_characteristics.png'))
        plot_segment_distribution(df, 'persona', os.path.join(output_dir, 'persona_distribution.png'))
        plot_ltv_comparison(df, 'persona', os.path.join(output_dir, 'ltv_by_persona.png'))
    
    if clustering_results:
        if 'k_values' in clustering_results and 'inertias' in clustering_results:
            plot_elbow_curve(
                clustering_results['k_values'],
                clustering_results['inertias'],
                os.path.join(output_dir, 'elbow_curve.png')
            )
        if 'k_values' in clustering_results and 'silhouette_scores' in clustering_results:
            plot_silhouette_analysis(
                clustering_results['k_values'],
                clustering_results['silhouette_scores'],
                os.path.join(output_dir, 'silhouette_analysis.png')
            )
    
    plot_segment_comparison(rfm_summary, cluster_summary, os.path.join(output_dir, 'segment_comparison.png'))

