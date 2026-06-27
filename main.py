import pandas as pd
import os
from app.data_preprocessing import preprocess_pipeline
from app.rfm_analysis import perform_rfm_analysis
from app.clustering import find_optimal_clusters, perform_kmeans_clustering, analyze_clusters, assign_persona_names, evaluate_clustering_quality, prepare_features_for_clustering
from app.visualization import create_all_visualizations


def main():
    data_file = 'data/raw/transactions.csv'
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please place your transaction data file in data/raw/transactions.csv")
        print("Required columns: customer_id, transaction_date, transaction_amount")
        return
    
    print("Starting E-commerce Customer Segmentation Analysis...")
    print("=" * 60)
    
    print("\n[1/5] Loading and preprocessing data...")
    transactions_df, customer_data = preprocess_pipeline(data_file)
    print(f"   Processed {len(transactions_df)} transactions")
    print(f"   Unique customers: {len(customer_data)}")
    
    print("\n[2/5] Performing RFM analysis...")
    rfm_df, rfm_summary = perform_rfm_analysis(customer_data)
    print(f"   RFM segments identified: {len(rfm_summary)}")
    print("\n   RFM Segment Summary:")
    print(rfm_summary[['segment', 'customer_count', 'avg_ltv']].to_string(index=False))
    
    print("\n[3/5] Finding optimal number of clusters...")
    features, scaler = prepare_features_for_clustering(rfm_df)
    optimal_k, clustering_results = find_optimal_clusters(features)
    print(f"   Optimal number of clusters: {optimal_k}")
    print(f"   Silhouette score: {clustering_results['silhouette_scores'][optimal_k-2]:.3f}")
    
    print("\n[4/5] Performing K-means clustering...")
    clustered_df, kmeans_model, scaler = perform_kmeans_clustering(rfm_df, n_clusters=optimal_k)
    clustered_df = assign_persona_names(clustered_df)
    cluster_summary = analyze_clusters(clustered_df)
    print(f"   Clusters created: {len(cluster_summary)}")
    print("\n   Cluster Summary:")
    print(cluster_summary[['cluster', 'customer_count', 'avg_ltv']].to_string(index=False))
    
    clustering_quality = evaluate_clustering_quality(features, clustered_df['cluster'].values)
    print(f"\n   Clustering Quality Metrics:")
    print(f"   - Silhouette Score: {clustering_quality['silhouette_score']:.3f}")
    print(f"   - Davies-Bouldin Index: {clustering_quality['davies_bouldin_score']:.3f}")
    print(f"   - Calinski-Harabasz Index: {clustering_quality['calinski_harabasz_score']:.2f}")
    
    ltv_difference = calculate_ltv_difference(clustered_df)
    print(f"\n   Lifetime Value Analysis:")
    print(f"   - Highest LTV segment: ${clustered_df.groupby('persona')['ltv'].mean().max():,.2f}")
    print(f"   - Lowest LTV segment: ${clustered_df.groupby('persona')['ltv'].mean().min():,.2f}")
    print(f"   - LTV difference: {ltv_difference:.1f}%")
    
    print("\n[5/5] Generating visualizations...")
    create_all_visualizations(
        clustered_df,
        rfm_summary,
        cluster_summary,
        clustering_results
    )
    print("   Visualizations saved to results/visualizations/")
    
    print("\n[6/6] Saving results...")
    os.makedirs('data/processed', exist_ok=True)
    clustered_df.to_csv('data/processed/customer_segments.csv', index=False)
    rfm_summary.to_csv('results/reports/rfm_summary.csv', index=False)
    cluster_summary.to_csv('results/reports/cluster_summary.csv', index=False)
    print("   Results saved to data/processed/ and results/reports/")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    print("\nKey Findings:")
    print(f"- Total customers analyzed: {len(clustered_df)}")
    print(f"- Customer segments identified: {len(cluster_summary)}")
    print(f"- LTV difference across segments: {ltv_difference:.1f}%")
    print(f"- Top segment: {cluster_summary.iloc[0]['cluster']} with {cluster_summary.iloc[0]['customer_count']} customers")




def calculate_ltv_difference(df):
    persona_ltv = df.groupby('persona')['ltv'].mean()
    max_ltv = persona_ltv.max()
    min_ltv = persona_ltv.min()
    difference = ((max_ltv - min_ltv) / min_ltv) * 100
    return difference


if __name__ == '__main__':
    main()

