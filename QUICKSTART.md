# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate sample data (optional):**
```bash
python generate_sample_data.py
```

Or place your own transaction data in `data/raw/transactions.csv` with columns:
- `customer_id`: Unique customer identifier
- `transaction_date`: Date in YYYY-MM-DD format
- `transaction_amount`: Transaction value (numeric)

## Running the Analysis

**Run the complete analysis:**
```bash
python main.py
```

This will:
1. Load and preprocess transaction data
2. Perform RFM analysis
3. Find optimal number of clusters
4. Perform K-means clustering
5. Generate visualizations
6. Save results to CSV files

## Output Files

- **Processed data:** `data/processed/customer_segments.csv`
- **RFM summary:** `results/reports/rfm_summary.csv`
- **Cluster summary:** `results/reports/cluster_summary.csv`
- **Visualizations:** `results/visualizations/*.png`

## Project Structure

```
ecommerce-segmentation-analysis/
├── app/                    # Core modules
│   ├── data_preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   └── visualization.py
├── data/
│   ├── raw/               # Input transaction data
│   └── processed/         # Processed customer data
├── notebooks/             # Jupyter notebooks (optional)
├── results/
│   ├── visualizations/    # Generated charts
│   └── reports/           # Analysis summaries
├── main.py                # Main entry point
└── generate_sample_data.py # Sample data generator
```

## Using Individual Modules

You can also use the modules independently:

```python
from app.data_preprocessing import preprocess_pipeline
from app.rfm_analysis import perform_rfm_analysis
from app.clustering import perform_kmeans_clustering

transactions, customers = preprocess_pipeline('data/raw/transactions.csv')
rfm_df, rfm_summary = perform_rfm_analysis(customers)
clustered_df, model, scaler = perform_kmeans_clustering(rfm_df, n_clusters=5)
```

