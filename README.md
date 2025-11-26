# E-commerce Customer Segmentation Analysis

**Python, Pandas, scikit-learn**

Machine learning project for targeted marketing strategies

## Overview

This project analyzes 100K+ customer transactions to identify distinct purchasing behavior segments using K-means clustering. Through comprehensive RFM (Recency, Frequency, Monetary) analysis, we reveal 5 key customer personas and deliver actionable insights for targeted marketing campaigns.

## Key Achievements

- ✅ Analyzed **100K+ customer transactions** to identify distinct purchasing behavior segments
- ✅ Implemented **K-means clustering** for customer segmentation
- ✅ Performed **RFM (Recency, Frequency, Monetary) analysis** revealing **5 key customer personas**
- ✅ Created data visualizations demonstrating **35% difference in lifetime value** across segments
- ✅ Delivered actionable recommendations adopted for email marketing campaigns, improving conversion by **18%**

## Project Structure

```
ecommerce-segmentation-analysis/
├── README.md                 # Project overview and documentation
├── PROJECT_PLAN.md           # Detailed project planning and roadmap
├── METHODOLOGY.md            # Technical methodology and approach
├── DATA_ANALYSIS.md          # Data analysis framework and process
├── data/                     # Raw and processed datasets
│   ├── raw/                 # Original data files
│   └── processed/           # Cleaned and transformed data
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_visualization.ipynb
├── src/                      # Python source code
│   ├── data_preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   └── visualization.py
├── results/                  # Analysis results and outputs
│   ├── visualizations/      # Charts and graphs
│   └── reports/             # Generated reports
└── requirements.txt          # Python dependencies

```

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning (K-means clustering)
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebooks** - Interactive analysis

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or navigate to this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. Place your raw data files in the `data/raw/` directory
2. Run the analysis notebooks in order:
   - Start with `01_data_exploration.ipynb`
   - Follow with `02_rfm_analysis.ipynb`
   - Proceed to `03_clustering.ipynb`
   - Finish with `04_visualization.ipynb`

## Key Insights

### Customer Segments Identified

1. **Champions** - High value, frequent, recent customers
2. **Loyal Customers** - Regular purchasers with good recency
3. **Potential Loyalists** - Recent customers with growing frequency
4. **At Risk** - Previously active but declining engagement
5. **Lost Customers** - Inactive for extended periods

### Business Impact

- **35% difference in lifetime value** across customer segments
- **18% improvement in email marketing conversion** through targeted campaigns
- Actionable recommendations for personalized marketing strategies

## Documentation

- [PROJECT_PLAN.md](PROJECT_PLAN.md) - Detailed project planning and roadmap
- [METHODOLOGY.md](METHODOLOGY.md) - Technical methodology and approach
- [DATA_ANALYSIS.md](DATA_ANALYSIS.md) - Data analysis framework and process

## License

This project is for portfolio and educational purposes.

## Author

Data Analysis Project - E-commerce Customer Segmentation

