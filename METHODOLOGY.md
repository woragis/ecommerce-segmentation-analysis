# Methodology: E-commerce Customer Segmentation Analysis

## Overview

This document outlines the technical methodology for performing customer segmentation analysis using RFM analysis and K-means clustering on e-commerce transaction data.

## 1. Data Preprocessing

### 1.1 Data Collection
- Source: E-commerce transaction database
- Minimum required fields:
  - `customer_id`: Unique customer identifier
  - `transaction_date`: Date of purchase
  - `transaction_amount`: Monetary value of transaction
  - `order_id`: Unique order identifier (optional but recommended)

### 1.2 Data Cleaning
**Steps:**
1. **Handle Missing Values**
   - Identify missing values in critical fields
   - Remove or impute based on business logic
   - Document missing data patterns

2. **Remove Duplicates**
   - Identify duplicate transactions
   - Decide on deduplication strategy
   - Preserve data integrity

3. **Outlier Detection**
   - Identify statistical outliers in transaction amounts
   - Investigate extreme values (fraud, errors, VIP customers)
   - Decide on outlier treatment (remove, cap, or keep)

4. **Data Type Conversion**
   - Convert dates to datetime format
   - Ensure numeric fields are properly typed
   - Validate data ranges

5. **Data Validation**
   - Check for negative transaction amounts (returns/refunds)
   - Validate date ranges (no future dates)
   - Ensure customer IDs are consistent

### 1.3 Feature Engineering

**Base Features:**
- Transaction count per customer
- Total spending per customer
- Average transaction value
- Days since first purchase
- Days since last purchase

**Derived Features:**
- Customer lifetime (days between first and last purchase)
- Purchase frequency (transactions per time period)
- Average days between purchases

## 2. RFM Analysis

### 2.1 RFM Score Calculation

**Recency (R):**
- Definition: Days since customer's last purchase
- Calculation: `R = Current Date - Last Purchase Date`
- Scoring: Lower values = higher recency (more recent)
- Typical scale: 1-5 (1 = most recent, 5 = least recent)

**Frequency (F):**
- Definition: Number of transactions in analysis period
- Calculation: Count of unique transactions per customer
- Scoring: Higher values = higher frequency
- Typical scale: 1-5 (1 = least frequent, 5 = most frequent)

**Monetary (M):**
- Definition: Total amount spent by customer
- Calculation: Sum of all transaction amounts per customer
- Scoring: Higher values = higher monetary value
- Typical scale: 1-5 (1 = lowest spending, 5 = highest spending)

### 2.2 RFM Scoring Methods

**Method 1: Quintile-Based Scoring**
```python
# Divide customers into 5 equal groups (20% each)
# Assign scores 1-5 based on quintile
```

**Method 2: Custom Thresholds**
```python
# Define business-specific thresholds
# Example: R=1 if <30 days, R=2 if 30-60 days, etc.
```

**Method 3: Statistical Distribution**
```python
# Use mean and standard deviation
# Score based on standard deviations from mean
```

### 2.3 RFM Segment Creation

**RFM Cell:**
- Combine R, F, M scores: `RFM_Score = R*100 + F*10 + M`
- Example: R=5, F=3, M=4 → RFM_Score = 534

**RFM Segments:**
- **Champions** (555, 554, 544, 545, 454, 455, 445): Best customers
- **Loyal Customers** (543, 444, 435, 355, 354, 345, 344, 335): Regular buyers
- **Potential Loyalists** (512, 511, 422, 421, 412, 411, 311): Recent customers
- **New Customers** (155, 154, 144, 214, 215, 115, 114): First-time buyers
- **Promising** (525, 524, 523, 522, 521, 515, 514, 513): Growing engagement
- **Need Attention** (332, 331, 321, 312, 221, 213): Declining engagement
- **About to Sleep** (255, 254, 245, 244, 234, 225, 224, 153, 152, 151, 145, 143, 142, 141, 132, 124, 123, 122, 121, 111, 112, 113, 114, 115): At risk
- **At Risk** (344, 343, 334, 333, 323, 322, 233, 232, 223, 222, 132, 123, 122, 121, 113, 112, 111): Low engagement
- **Lost** (155, 154, 144, 214, 215, 115, 114, 111): Inactive customers

## 3. K-means Clustering

### 3.1 Feature Selection

**Primary Features:**
- Recency (normalized)
- Frequency (normalized)
- Monetary value (normalized)
- Average transaction value
- Customer lifetime
- Purchase frequency rate

**Feature Scaling:**
- Standardization: `(x - mean) / std`
- Normalization: `(x - min) / (max - min)`
- **Critical:** K-means requires scaled features

### 3.2 Optimal Cluster Number Determination

**Elbow Method:**
```python
# Calculate within-cluster sum of squares (WCSS) for k=1 to k=10
# Plot WCSS vs k
# Identify "elbow" point where rate of decrease slows
```

**Silhouette Analysis:**
```python
# Calculate silhouette score for each k
# Silhouette score ranges from -1 to 1
# Higher scores indicate better-defined clusters
# Optimal k has highest average silhouette score
```

**Gap Statistic:**
```python
# Compare total within intra-cluster variation
# with expected variation under null reference distribution
```

### 3.3 K-means Implementation

**Algorithm Steps:**
1. Initialize k centroids randomly
2. Assign each data point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence
5. Evaluate cluster quality

**Parameters:**
- `n_clusters`: Optimal number (determined from elbow/silhouette)
- `init`: Initialization method ('k-means++' recommended)
- `n_init`: Number of initializations (10-20)
- `max_iter`: Maximum iterations (300)
- `random_state`: For reproducibility

### 3.4 Cluster Evaluation

**Metrics:**
- **Inertia (WCSS):** Lower is better (within-cluster sum of squares)
- **Silhouette Score:** Higher is better (-1 to 1)
- **Davies-Bouldin Index:** Lower is better
- **Calinski-Harabasz Index:** Higher is better

**Interpretation:**
- Analyze cluster centers (mean values for each feature)
- Compare cluster sizes
- Identify distinguishing characteristics
- Validate with business domain knowledge

## 4. Customer Persona Development

### 4.1 Persona Characteristics

For each cluster, analyze:
- **Demographics** (if available): Age, gender, location
- **Behavioral Patterns:**
  - Purchase frequency
  - Average transaction value
  - Preferred products/categories
  - Time of purchase (day, month, season)
- **Value Metrics:**
  - Total lifetime value
  - Average order value
  - Customer acquisition cost (if available)
  - Retention rate

### 4.2 Persona Naming

Create descriptive names based on characteristics:
- **Champions:** High value, frequent, recent
- **Loyal Customers:** Regular, consistent buyers
- **Potential Loyalists:** Growing engagement
- **At Risk:** Declining engagement
- **Lost:** Inactive customers

### 4.3 Lifetime Value Calculation

**Simple LTV:**
```
LTV = Average Order Value × Purchase Frequency × Customer Lifespan
```

**Advanced LTV:**
```
LTV = (Average Order Value × Purchase Frequency × Gross Margin) / Churn Rate
```

## 5. Visualization Strategy

### 5.1 RFM Visualizations
- RFM score distribution (histograms)
- RFM segment pie chart
- RFM heatmap (R vs F, colored by M)
- Segment comparison bar charts

### 5.2 Clustering Visualizations
- 2D/3D scatter plots (PCA for dimensionality reduction)
- Cluster centers comparison
- Silhouette plot
- Elbow curve
- Feature importance by cluster

### 5.3 Business Insights Visualizations
- Lifetime value by segment (bar chart)
- Segment size comparison
- Purchase behavior trends
- Revenue contribution by segment
- Customer journey visualization

## 6. Validation & Testing

### 6.1 Cluster Stability
- Run K-means multiple times with different random seeds
- Check if clusters remain consistent
- Measure cluster assignment stability

### 6.2 Business Validation
- Review clusters with domain experts
- Validate personas make business sense
- Check alignment with known customer behaviors

### 6.3 Statistical Validation
- Test for significant differences between clusters
- ANOVA for continuous variables
- Chi-square tests for categorical variables

## 7. Implementation Code Structure

```python
# 1. Data Loading
import pandas as pd
df = pd.read_csv('transactions.csv')

# 2. RFM Calculation
def calculate_rfm(df, analysis_date):
    rfm = df.groupby('customer_id').agg({
        'transaction_date': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'transaction_amount': 'sum'  # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

# 3. RFM Scoring
def rfm_score(rfm):
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    return rfm

# 4. K-means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)
rfm['Cluster'] = clusters
```

## 8. Expected Outcomes

1. **5 Distinct Customer Segments** with clear characteristics
2. **35% Lifetime Value Difference** between highest and lowest segments
3. **Actionable Insights** for each segment
4. **Marketing Recommendations** tailored to each persona
5. **Visualizations** demonstrating findings

## 9. Best Practices

- Document all assumptions and decisions
- Use reproducible random seeds
- Validate findings with business stakeholders
- Iterate based on feedback
- Keep code modular and well-commented
- Version control all analysis steps

