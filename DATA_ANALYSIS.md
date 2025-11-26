# Data Analysis Framework: E-commerce Customer Segmentation

## Data Analysis Workflow

This document outlines the systematic approach to analyzing e-commerce customer data for segmentation purposes.

## 1. Data Understanding

### 1.1 Data Source Identification
- **Primary Data:** Customer transaction records
- **Data Volume:** 100K+ transactions
- **Time Period:** Define analysis window (e.g., last 12-24 months)
- **Data Freshness:** Ensure data is current and complete

### 1.2 Data Schema

**Required Fields:**
```
customer_id: string/int - Unique customer identifier
transaction_date: datetime - Date and time of transaction
transaction_amount: float - Monetary value of transaction
order_id: string/int - Unique order identifier
```

**Optional Fields (if available):**
```
product_id: string/int - Product purchased
product_category: string - Product category
customer_age: int - Customer age
customer_gender: string - Customer gender
customer_location: string - Geographic location
customer_segment: string - Existing segment (if any)
```

### 1.3 Data Quality Assessment

**Checklist:**
- [ ] Total record count
- [ ] Date range coverage
- [ ] Unique customer count
- [ ] Missing value percentage
- [ ] Duplicate record count
- [ ] Data type consistency
- [ ] Value range validation
- [ ] Outlier identification

## 2. Exploratory Data Analysis (EDA)

### 2.1 Univariate Analysis

**Transaction Amount:**
- Distribution (histogram, box plot)
- Central tendencies (mean, median, mode)
- Spread (std, IQR, range)
- Skewness and kurtosis
- Outlier detection

**Transaction Date:**
- Time series distribution
- Seasonality patterns
- Day of week patterns
- Monthly trends
- Peak periods identification

**Customer Metrics:**
- Number of transactions per customer
- Total spending per customer
- Average transaction value per customer
- Customer lifetime distribution

### 2.2 Bivariate Analysis

**Relationships to Explore:**
- Transaction amount vs. Date (trends over time)
- Transaction count vs. Total spending
- Customer lifetime vs. Total spending
- Frequency vs. Recency
- Monetary vs. Frequency

**Visualizations:**
- Scatter plots
- Correlation matrices
- Heatmaps

### 2.3 Multivariate Analysis

**Key Relationships:**
- RFM dimensions correlation
- Customer behavior patterns
- Product preferences by customer groups
- Temporal purchasing patterns

## 3. RFM Analysis Framework

### 3.1 RFM Calculation Process

**Step 1: Define Analysis Date**
```python
analysis_date = max(transaction_date)  # Most recent date
# OR
analysis_date = datetime.now()  # Current date
```

**Step 2: Calculate RFM Metrics**

**Recency:**
```python
recency = analysis_date - last_transaction_date
# Convert to days
recency_days = recency.days
```

**Frequency:**
```python
frequency = count(unique_transactions_per_customer)
```

**Monetary:**
```python
monetary = sum(transaction_amounts_per_customer)
```

**Step 3: Create RFM Table**
```python
rfm_table = customer_id | recency | frequency | monetary
```

### 3.2 RFM Scoring

**Scoring Strategy:**
1. **Quintile Method (Recommended):**
   - Divide each metric into 5 equal groups
   - Assign scores 1-5
   - Recency: Lower = Higher score (1 = most recent)
   - Frequency: Higher = Higher score (5 = most frequent)
   - Monetary: Higher = Higher score (5 = highest spending)

2. **Custom Thresholds:**
   - Business-defined cutoffs
   - Example: R=1 if <30 days, R=2 if 30-60 days

3. **Statistical Method:**
   - Based on mean and standard deviation
   - Z-score based scoring

### 3.3 RFM Segment Assignment

**Segment Mapping:**
```python
# Example segment definitions
segments = {
    'Champions': ['555', '554', '544', '545', '454', '455', '445'],
    'Loyal Customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
    'Potential Loyalists': ['512', '511', '422', '421', '412', '411', '311'],
    'New Customers': ['155', '154', '144', '214', '215', '115', '114'],
    'At Risk': ['344', '343', '334', '333', '323', '322', '233', '232', '223', '222'],
    'Lost': ['111', '112', '113', '114', '115']
}
```

## 4. Clustering Analysis Framework

### 4.1 Feature Engineering for Clustering

**Base Features:**
- Recency (days)
- Frequency (count)
- Monetary (total amount)

**Derived Features:**
- Average transaction value
- Customer lifetime (days)
- Purchase frequency rate (transactions per month)
- Days between purchases (average)
- Recency ratio (recency / customer lifetime)

**Feature Selection:**
- Remove highly correlated features
- Select features that capture distinct behaviors
- Consider business relevance

### 4.2 Data Preprocessing for Clustering

**Scaling:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

**Why Scaling is Critical:**
- K-means uses Euclidean distance
- Features with larger scales dominate
- All features should be on similar scale

### 4.3 Optimal Cluster Number

**Elbow Method:**
```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)

# Plot inertias vs K
# Identify elbow point
```

**Silhouette Analysis:**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(score)

# Optimal k has highest silhouette score
```

### 4.4 Cluster Interpretation

**For Each Cluster, Analyze:**
1. **Size:** Number of customers
2. **Centroid Values:** Mean RFM values
3. **Characteristics:**
   - Average recency
   - Average frequency
   - Average monetary value
   - Average transaction value
4. **Distribution:** Spread of values within cluster
5. **Business Meaning:** What does this cluster represent?

## 5. Customer Persona Development

### 5.1 Persona Template

**For Each Segment/Cluster:**

```
Persona Name: [Descriptive Name]

Characteristics:
- Recency: [Description]
- Frequency: [Description]
- Monetary: [Description]
- Average Transaction Value: $X
- Customer Lifetime: X days
- Total Lifetime Value: $X

Behavioral Patterns:
- Purchase frequency: [Description]
- Preferred timing: [Description]
- Product preferences: [If available]

Business Value:
- Segment size: X customers (Y%)
- Revenue contribution: $X (Y%)
- Average LTV: $X

Marketing Recommendations:
- [Action 1]
- [Action 2]
- [Action 3]
```

### 5.2 Lifetime Value Analysis

**Calculate LTV by Segment:**
```python
# Simple LTV
ltv = avg_order_value * purchase_frequency * customer_lifespan_months

# Segment comparison
ltv_by_segment = df.groupby('segment')['ltv'].agg(['mean', 'median', 'sum'])
```

**Visualization:**
- Bar chart: Average LTV by segment
- Box plot: LTV distribution by segment
- Calculate percentage difference between segments

### 5.3 Segment Comparison

**Metrics to Compare:**
- Segment size (count and percentage)
- Average RFM scores
- Lifetime value
- Revenue contribution
- Purchase patterns
- Retention rates (if available)

**Visualizations:**
- Comparison bar charts
- Radar charts for RFM comparison
- Heatmaps for segment characteristics

## 6. Validation & Quality Checks

### 6.1 Statistical Validation

**Cluster Quality:**
- Silhouette score > 0.3 (acceptable)
- Silhouette score > 0.5 (good)
- Silhouette score > 0.7 (excellent)

**Segment Differences:**
- ANOVA test for continuous variables
- Chi-square test for categorical variables
- Verify significant differences between segments

### 6.2 Business Validation

**Questions to Answer:**
- Do segments make business sense?
- Are personas actionable?
- Can marketing team use these insights?
- Do segments align with known customer behaviors?

### 6.3 Data Validation

**Checks:**
- All customers assigned to a segment
- No duplicate assignments
- Segment sizes are reasonable
- No empty segments

## 7. Visualization Strategy

### 7.1 RFM Visualizations

**RFM Distribution:**
- Histograms for R, F, M scores
- Box plots for RFM values

**RFM Heatmap:**
- R (rows) vs F (columns)
- Color intensity = Monetary value
- Shows value distribution across RFM combinations

**Segment Distribution:**
- Pie chart: Segment sizes
- Bar chart: Customer count by segment

### 7.2 Clustering Visualizations

**Cluster Scatter Plot:**
- Use PCA to reduce to 2D/3D
- Color by cluster assignment
- Show cluster centers

**Cluster Characteristics:**
- Bar chart: Average RFM by cluster
- Radar chart: Multi-dimensional comparison
- Box plots: Distribution within clusters

**Cluster Evaluation:**
- Elbow curve
- Silhouette plot
- Cluster size comparison

### 7.3 Business Insights Visualizations

**Value Analysis:**
- Bar chart: Average LTV by segment
- Stacked bar: Revenue contribution
- Line chart: LTV trends over time (if applicable)

**Behavioral Analysis:**
- Purchase frequency distribution
- Transaction amount distribution
- Time-based patterns

**Comparison Charts:**
- Side-by-side segment comparisons
- Before/after analysis (if applicable)

## 8. Reporting Framework

### 8.1 Executive Summary
- Key findings (1-2 pages)
- Business impact
- Top recommendations

### 8.2 Technical Report
- Methodology
- Data processing steps
- Analysis results
- Statistical validation

### 8.3 Actionable Recommendations
- Segment-specific strategies
- Marketing campaign ideas
- Expected outcomes
- Implementation roadmap

## 9. Analysis Checklist

### Pre-Analysis
- [ ] Data collected and validated
- [ ] Data quality issues identified and resolved
- [ ] Analysis objectives clear
- [ ] Tools and environment set up

### During Analysis
- [ ] EDA completed
- [ ] RFM scores calculated
- [ ] RFM segments assigned
- [ ] Optimal cluster number determined
- [ ] K-means clustering performed
- [ ] Clusters interpreted
- [ ] Personas developed
- [ ] LTV calculated
- [ ] Visualizations created

### Post-Analysis
- [ ] Results validated
- [ ] Business stakeholders consulted
- [ ] Recommendations developed
- [ ] Reports generated
- [ ] Documentation completed

## 10. Expected Deliverables

1. **Data Quality Report**
2. **EDA Report** with visualizations
3. **RFM Analysis Results**
4. **Clustering Analysis Results**
5. **Customer Persona Profiles** (5 personas)
6. **Lifetime Value Analysis**
7. **Visualization Portfolio**
8. **Executive Summary Report**
9. **Marketing Recommendations Document**
10. **Technical Documentation**

