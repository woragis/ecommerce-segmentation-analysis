import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_rfm_scores(df: pd.DataFrame,
                        recency_col: str = 'recency',
                        frequency_col: str = 'frequency',
                        monetary_col: str = 'monetary',
                        method: str = 'quintile') -> pd.DataFrame:
    
    rfm_df = df.copy()
    
    if method == 'quintile':
        rfm_df['R_Score'] = pd.qcut(
            rfm_df[recency_col].rank(method='first'),
            5,
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        ).astype(int)
        
        rfm_df['F_Score'] = pd.qcut(
            rfm_df[frequency_col].rank(method='first'),
            5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
        
        rfm_df['M_Score'] = pd.qcut(
            rfm_df[monetary_col].rank(method='first'),
            5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
    
    elif method == 'custom':
        rfm_df['R_Score'] = pd.cut(
            rfm_df[recency_col],
            bins=[0, 30, 60, 90, 180, float('inf')],
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        rfm_df['F_Score'] = pd.cut(
            rfm_df[frequency_col],
            bins=[0, 1, 2, 3, 5, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        rfm_df['M_Score'] = pd.cut(
            rfm_df[monetary_col],
            bins=[0, 50, 100, 200, 500, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
    
    rfm_df['RFM_Score'] = (
        rfm_df['R_Score'].astype(str) +
        rfm_df['F_Score'].astype(str) +
        rfm_df['M_Score'].astype(str)
    )
    
    return rfm_df


def assign_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    
    segment_map = {
        'Champions': ['555', '554', '544', '545', '454', '455', '445'],
        'Loyal Customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
        'Potential Loyalists': ['512', '511', '422', '421', '412', '411', '311'],
        'New Customers': ['155', '154', '144', '214', '215', '115', '114'],
        'Promising': ['525', '524', '523', '522', '521', '515', '514', '513'],
        'Need Attention': ['332', '331', '321', '312', '221', '213'],
        'About to Sleep': ['255', '254', '245', '244', '234', '225', '224', '153', '152', '151', '145', '143', '142', '141', '132', '124', '123', '122', '121', '111', '112', '113', '114', '115'],
        'At Risk': ['344', '343', '334', '333', '323', '322', '233', '232', '223', '222', '132', '123', '122', '121', '113', '112', '111'],
        'Lost': ['111', '112', '113', '114', '115']
    }
    
    reverse_map = {}
    for segment, scores in segment_map.items():
        for score in scores:
            if score not in reverse_map:
                reverse_map[score] = segment
    
    df['RFM_Segment'] = df['RFM_Score'].map(reverse_map).fillna('Other')
    
    return df


def calculate_ltv(df: pd.DataFrame,
                  avg_transaction_col: str = 'avg_transaction_value',
                  frequency_col: str = 'frequency',
                  lifetime_col: str = 'customer_lifetime') -> pd.DataFrame:
    
    df = df.copy()
    
    df['customer_lifetime_months'] = df[lifetime_col] / 30
    df['customer_lifetime_months'] = df['customer_lifetime_months'].replace(0, 1)
    
    df['ltv'] = (
        df[avg_transaction_col] *
        df[frequency_col] *
        (df['customer_lifetime_months'] / 12)
    )
    
    return df


def analyze_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    
    segment_analysis = df.groupby('RFM_Segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'ltv': 'mean',
        'avg_transaction_value': 'mean'
    }).reset_index()
    
    segment_analysis.columns = [
        'segment',
        'customer_count',
        'avg_recency',
        'avg_frequency',
        'avg_monetary',
        'avg_ltv',
        'avg_transaction_value'
    ]
    
    segment_analysis['customer_percentage'] = (
        segment_analysis['customer_count'] / segment_analysis['customer_count'].sum() * 100
    )
    
    segment_analysis = segment_analysis.sort_values('avg_ltv', ascending=False)
    
    return segment_analysis


def perform_rfm_analysis(customer_data: pd.DataFrame,
                        scoring_method: str = 'quintile') -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    rfm_df = calculate_rfm_scores(customer_data, method=scoring_method)
    rfm_df = assign_rfm_segments(rfm_df)
    rfm_df = calculate_ltv(rfm_df)
    segment_summary = analyze_rfm_segments(rfm_df)
    
    return rfm_df, segment_summary

