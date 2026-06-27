import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def clean_transactions(df: pd.DataFrame, 
                     date_column: str = 'transaction_date',
                     amount_column: str = 'transaction_amount',
                     customer_column: str = 'customer_id') -> pd.DataFrame:
    
    df = df.copy()
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce')
    
    initial_count = len(df)
    
    df = df.dropna(subset=[date_column, amount_column, customer_column])
    
    df = df[df[amount_column] > 0]
    
    df = df[df[date_column] <= datetime.now()]
    
    df = df.drop_duplicates()
    
    removed_count = initial_count - len(df)
    
    return df


def remove_outliers(df: pd.DataFrame, 
                   amount_column: str = 'transaction_amount',
                   method: str = 'iqr',
                   factor: float = 1.5) -> pd.DataFrame:
    
    df = df.copy()
    
    if method == 'iqr':
        Q1 = df[amount_column].quantile(0.25)
        Q3 = df[amount_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[amount_column] >= lower_bound) & (df[amount_column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[amount_column] - df[amount_column].mean()) / df[amount_column].std())
        df = df[z_scores < 3]
    
    return df


def prepare_customer_data(df: pd.DataFrame,
                         date_column: str = 'transaction_date',
                         amount_column: str = 'transaction_amount',
                         customer_column: str = 'customer_id',
                         analysis_date: Optional[datetime] = None) -> pd.DataFrame:
    
    if analysis_date is None:
        analysis_date = df[date_column].max()
    
    customer_data = df.groupby(customer_column).agg({
        date_column: ['min', 'max', 'count'],
        amount_column: ['sum', 'mean']
    }).reset_index()
    
    customer_data.columns = [
        customer_column,
        'first_purchase_date',
        'last_purchase_date',
        'frequency',
        'monetary',
        'avg_transaction_value'
    ]
    
    customer_data['recency'] = (analysis_date - customer_data['last_purchase_date']).dt.days
    customer_data['customer_lifetime'] = (
        customer_data['last_purchase_date'] - customer_data['first_purchase_date']
    ).dt.days
    
    customer_data['customer_lifetime'] = customer_data['customer_lifetime'].replace(0, 1)
    
    customer_data['purchase_frequency_rate'] = (
        customer_data['frequency'] / (customer_data['customer_lifetime'] / 30)
    ).fillna(0)
    
    return customer_data


def preprocess_pipeline(file_path: str,
                       required_columns: list = ['customer_id', 'transaction_date', 'transaction_amount'],
                       remove_outliers_flag: bool = True,
                       outlier_method: str = 'iqr') -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df = load_data(file_path)
    validate_data(df, required_columns)
    df = clean_transactions(df)
    
    if remove_outliers_flag:
        df = remove_outliers(df, method=outlier_method)
    
    customer_data = prepare_customer_data(df)
    
    return df, customer_data

