import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_transactions(n_customers=10000, n_transactions=100000, output_path='data/raw/transactions.csv'):
    np.random.seed(42)
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days
    
    customer_ids = [f'CUST_{i:06d}' for i in range(1, n_customers + 1)]
    
    transactions = []
    
    for i in range(n_transactions):
        customer_id = np.random.choice(customer_ids)
        
        transaction_date = start_date + timedelta(days=np.random.randint(0, date_range))
        
        base_amount = np.random.lognormal(mean=3.5, sigma=1.2)
        transaction_amount = round(base_amount, 2)
        
        order_id = f'ORD_{i+1:08d}'
        
        transactions.append({
            'customer_id': customer_id,
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'transaction_amount': transaction_amount,
            'order_id': order_id
        })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('transaction_date')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} transactions for {df['customer_id'].nunique()} unique customers")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Total revenue: ${df['transaction_amount'].sum():,.2f}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    generate_sample_transactions()

