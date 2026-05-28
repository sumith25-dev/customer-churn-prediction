#!/usr/bin/env python3
"""Download the Telco customer churn dataset from a public source."""
import urllib.request
import os

url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
filepath = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

print('📥 Downloading Telco customer churn dataset...')
try:
    urllib.request.urlretrieve(url, filepath)
    print(f'✅ Dataset downloaded to {filepath}')
    import pandas as pd
    df = pd.read_csv(filepath)
    print(f'   Shape: {df.shape}')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
