import pandas as pd
import os

data_files = [
    'data/Fake.csv',
    'data/True.csv',
    'data/clickbait.csv',
]

for fname in data_files:
    path = os.path.abspath(fname)
    print('FILE:', path)
    print('EXISTS:', os.path.exists(path))
    if os.path.exists(path):
        df = pd.read_csv(path)
        print('SHAPE:', df.shape)
        print('COLUMNS:', list(df.columns)[:10])
        print('HEAD:', df.head(2).to_dict(orient='records'))
    print('---')
