import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

merged = pd.read_csv('final_merged_updated.csv')

merged.drop(columns={'Unnamed: 0'}, inplace=True)

merged['financialDate'] = pd.to_datetime(merged['financialDate'])

merged = merged.sort_values(by=['Supplier.Number','financialDate'])

most_recent_records = merged.loc[merged.groupby('Supplier.Number')['financialDate'].idxmax()].reset_index(drop=True)

most_recent_records['diff_days'] = 365

most_recent_records = most_recent_records.loc[most_recent_records['eqyYear'] >= 2024]

most_recent_records = most_recent_records[most_recent_records['financialDate'] >= pd.to_datetime('2024-04-08')]

most_recent_records.to_csv('final_future_merged.csv', index=False)