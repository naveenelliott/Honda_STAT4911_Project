import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('Detailed Supplier.xlsx')

currencies = df['currency'].value_counts()

currency_count = df.groupby(['currency', 'eqyYear']).size().reset_index(name='count')

periods = df['period'].value_counts()

na_percentage = df.isna().mean() * 100

df.drop(columns={'href', 'Unnamed: 1', 'Vlookup Supplier Category'}, inplace=True)

ext_income = df.loc[df['netExtraordinaryIncome'] > 0]

cash_flow = df.loc[df['cashFromOperationsFromCashFlow'] > 0]

ext_expense = df.loc[df['extraordinaryExpenses'] > 0]

highly_absent_columns = ['cashFromOperationsFromCashFlow', 'extraordinaryIncome', 'extraordinaryExpenses',
                         'netExtraordinaryIncome', 'shortTermProvisions', 'provisions', 'minorityInterestBalance', 'deferredTaxation']

df.drop(columns=highly_absent_columns, inplace=True)

# Prepare a dictionary to store the results for each column
results = {}

for col in df.columns:
    pct_missing = df[col].isna().mean() * 100  # Percentage of missing values
    # Only compute zeros if the column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        pct_zeros = (df[col] == 0).mean() * 100
        pct_missing_or_zero = (((df[col].isna()) | (df[col] == 0)).mean()) * 100
    else:
        pct_zeros = np.nan
        pct_missing_or_zero = pct_missing  # For non-numeric, it's just missing values
    
    results[col] = {
        'pct_missing': pct_missing,
        'pct_zeros': pct_zeros,
        'pct_missing_or_zero': pct_missing_or_zero
    }

# Convert the results dictionary into a DataFrame for a nicer display
summary_df = pd.DataFrame(results).T

# Compute the correlation matrix (only numeric columns are considered)
corr_matrix = df.corr()

df.to_csv('cleaned_detailed_supplier.csv', index=False)
