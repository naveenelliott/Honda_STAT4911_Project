import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('Detailed Supplier.xlsx')

currencies = df['currency'].value_counts()

currency_count = df.groupby(['currency', 'eqyYear']).size().reset_index(name='count')

currency_count = currency_count.sort_values(['currency', 'eqyYear'], ascending=False).reset_index(drop=True)

currency_count = currency_count[16:]

conversion_values = [32.117, 31.16, 29.813, 27.932, 29.461, 30.8869, 30.1316,
                                   10.756, 10.564, 9.619, 
                                   18.33, 17.733, 20.11, 20.284, 21.466, 19.2472, 19.2247, 18.9093, 18.6895, 15.878,
                                   1364.153, 1306.686, 1291.729, 1144.883,
                                   152.9137, 151.353, 140.511, 131.454, 109.817, 106.725, 110.4201, 112.1405, 105.9237, 97.7027]

# Extend the list with np.nan if necessary
num_rows = len(currency_count)
if len(conversion_values) < num_rows:
    conversion_values_extended = conversion_values + [np.nan] * (num_rows - len(conversion_values))
else:
    conversion_values_extended = conversion_values[:num_rows]

# Add the new column to the DataFrame
currency_count['Other Currency to USD'] = conversion_values_extended

currency_count['USD to Currency'] = 1 / np.array(conversion_values_extended, dtype=float)

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
