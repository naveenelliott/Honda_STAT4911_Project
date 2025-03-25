import pandas as pd

merged = pd.read_csv('merged.csv')

merged.drop(columns={'Unnamed: 0', 'X.x', 'Data.Source', 'Group.Classification', 'X.y', 
                     'count', 'USD.to.Other.Currency'}, inplace=True)

merged.to_csv('final_merged.csv', index=False)