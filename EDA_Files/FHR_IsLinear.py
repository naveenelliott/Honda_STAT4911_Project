import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from highlight_text import fig_text, ax_text
from matplotlib.font_manager import FontProperties



# loading font for plots
font_path = "C:/Users/Owner/Downloads/SoccermaticsForPython-master/SoccermaticsForPython-master/AccidentalPresidency.ttf"
belanosima = FontProperties(fname=font_path)

merged = pd.read_csv('final_merged_updated.csv')

merged.drop(columns={'Unnamed: 0'}, inplace=True)

merged = merged.sort_values(by=['Supplier.Number','financialDate'])

merged['financialDate'] = pd.to_datetime(merged['financialDate'])

# ----------------------------------------------------------------------------
# 4) Within each supplier, "shift" to get the *next* row’s FHR and date
# ----------------------------------------------------------------------------
merged['prev_FHR'] = merged.groupby('Supplier.Number')['FHR'].shift(1)
merged['prev_CHS'] = merged.groupby('Supplier.Number')['CHS'].shift(1)
merged['prev_financialDate'] = merged.groupby('Supplier.Number')['financialDate'].shift(1)

merged['prev_financialDate'] = pd.to_datetime(merged['prev_financialDate'])
merged['diff_days'] = (merged['financialDate'] - merged['prev_financialDate']).dt.days

merged['diff_FHR'] = merged['FHR'] - merged['prev_FHR']

# Create the scatterplot
fig, ax = plt.subplots(figsize=(10, 6))
# Draw the regression line without scatter points
sns.regplot(
    x="diff_days", 
    y="diff_FHR", 
    data=merged, 
    scatter=False, 
    line_kws={'color': 'black'}
)

# Overlay the scatter plot with color mapping based on diff_days
sc = ax.scatter(
    merged['diff_days'], 
    merged['diff_FHR'],
    color='red',
    alpha=0.8
)

# Add labels, title, and grid
plt.xlabel("Difference in Days", fontsize=18)
plt.ylabel("Difference in FHR", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

fig_text(
    x = 0.5, y = .92, 
    s = "How does FHR Change Over Time?",  # Use <> around the text to be styled
    va = "bottom", ha = "center",
    color = "black", fontproperties = belanosima, weight = "bold", size=30
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Display the plot
plt.show()