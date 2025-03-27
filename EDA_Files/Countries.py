import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from highlight_text import fig_text, ax_text
from matplotlib.font_manager import FontProperties

merged = pd.read_csv('final_merged_updated.csv')

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=merged,
    x='currency',     # Binary variable
    y='FHR',         # Numeric variable
)

plt.xlabel("Difference in Days", fontsize=15)
plt.ylabel("FHR", fontsize=15)

# Title text using your custom font
font_path = 'C:/Users/Owner/Downloads/SoccermaticsForPython-master/SoccermaticsForPython-master/RussoOne-Regular.ttf'
title = FontProperties(fname=font_path)

# Compute group means for 'distance'
group_means = merged.groupby('currency')['FHR'].median()


group_means = group_means.astype(int)

group_means = round(group_means, 0)

# Determine a small vertical offset for the text label
y_range = merged['FHR'].max() - merged['FHR'].min()

# Plot the mean marker and annotate with the mean value
for group, mean_value in group_means.items():
    # Plot a blue diamond marker at the mean
    # Annotate the mean just above the marker
    if group == False:
        ax.text(group, mean_value, f"{mean_value:.0f}", ha='center', va='bottom', color='black', fontsize=12)
    else:
        ax.text(group, mean_value, f"{mean_value:.0f}", ha='center', va='bottom', color='black', fontsize=12)


fig_text(
    x = 0.5, y = .91, 
    s = "Currency and FHR",
    va = "bottom", ha = "center",
    color = "black", fontproperties=title, fontsize = 15
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()