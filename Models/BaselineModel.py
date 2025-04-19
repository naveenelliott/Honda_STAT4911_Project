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

merged.drop(columns={'Unnamed: 0', 'X.x', 'Data.Source', 'Group.Classification', 'X.y', 
                     'count', 'USD.to.Other.Currency'}, inplace=True)

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

plotting = merged.copy()

plotting.dropna(subset=['prev_FHR'], inplace=True)

plotting = plotting.sort_values(by='diff_days', ascending=True)


mean_prev_fhr = merged['prev_FHR'].mean()
mean_prev_chs = merged['prev_CHS'].mean()
mean_diff_days = merged['diff_days'].mean()

merged['prev_FHR'] = merged['prev_FHR'].fillna(mean_prev_fhr)
merged['prev_CHS'] = merged['prev_CHS'].fillna(mean_prev_chs)
merged['diff_days'] = merged['diff_days'].fillna(mean_diff_days)


merged['diff_days_category'] = np.where(merged['diff_days'] > 365, 1, 0)

above_1000 = len(merged.loc[merged['diff_days_category'] == 1])

below_1000 = len(merged.loc[merged['diff_days_category'] == 0])

merged_box = merged[['FHR', 'prev_FHR', 'diff_days_category']]

merged_box['FHR_diff'] = merged_box['FHR'] - merged_box['prev_FHR']

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=merged_box,
    x='diff_days_category',     # Binary variable
    y='FHR_diff',
    palette=['#666666', '#BB0000']          # Numeric variable
)

# Optionally, set more readable x-axis labels
plt.xticks([0, 1], ['Below 1500 Days', 'Above 1500 Days'], fontsize=12)

plt.xlabel("Difference in Days", fontsize=15)
plt.ylabel("Change in FHR", fontsize=15)

# Title text using your custom font
font_path = 'C:/Users/Owner/Downloads/SoccermaticsForPython-master/SoccermaticsForPython-master/RussoOne-Regular.ttf'
title = FontProperties(fname=font_path)

# Compute group means for 'distance'
group_means = merged_box.groupby('diff_days_category')['FHR_diff'].median()


group_means = group_means.astype(int)

group_means = round(group_means, 0)

# Determine a small vertical offset for the text label
y_range = merged_box['FHR_diff'].max() - merged_box['FHR_diff'].min()

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
    s = "Change in FHR for Large Difference in Days",
    va = "bottom", ha = "center",
    color = "black", fontproperties=title, fontsize = 15
)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()



# Create the scatterplot
fig, ax = plt.subplots(figsize=(10, 6))
# Draw the regression line without scatter points
sns.regplot(
    x="prev_FHR", 
    y="FHR", 
    data=plotting, 
    scatter=False, 
    line_kws={'color': 'black'}
)

# Overlay the scatter plot with color mapping based on diff_days
sc = ax.scatter(
    plotting['prev_FHR'], 
    plotting['FHR'], 
    c=plotting['diff_days'], 
    cmap='coolwarm', 
    alpha=0.8
)

# Add labels, title, and grid
plt.xlabel("Previous FHR", fontsize=18)
plt.ylabel("FHR", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

fig_text(
    x = 0.5, y = .92, 
    s = "Previous FHR vs Current FHR",  # Use <> around the text to be styled
    va = "bottom", ha = "center",
    color = "black", fontproperties = belanosima, weight = "bold", size=30
)

cbar = plt.colorbar(sc)
cbar.set_label("Difference in Days", fontsize=15)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Display the plot
plt.show()

df_reg = plotting.dropna(subset=['prev_FHR'])

# Define the feature (previous FHR) and target (current FHR)
X = df_reg[['prev_FHR']]
y = df_reg['FHR']

# Optional: split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('R-Squared', r2)
print("Root Mean Squared Error:", rmse)
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)


del merged['CHS'], merged['prev_financialDate']
