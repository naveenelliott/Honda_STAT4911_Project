import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Load data
joined_data = pd.read_csv('final_previous_merged.csv')

joined_data = pd.get_dummies(joined_data, columns=["prev_currency"], prefix="dummy", drop_first=False)

df_model = joined_data.drop(columns={'prev_X.Other.Currency.to.USD', 'prev_inf_factor', 'prev_financialDate'})

# Define feature matrix X and target vector y
X = df_model.drop(columns=['FHR'])
y = df_model['FHR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a range of alphas to test
alphas = [0.05, 0.1, 0.5, 1, 10]

# Perform cross-validation with LassoCV
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# Best alpha found via CV
best_alpha = lasso_cv.alpha_
print("Best alpha from LassoCV:", best_alpha)

# ---- Step 1: LASSO for Feature Selection ----
lasso = Lasso(alpha=0.5, max_iter=10000)  # Adjust alpha to control feature selection
lasso.fit(X_train, y_train)

# Convert the scaled training data back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Now, you can use .columns on X_train_scaled_df
selected_features = X_train_scaled_df.columns[lasso.coef_ != 0].tolist()
print(f'# of Features {len(selected_features)}')

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Reduce feature set
X_train_selected = X_train_scaled_df[selected_features]
X_test_selected = X_test_scaled_df[selected_features]

# ---- Step 2: Train Logistic Regression ----
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("R^2 Score:", r2)

# ---- Step 4: Coefficient Importance ----
coefficients = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.loc[~coefficients['Feature'].str.contains('dummy')]

# Get top features
top_positive = coefficients.sort_values(by='Coefficient', ascending=False).head(10)
top_negative = coefficients.sort_values(by='Coefficient', ascending=True).head(10)
coefficients = pd.concat([top_negative, top_positive]).sort_values(by='Coefficient')

coefficients['Feature'] = coefficients['Feature'].str.replace(r'^prev_', '', regex=True)


# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
sns.barplot(data=coefficients, x='Coefficient', y='Feature', palette='coolwarm')

fig_text(
    x=0.12, y=.92, 
    s="Top Features from LASSO Regression Model",
    va="bottom", ha="left",
    color="black", weight="bold", size=22
)

plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('', fontsize=14)
plt.yticks(fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()