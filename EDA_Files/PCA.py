import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('cleaned_detailed_supplier.csv')

df.fillna(0, inplace=True)

# Identify continuous variables: these are usually numeric types.
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

# Identify categorical variables: these are typically objects or category types.
categorical_vars = df.select_dtypes(exclude=[np.number]).columns.tolist()

cont_data = df[continuous_vars]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cont_data)

scaled_cont_data = pd.DataFrame(scaled_data, columns=continuous_vars, index=cont_data.index)

feature_names = scaled_cont_data.columns

# Instantiate PCA and reduce to 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_cont_data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Instantiate PCA and reduce to 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_cont_data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Print the explained variance ratio for each principal component
print("\nExplained variance ratio:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=feature_names)



top10_PC1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
top10_PC1 = loadings.loc[top10_PC1.index, 'PC1']

top10_PC1.rename(index={'_Ball Winning Central Defender': 'Ball-Winning CB'}, inplace=True)

top10_PC1.index.name = 'Features'

colors = sns.color_palette("coolwarm", len(top10_PC1))

# Plot the top 10 loadings for PC1
fig, ax = plt.subplots(figsize=(10, 6))
top10_PC1.sort_values().plot(kind='barh', color=colors, ax=ax)
plt.title('Top 10 Loadings for PC1 (24% of Variance Explained)')
plt.xlabel('Loading Value')
plt.ylabel('Feature')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

top10_PC2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)
top10_PC2 = loadings.loc[top10_PC2.index, 'PC1']

top10_PC2.rename(index={'_Ball Playing Central Defender': 'Ball-Playing CB',
                        '_Central Creator': 'Central Creator'}, inplace=True)

colors = sns.color_palette("coolwarm", len(top10_PC2))

# Plot the top 10 loadings for PC1
fig, ax = plt.subplots(figsize=(10, 6))
top10_PC2.sort_values().plot(kind='barh', color=colors)
plt.title('Top 10 Loadings for PC2 (19% of Variance Explained)')
plt.xlabel('Loading Value')
plt.ylabel('Feature')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()