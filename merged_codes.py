### The following code is merged from three jupyter notebooks where we worked on the project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = pd.read_csv("data/cleaned_5250.csv")
missing_values_idx = dataset.isna().any(axis=1)
clean_dataset = dataset[~missing_values_idx]
clean_dataset.shape

num_obeservation = 4000
df = clean_dataset.iloc[range(num_obeservation)]
full_df = df.copy()
df = df.drop("name", axis=1)

# mass transformation: The dataset contains a mass calculation based on two planets. We unify that into a single mass variable
jupiter_mass_kg = 1.898 * 10**27 #kg
jupiter_radius_km = 69911 #km
earth_mass_kg = 5.972 * 10**24
earth_radius_km = 6378
df["mass_wrt"] = np.where(df["mass_wrt"] == "Jupiter", jupiter_mass_kg, earth_mass_kg)
df["mass"] = np.multiply(df["mass_multiplier"], df["mass_wrt"])
df["radius_wrt"] = np.where(df["radius_wrt"] == "Jupiter", jupiter_radius_km, earth_radius_km)
df["radius"] = np.multiply(df["radius_multiplier"], df["radius_wrt"])
df = df.drop(["mass_wrt", "radius_wrt", "mass_multiplier", "radius_multiplier"], axis=1)

transformed_df = df.copy()
columns_to_log = ["distance", "orbital_period", "orbital_radius", "radius", "mass"]
for column in columns_to_log:
    transformed_df[column] = np.log10(transformed_df[column])

raw_df_wothout_type = df.copy()
raw_df_wothout_type = raw_df_wothout_type.drop(["planet_type", "detection_method", "discovery_year"] , axis=1)
encoded_df = transformed_df.copy()
#encoded_df = df.copy()
#we encode the categorical variables, to make it digestable for the training stage later
encoded_df["detection_method"] = encoded_df["detection_method"].astype("category").cat.codes
encoded_df["planet_type"] = encoded_df["planet_type"].astype("category").cat.codes
encoded_df["discovery_year"] = encoded_df["discovery_year"].astype("category").cat.codes

#we have a separate df, so that the standardization is only done for the non-categorical variables
df_without_type = encoded_df.drop(["planet_type", "detection_method", "discovery_year"] , axis=1)

df_std = (df_without_type - np.mean(df_without_type, axis=0)) / np.std(df_without_type, axis=0)
print(np.std(df_std, axis=0))

#we add back everything
df_std["planet_type"] = encoded_df["planet_type"]
df_std["detection_method"] = encoded_df["detection_method"]
df_std["discovery_year"] = encoded_df["discovery_year"]


planet_cats = df["planet_type"].astype("category")
print(planet_cats.cat.categories) 
detection_cats = df["detection_method"].astype("category")
print(detection_cats.cat.categories)
discovery_cats = df["discovery_year"].astype("category")
print(discovery_cats.cat.categories) 

X = df_std.drop(["planet_type", "mass"], axis=1)
y_classification = df_std["planet_type"]
y_regression = df_std["mass"]

# PCA analysis
pca = PCA()

X_pca = X.drop(["detection_method", "discovery_year"], axis=1)

B = pca.fit_transform(X_pca)
V = pca.components_.T

rho = pca.explained_variance_ratio_

threshold = 0.90


fig, axs = plt.subplots(1, 2, figsize=(20, 5))

axs[0].plot(range(1, len(rho)+1), rho, "x-")
axs[0].plot(range(1, len(rho)+1), np.cumsum(rho), "o-")
axs[0].plot([1, len(rho)], [threshold, threshold], "k--")
axs[0].set_title("Variance explained by principal components")
axs[0].set_xlabel("Principal component")
axs[0].set_ylabel("Variance explained")
axs[0].legend(["Individual", "Cumulative", "Threshold"])
bw = 0.2

for i, val in enumerate(rho):
    axs[0].text(
        i + 1, 
        val + 0.02, 
        f'{val:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9
    )

r = np.arange(1, X_pca.shape[1] + 1)
axs[1].set_title("PCA Component Coefficients")
for i, pc in enumerate(V[:, :4].T):
    axs[1].bar(r + i * bw, pc, width=bw, label=f"PC{i+1}")
axs[1].set_xticks(r + bw, X_pca.columns)
axs[1].set_xlabel("Attributes")
axs[1].set_ylabel("Component coefficients")
axs[1].legend()
axs[1].grid()

for i, val in enumerate(np.cumsum(rho)):
    axs[0].text(
        i + 1, 
        val + 0.02, 
        f'{val:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9
    )
plt.show()

df_raw = df.copy()
df_continous = df_std.drop(["planet_type", "detection_method", "discovery_year"], axis=1)
df_categorial = df_std[["planet_type", "detection_method", "discovery_year"]]

# Plot a boxplot of the attributes
raw_df_wothout_type.plot(kind='box', subplots=True, layout=(2, 4), figsize=(16,7), sharex=False, sharey=False)
plt.show()

candidate_1 = full_df.iloc[np.where(raw_df_wothout_type["orbital_radius"] > 6000)]
candidate_2 = full_df.iloc[np.where(raw_df_wothout_type["mass"] > 10**30)]

# get correlation from the unstandardized data. Correlation = covariance of the standardized data
cor_pairs = raw_df_wothout_type.corr()
cor_pairs

import seaborn as sns
sns.heatmap(cor_pairs, annot=True, fmt=".2f", cmap="coolwarm")

log10_cols = ["mass", "radius", "orbital_radius", "orbital_period", "distance", "eccentricity"]
# log10_cols = []

numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    data = df[col].dropna()
    if col in log10_cols:
        data = data[data > 0]
        data = np.log10(data)
        xlabel = f"log10({col})"
        title = f"Histogram of log10({col})"
    else:
        xlabel = col
        title = f"Histogram of {col}"
    if col == "discovery_year":
        sns.histplot(data, bins=np.arange(int(data.min()), int(data.max()) + 1) - 0.5, edgecolor='black', alpha=0.7, ax=ax)
    elif col == "eccentricity":
        sns.histplot(data, bins=30, kde=True, edgecolor='black', alpha=0.7, ax=ax)
    else:
        sns.histplot(data, bins=100, kde=True, edgecolor='black', alpha=0.7, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)

plt.tight_layout()
plt.show()

# Plot histogram of eccentricity (linear scale)
plt.figure(figsize=(8, 5))
sns.histplot(df["eccentricity"].dropna(), bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("eccentricity")
plt.ylabel("Frequency")
plt.title("Histogram of eccentricity (linear scale)")
plt.show()

numzero = np.count_nonzero(df["eccentricity"] == 0)
print(f"Number of zero eccentricity values: {numzero}")
print(f"Total number of values: {df['eccentricity'].shape[0]}")

pairplot_df = df.copy()
log10_cols = ["mass", "radius", "orbital_radius", "orbital_period", "distance"]

# Apply log10 transformation to log-normal columns (avoid <=0 values)
for col in log10_cols:
    pairplot_df[col] = pairplot_df[col].apply(lambda x: np.log10(x) if x > 0 else np.nan)

cols_to_plot = log10_cols + ["planet_type", "stellar_magnitude"]

sns.pairplot(pairplot_df[cols_to_plot], hue="planet_type", diag_kind="kde", plot_kws={"s":5})
plt.suptitle("Pairplot of log10-transformed attributes colored by planet type", y=1.02)
plt.show()