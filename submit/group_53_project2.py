#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[31]:


dataset = pd.read_csv("data/cleaned_5250.csv")
missing_values_idx = dataset.isna().any(axis=1)
clean_dataset = dataset[~missing_values_idx]


# In[32]:


num_obeservation = 4000
df = clean_dataset.iloc[range(num_obeservation)]
full_df = df.copy()
df = df.drop("name", axis=1)


# In[33]:


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
print(df, df.shape)


# In[34]:


outlier = df.iloc[np.where(df["mass"] > 10**30)]
df = df.drop(outlier.index)


# In[35]:


planet_type = df["planet_type"]
encoded_df = df.copy()
#we encode the categorical variables, to make it digestable for the training stage later
encoded_df["detection_method"] = encoded_df["detection_method"].astype("category").cat.codes
encoded_df["planet_type"] = encoded_df["planet_type"].astype("category").cat.codes
encoded_df["discovery_year"] = encoded_df["discovery_year"].astype("category").cat.codes

#we have a separate df, so that the standardization is only done for the non-categorical variables
df_without_type = encoded_df.drop(["planet_type", "detection_method", "discovery_year"] , axis=1)


# In[36]:


planet_cats = df["planet_type"].astype("category")
print(planet_cats.cat.categories) 
detection_cats = df["detection_method"].astype("category")
print(detection_cats.cat.categories)
discovery_cats = df["discovery_year"].astype("category")
print(discovery_cats.cat.categories) 


# In[37]:


df_train = df_without_type.copy().drop("orbital_period", axis=1)
df_train["planet_type"] = df["planet_type"]
df_train = pd.get_dummies(df_train, columns=['planet_type'])


# In[38]:


# Define features, and target variable
X = df_train.drop(columns=['mass']).values
y = df_train['mass'].values
print(X.shape, y.shape)


# In[39]:


def standardize_data(X_train_orig, X_test_orig, y_train_orig, y_test_orig, numerical_indices, encoded_indices):

    # Scale Numerical Features
    X_train_numerical = X_train_orig[:, numerical_indices]
    X_test_numerical = X_test_orig[:, numerical_indices]

    feature_scaler = StandardScaler()
    X_train_scaled_numerical = feature_scaler.fit_transform(X_train_numerical)
    X_test_scaled_numerical = feature_scaler.transform(X_test_numerical)

    # Recombine scaled numerical + encoded parts
    X_train_scaled = np.concatenate([X_train_scaled_numerical, X_train_orig[:, encoded_indices]], axis=1)
    X_test_scaled = np.concatenate([X_test_scaled_numerical, X_test_orig[:, encoded_indices]], axis=1)

    # Scale Target Variable
    mu_y = y_train_orig.mean()
    sigma_y = y_train_orig.std()
    epsilon = 1e-8

    y_train_scaled = (y_train_orig - mu_y) / (sigma_y + epsilon)
    y_test_scaled = (y_test_orig - mu_y) / (sigma_y + epsilon)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), y_train_scaled, y_test_scaled, feature_scaler, mu_y, sigma_y


# In[40]:


lambdas = np.logspace(-6, 4, 100)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_errors = []
test_errors = []
coeff = []


# In[41]:


# Define column indices
total_columns = X.shape[1]
numerical_indices = list(range(total_columns - 4))
encoded_indices = list(range(total_columns - 4, total_columns))

# A safety check in case there are no numerical columns
has_numerical_features = len(numerical_indices) > 0

for lam in lambdas:

    fold_train_errors = []
    fold_test_errors = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize processed arrays with the one-hot encoded parts
        X_train, X_test, y_train, y_test, a, b, c = standardize_data(
            X_train, X_test, y_train, y_test,
            numerical_indices, encoded_indices
        )

        # Fit the model on the correctly processed data
        ridge = Ridge(alpha=lam)
        ridge.fit(X_train, y_train)

        y_pred_train = ridge.predict(X_train)
        y_pred_test = ridge.predict(X_test)

        fold_train_errors.append(mean_squared_error(y_train, y_pred_train))
        fold_test_errors.append(mean_squared_error(y_test, y_pred_test))

    train_errors.append(np.mean(fold_train_errors))
    test_errors.append(np.mean(fold_test_errors))
    coeff.append(ridge.coef_)


# In[42]:


#optimal lambda
optimal_idx = np.argmin(test_errors)
optimal_lambda = lambdas[optimal_idx]
optimal_error = test_errors[optimal_idx]
optimal_coeff = coeff[optimal_idx]

print(f"Optimal lambda = {optimal_lambda:.4f}")
print(f"Minimum Cross-Validation error = {optimal_error:.4f}")
print(f"Optimal coefficients = {optimal_coeff}")
plt.figure(figsize=(8, 5))
plt.plot(lambdas, train_errors, marker='o', label='Training error')
plt.plot(lambdas, test_errors, marker='s', label='Validation error')

# Highlight optimal lambda
plt.axvline(optimal_lambda, color='red', linestyle='--', label=f'Optimal lambda = {optimal_lambda:.2f}')
plt.scatter(optimal_lambda, optimal_error, color='red', s=80, zorder=5)

# Labels and legend
plt.xlabel('Regularization parameter lambda')
plt.ylabel('Mean squared error (on standardized y)')
plt.title('Generalization error vs. lambda (10-fold Cross-Validation)')
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


columns_to_log = ["distance", "orbital_period", "orbital_radius", "radius", "mass"]
for col in columns_to_log:
    df[col] = np.log10(df[col])


# In[44]:


planet_type = df["planet_type"]
encoded_df = df.copy()
#we encode the categorical variables, to make it digestable for the training stage later
encoded_df["detection_method"] = encoded_df["detection_method"].astype("category").cat.codes
encoded_df["planet_type"] = encoded_df["planet_type"].astype("category").cat.codes
encoded_df["discovery_year"] = encoded_df["discovery_year"].astype("category").cat.codes

#we have a separate df, so that the standardization is only done for the non-categorical variables
df_without_type = encoded_df.drop(["planet_type", "detection_method", "discovery_year"] , axis=1)


# In[45]:


df_train = df_without_type.copy().drop("orbital_period", axis=1)
df_train["planet_type"] = df["planet_type"]
df_train = pd.get_dummies(df_train, columns=['planet_type'])


# In[46]:


# Define features, and target variable
X = df_train.drop(columns=['mass']).values
y = df_train['mass'].values
print(X.shape, y.shape)


# In[47]:


lambdas = np.logspace(-6, 2, 100)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_errors = []
test_errors = []
coeff = []


# In[48]:


# Define column indices
total_columns = X.shape[1]
numerical_indices = list(range(total_columns - 4))
encoded_indices = list(range(total_columns - 4, total_columns))

# A safety check in case there are no numerical columns
has_numerical_features = len(numerical_indices) > 0

for lam in lambdas:

    fold_train_errors = []
    fold_test_errors = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize processed arrays with the one-hot encoded parts
        X_train, X_test, y_train, y_test, a, b, c = standardize_data(
            X_train, X_test, y_train, y_test,
            numerical_indices, encoded_indices
        )

        # Fit the model on the correctly processed data
        ridge = Ridge(alpha=lam)
        ridge.fit(X_train, y_train)

        y_pred_train = ridge.predict(X_train)
        y_pred_test = ridge.predict(X_test)

        fold_train_errors.append(mean_squared_error(y_train, y_pred_train))
        fold_test_errors.append(mean_squared_error(y_test, y_pred_test))

    train_errors.append(np.mean(fold_train_errors))
    test_errors.append(np.mean(fold_test_errors))
    coeff.append(ridge.coef_)


# In[49]:


#optimal lambda
optimal_idx = np.argmin(test_errors)
optimal_lambda = lambdas[optimal_idx]
optimal_error = test_errors[optimal_idx]
optimal_coeff = coeff[optimal_idx]

print(f"Optimal lambda = {optimal_lambda:.4f}")
print(f"Minimum Cross-Validation error = {optimal_error:.4f}")
print(f"Optimal coefficients = {optimal_coeff}")

plt.figure(figsize=(8, 5))
plt.plot(lambdas, train_errors, marker='o', label='Training error')
plt.plot(lambdas, test_errors, marker='s', label='Validation error')

# Highlight optimal lambda
plt.axvline(optimal_lambda, color='red', linestyle='--', label=f'Optimal lambda = {optimal_lambda:.2f}')
plt.scatter(optimal_lambda, optimal_error, color='red', s=80, zorder=5)

# Labels and legend
plt.xlim(-1, 50)
plt.xlabel('Regularization parameter lambda')
plt.ylabel('Mean squared error (on standardized and log10 transformed y)')
plt.title('Generalization error vs. lambda (10-fold Cross-Validation)')
plt.legend()
plt.grid(True)
plt.show()


# In[51]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the column indices for numerical and encoded features
total_columns = X.shape[1]
numerical_indices = list(range(total_columns - 4))
encoded_indices = list(range(total_columns - 4, total_columns))

# Isolate the numerical parts from the new splits
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, mu_y, sigma_y = standardize_data(
    X_train_orig, X_test_orig, y_train_orig, y_test_orig,
    numerical_indices, encoded_indices
)


print("--- Final Model Training ---")
print(f"Using optimal lambda: {optimal_lambda}")

# Initialize the model with the best lambda
ridge = Ridge(alpha=optimal_lambda)
ridge.fit(X_train_scaled, y_train_scaled)

# Store the final coefficients and intercept
coeffs = ridge.coef_
intercept = ridge.intercept_

# Make predictions on the scaled training and test sets
y_pred_train_scaled = ridge.predict(X_train_scaled)
y_pred_test_scaled = ridge.predict(X_test_scaled)

train_error_scaled = mean_squared_error(y_train_scaled, y_pred_train_scaled)
test_error_scaled = mean_squared_error(y_test_scaled, y_pred_test_scaled)

print(f"\nFinal training MSE (on scaled data): {train_error_scaled:.4f}")
print(f"Final test MSE (on scaled data): {test_error_scaled:.4f}")
print(f"\nModel Coefficients: {coeffs}")
print(f"Model Intercept: {intercept}")


# In[52]:


# Feature names for regression (excluding 'mass' which is the target)
feature_names_planets = ['distance', 'stellar_magnitude', 'orbital_radius', 'eccentricity', 'radius', 'planet_type_Gas Giant', 'planet_type_Neptune-like', 'planet_type_SuperEarth', 'planet_type_Terrestrial']
#feature_names = ['distance', 'stellar_magnitude', 'orbital_radius', 'eccentricity', 'radius']

# Create DataFrame for coefficients
coeffs_df = pd.DataFrame({
    'Feature': feature_names_planets,
    'Coefficient': coeffs
}).sort_values('Coefficient', key=abs, ascending=False)

print("Ridge Regression Coefficients:")
print(coeffs_df.to_string(index=False))
print(f"\nIntercept: {intercept:.4f}")
print(f"Train Error (MSE): {train_error_scaled:.4f}")
print(f"Test Error (MSE): {test_error_scaled:.4f}")

# Visualize coefficients
plt.figure(figsize=(10, 6))
plt.barh(coeffs_df['Feature'], coeffs_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title(f'Ridge Regression Coefficients (λ = {optimal_lambda:.4f})')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Display the DataFrame
coeffs_df


# In[53]:


K1 = 10  # outer folds
K2 = 10  # inner folds

# Regularization and ANN hyperparameters
lambdas = np.logspace(-6, 2, 50)
hidden_units = [1, 8, 16, 32, 64, 128]


# In[ ]:


# Storage
test_errors_outer = {
    'baseline': np.zeros(K1),
    'ridge': np.zeros(K1),
    'ann': np.zeros(K1),
}
y_true = []
y_preds = {
    'baseline': [],
    'ridge': [],
    'ann': [],
}


# In[55]:


optimal_hs = np.zeros(K1)
optimal_hs_deep = np.zeros(K1)
optimal_lambdas = np.zeros(K1)


# In[56]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PredictorANN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(PredictorANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.model(x)


# In[58]:


def train_custom_ann(X_train, y_train, X_val, y_val, hidden_units, 
                      lr=1e-8, weight_decay=0.0, epochs=10, batch_size=32, verbose=False, deep=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert numpy arrays to torch tensors

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    # Datasets and loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, optimizer

    model = PredictorANN(X_train.shape[1], hidden_units).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        step_val_loss = criterion(model(X_val_t), y_val_t).item()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {step_val_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()

    # print(f"Validation Loss: {val_loss:.4f}")
    return val_loss, model


# In[59]:


# Outer cross-validation loop
from sklearn.neural_network import MLPRegressor


outer_cv = KFold(K1, shuffle=True, random_state=42)

for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"Outer fold {outer_fold + 1}/{K1}")

    X_train_outer, y_train_outer = X[outer_train_idx], y[outer_train_idx]
    X_test_outer, y_test_outer = X[outer_test_idx], y[outer_test_idx]

    # Standardize (based on training data)
    X_train_outer, X_test_outer, y_train_outer, y_test_outer, feature_scaler, mu_y, sigma_y = standardize_data(
        X_train_outer, X_test_outer, y_train_outer, y_test_outer,
        numerical_indices, encoded_indices
    )

    # BASELINE MODEL
    y_pred_baseline = np.full_like(y_test_outer, np.mean(y_train_outer))
    test_errors_outer['baseline'][outer_fold] = np.mean((y_test_outer - y_pred_baseline) ** 2)
    y_preds['baseline'].append(y_pred_baseline)

    #INNER CV for Ridge
    inner_cv = KFold(K2, shuffle=True, random_state=42)
    ridge_val_errors = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        inner_errors = []
        for train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_idx], X_train_outer[val_idx]
            y_train_inner, y_val_inner = y_train_outer[train_idx], y_train_outer[val_idx]

            model = Ridge(alpha=lam)
            model.fit(X_train_inner, y_train_inner)
            y_val_pred = model.predict(X_val_inner)
            inner_errors.append(mean_squared_error(y_val_inner, y_val_pred))
        ridge_val_errors[i] = np.mean(inner_errors)

    optimal_lambda = lambdas[np.argmin(ridge_val_errors)]
    optimal_lambdas[outer_fold] = optimal_lambda

    # Train Ridge on full outer training set
    ridge_model = Ridge(alpha=optimal_lambda)
    ridge_model.fit(X_train_outer, y_train_outer)
    ridge_pred = ridge_model.predict(X_test_outer)
    ridge_test_error = np.mean((y_test_outer - ridge_pred) ** 2)
    test_errors_outer['ridge'][outer_fold] = ridge_test_error
    y_preds['ridge'].append(ridge_pred)

    # ----- INNER CV for ANN (PyTorch version) -----
    ann_val_errors = np.zeros(len(hidden_units))
    ann_val_errors_deep = np.zeros(len(hidden_units))

    for j, h in enumerate(hidden_units):
        inner_errors = []
        inner_errors_deep = []
        for train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_idx], X_train_outer[val_idx]
            y_train_inner, y_val_inner = y_train_outer[train_idx], y_train_outer[val_idx]

            val_loss, _ = train_custom_ann(X_train_inner, y_train_inner, X_val_inner, y_val_inner, hidden_units=h, 
                      lr=1e-3,epochs=20, batch_size=32, deep=False)
            inner_errors.append(val_loss)

            val_loss, _ = train_custom_ann(X_train_inner, y_train_inner, X_val_inner, y_val_inner, hidden_units=h, 
                      lr=1e-3,epochs=20, batch_size=32, deep=True)
            inner_errors_deep.append(val_loss)

        ann_val_errors[j] = np.mean(inner_errors)
        ann_val_errors_deep[j] = np.mean(inner_errors_deep)

    optimal_h = hidden_units[np.argmin(ann_val_errors)]
    optimal_hs[outer_fold] = optimal_h

    optimal_h_deep = hidden_units[np.argmin(ann_val_errors_deep)]
    optimal_hs_deep[outer_fold] = optimal_h_deep

    # Train final ANN model on full outer training set
    # _, ann_model = train_custom_ann(X_train_outer, y_train_outer, X_test_outer, y_test_outer,
    #                                 hidden_units=optimal_h, lr=1e-3, epochs=10)
    _, ann_model = train_custom_ann(X_train_outer, y_train_outer, X_test_outer, y_test_outer, hidden_units=optimal_h, 
                      lr=1e-3,epochs=20, batch_size=32, deep=False)
    ann_model.eval()

    device = next(ann_model.parameters()).device  # get model’s device
    X_test_t = torch.tensor(X_test_outer, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_outer.reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_t = ann_model(X_test_t).cpu().numpy().flatten()

    ann_test_error = mean_squared_error(y_test_outer, y_pred_t)
    test_errors_outer['ann'][outer_fold] = ann_test_error
    y_preds['ann'].append(y_pred_t)

    y_true.append(y_test_outer)


# Results summary
print("\nMean test errors across outer folds:")
for model_name, errors in test_errors_outer.items():
    print(f"{model_name:10s}: {np.mean(errors):.4f} ± {np.std(errors):.4f}")


# In[60]:


y_true = np.concatenate(y_true)
y_preds_concat = {model: np.concatenate(model_preds) for model, model_preds in y_preds.items()}


# In[61]:


# Summarize Results in Table
results_df = pd.DataFrame({
    'Fold': np.arange(1, K1 + 1),
    'lambda* (Ridge)': optimal_lambdas,
    'h* (ANN)': optimal_hs,
    'Baseline Test Error': test_errors_outer['baseline'],
    'Ridge Test Error': test_errors_outer['ridge'],
    'ANN Test Error': test_errors_outer['ann'],
})

# Display the table
print("\n===== Table 1: Cross-Validation Results =====")
print(results_df.to_string(index=False))

# Summary statistics
print("\nMean ± Std of Test Errors across folds:")
for model_name, errors in test_errors_outer.items():
    print(f"{model_name:10s}: {np.mean(errors):.4f} ± {np.std(errors):.4f}")


# In[62]:


import scipy.stats as st
def confidence_interval_comparison(y_true, y_preds_A, y_preds_B, loss_fn, alpha=0.05):
    z = loss_fn(y_true, y_preds_A) - loss_fn(y_true, y_preds_B)
    z_hat = np.mean(z)
    n = len(y_true)
    nu = n - 1  # degrees of freedom
    sem = np.sqrt(sum(((z - z_hat)**2) / (n * nu))) # or st.sem(loss_fn(y_true, y_preds))
    CI = st.t.interval(1 - alpha, df=nu, loc=z_hat, scale=sem)  # Confidence interval
    t_stat = -np.abs(np.mean(z)) / st.sem(z)
    p_value = 2 * st.t.cdf(t_stat, df=nu)  # p-value

    return z_hat, CI, p_value


# In[63]:


l2_loss = lambda y, y_pred: (y - y_pred)**2
alpha = 0.05

setup1_storage = {
    'ridge_ann': [],
    'ann_baseline': [],
    'ridge_baseline': []
}

z_hat, CI, p_value = confidence_interval_comparison(y_true, y_preds_concat["ridge"], y_preds_concat["ann"], l2_loss, alpha=alpha)
setup1_storage['ridge_ann'].append((z_hat, CI, p_value))
print(f"Difference in loss between Ridge and ANN: \nz_hat: {z_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p_value}")

z_hat, CI, p_value = confidence_interval_comparison(y_true, y_preds_concat["ridge"], y_preds_concat["baseline"], l2_loss, alpha=alpha)
setup1_storage['ridge_baseline'].append((z_hat, CI, p_value))
print(f"Difference in loss between Ridge and Baseline: \nz_hat: {z_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p_value}")

z_hat, CI, p_value = confidence_interval_comparison(y_true, y_preds_concat["ann"], y_preds_concat["baseline"], l2_loss, alpha=alpha)
setup1_storage['ann_baseline'].append((z_hat, CI, p_value))
print(f"Difference in loss between ANN and Baseline: \nz_hat: {z_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p_value}")


# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[44]:


dataset = pd.read_csv("data/cleaned_5250.csv")
missing_values_idx = dataset.isna().any(axis=1)
clean_dataset = dataset[~missing_values_idx]
clean_dataset.shape


# In[45]:


num_obeservation = 4000
df = clean_dataset.iloc[range(num_obeservation)]
full_df = df.copy()
df = df.drop("name", axis=1)


# In[46]:


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
print(df, df.shape)


# In[47]:


columns_to_log = ["distance", "orbital_period", "orbital_radius", "radius", "mass"]
for col in columns_to_log:
    df[col] = np.log10(df[col])


# In[48]:


planet_type = df["planet_type"]
encoded_df = df.copy()
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
df_std


# In[49]:


planet_cats = df["planet_type"].astype("category")
print(planet_cats.cat.categories) 
detection_cats = df["detection_method"].astype("category")
print(detection_cats.cat.categories)
discovery_cats = df["discovery_year"].astype("category")
print(discovery_cats.cat.categories) 


# In[50]:


class_y_true = []
class_y_preds = {
    'baseline': [],
    'logistic_regression': [],
    'KNN': []
}


# In[51]:


# features / labels (already defined above but repeated here for clarity)
X = encoded_df.drop(columns=['planet_type']).values
y = encoded_df["planet_type"].values

# 10-fold outer CV to evaluate the majority-class baseline
K_out = 10
outer_cv = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=42)

baseline_misclass = np.zeros(K_out)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # find most frequent planet type in training set
    # y are encoded integer labels so np.bincount works
    most_freq = np.bincount(y_train).argmax()

    # predict that class for all test samples
    y_pred = np.full_like(y_test, fill_value=most_freq)

    # compute misclassification percentage
    misclass_pct = 100.0 * np.mean(y_pred != y_test)
    baseline_misclass[fold_idx] = misclass_pct

    class_y_true.append(y_test)
    class_y_preds['baseline'].append(y_pred)

    print(f"Fold {fold_idx+1}/{K_out}: majority class = {most_freq}, test misclassification = {misclass_pct:.2f}%")


# In[52]:


# features / labels
X = encoded_df.drop(columns=['planet_type', 'discovery_year']).values
y = encoded_df["planet_type"].values

# Nested cross-validation parameters
lambdas = np.logspace(-6, 4, 200)   # inverse-regularization grid for selection (lambda)
K_out = 10
K_in  = 10
outer_cv = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=K_in, shuffle=True, random_state=1)

outer_test_acc = np.zeros(K_out)
outer_test_misclass = np.zeros(K_out)
selected_lambda = np.zeros(K_out)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # standardize based on outer training fold
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    # inner CV to pick best lambda
    mean_scores = []
    for lam in lambdas:
        model = LogisticRegression(penalty='l2', C=1/lam, max_iter=1000)
        scores = cross_val_score(model, X_train_std, y_train, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        mean_scores.append(scores.mean())

    best_idx = int(np.argmax(mean_scores))
    optimal_lambda = lambdas[best_idx]
    selected_lambda[fold_idx] = optimal_lambda

    # retrain on full outer training with optimal lambda and evaluate on outer test set
    final_model = LogisticRegression(penalty='l2', C=1/optimal_lambda, max_iter=1000)
    final_model.fit(X_train_std, y_train)
    y_test_pred = final_model.predict(X_test_std)
    outer_test_acc[fold_idx] = accuracy_score(y_test, y_test_pred)
    outer_test_misclass[fold_idx] = 100.0 * np.mean(y_test_pred != y_test)

    class_y_preds['logistic_regression'].append(y_test_pred)

    print(f"Outer fold {fold_idx+1}/{K_out}: optimal lambda={optimal_lambda:.3g}, test acc={outer_test_acc[fold_idx]:.4f}, misclass={outer_test_misclass[fold_idx]:.2f}%")

print(f"\nNested CV test accuracy: mean={outer_test_acc.mean():.4f}, std={outer_test_acc.std():.4f}")
print("Selected lambda per outer fold:",
      np.array2string(selected_lambda,
                      formatter={'float_kind': lambda x: f"{x:.3g}"},
                      separator=", "))


# In[53]:


print("Selected lambda per outer fold:",
      np.array2string(selected_lambda,
                      formatter={'float_kind': lambda x: f"{x:.3g}"},
                      separator=", "))


# In[54]:


# features / labels (already defined above but repeated here for clarity)
X = encoded_df.drop(columns=['planet_type', "discovery_year"]).values
y = encoded_df["planet_type"].values

# Nested (two-level) CV with 10x10 to estimate generalization and select k (KNN) per outer fold
ks = list(range(1,20 , 1))   # odd k values to avoid ties
K_out = 10
K_in  = 10
outer_cv = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=K_in, shuffle=True, random_state=1)

outer_test_acc = np.zeros(K_out)
outer_test_misclass = np.zeros(K_out)
selected_k = np.zeros(K_out, dtype=int)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # standardize based on outer training fold
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    # avoid division by zero
    sigma[sigma == 0] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    # inner CV to pick best k
    mean_scores = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(model, X_train_std, y_train, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        mean_scores.append(scores.mean())

    best_idx = int(np.argmax(mean_scores))
    best_k = ks[best_idx]
    selected_k[fold_idx] = best_k

    # retrain on full outer training with best_k and evaluate on outer test set
    final_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    final_model.fit(X_train_std, y_train)
    y_test_pred = final_model.predict(X_test_std)
    outer_test_acc[fold_idx] = accuracy_score(y_test, y_test_pred)
    outer_test_misclass[fold_idx] = 100.0 * np.mean(y_test_pred != y_test)

    class_y_preds['KNN'].append(y_test_pred)

    print(f"Outer fold {fold_idx+1}/{K_out}: best k={best_k}, test acc={outer_test_acc[fold_idx]:.4f}, misclass={outer_test_misclass[fold_idx]:.2f}%")

print(f"\nNested CV test accuracy: mean={outer_test_acc.mean():.4f}, std={outer_test_acc.std():.4f}")
print("Selected k per outer fold:", selected_k)


# In[55]:


print(df["planet_type"].value_counts())
print(df["planet_type"].value_counts().idxmax())
print(encoded_df["planet_type"].value_counts())
print(encoded_df["planet_type"].value_counts().idxmax())


# In[56]:


def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    nn = np.zeros((2, 2))
    cA = yhatA == y_true
    cB = yhatB == y_true
    nn[0, 0] = sum([cA[i] * cB[i] for i in range(len(cA))]) 
    nn[0, 1] = sum(cA & ~cB)
    nn[1, 0] = sum(~cA & cB)
    nn[1, 1] = sum(~cA & ~cB)
    n = len(y_true)
    n12 = nn[0, 1]
    n21 = nn[1, 0]
    E_theta = (n12 - n21) / n
    Q = (
        n**2
        * (n + 1)
        * (E_theta + 1)
        * (1 - E_theta)
        / ((n * (n12 + n21) - (n12 - n21) ** 2))
    )
    f = (E_theta + 1)/2 * (Q - 1)
    g = (1 - E_theta)/2 * (Q - 1)
    CI = tuple(bound * 2 - 1 for bound in st.beta.interval(1 - alpha, a=f, b=g))
    p = 2 * st.binom.cdf(min([n12, n21]), n=n12 + n21, p=0.5)

    print(f"Result of McNemars test using alpha = {alpha}\n")
    print("Contingency table")
    print(nn, "\n")
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print(f"Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = {CI[0]:.4f}, {CI[1]:.4f}\n")
    print(
        f"p-value for two-sided test A and B have same accuracy (exact binomial test): p={p}\n"
    )

    return E_theta, CI, p


# In[57]:


class_y_true = np.concatenate(class_y_true)
class_y_preds_concat = {model: np.concatenate(model_preds) for model, model_preds in class_y_preds.items()}


# In[58]:


l2_loss = lambda y, y_pred: (y - y_pred)**2
alpha = 0.05

setup1_class_storage = {
    'logistic_regression_KNN': [],
    'KNN_baseline': [],
    'logistic_regression_baseline': []
}

[theta_hat, CI, p] = mcnemar(class_y_true, class_y_preds_concat["logistic_regression"], class_y_preds_concat["KNN"], alpha=alpha)
setup1_class_storage['logistic_regression_KNN'].append((theta_hat, CI, p))
print(f"Difference in loss between Logistic Regression and KNN: \ntheta_hat: {theta_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p}")

[theta_hat, CI, p] = mcnemar(class_y_true, class_y_preds_concat["KNN"], class_y_preds_concat["baseline"], alpha=alpha)
setup1_class_storage['KNN_baseline'].append((theta_hat, CI, p))
print(f"Difference in loss between KNN and Baseline: \ntheta_hat: {theta_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p}")

[theta_hat, CI, p] = mcnemar(class_y_true, class_y_preds_concat["logistic_regression"], class_y_preds_concat["baseline"], alpha=alpha)
setup1_class_storage['logistic_regression_baseline'].append((theta_hat, CI, p))
print(f"Difference in loss between Logistic Regression and Baseline: \ntheta_hat: {theta_hat:.4f}, \nCI: [{CI[0]:.4f}, {CI[1]:.4f}], \np-value: {p}")


# In[59]:


# Use the optimal lambda from the last outer cv fold as the final lambda
optimal_lambda = selected_lambda[-1]
print(f"Using optimal lambda (from last outer CV fold): {optimal_lambda:.4f}")

# Prepare full dataset
X_full = encoded_df.drop(columns=['planet_type', 'discovery_year']).values
y_full = encoded_df["planet_type"].values

# Standardize features
mu_full = np.mean(X_full, axis=0)
sigma_full = np.std(X_full, axis=0)
X_full_std = (X_full - mu_full) / sigma_full

# Train final logistic regression model on full dataset
final_logistic_model = LogisticRegression(
    penalty='l2',
    C=1/optimal_lambda,
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)
final_logistic_model.fit(X_full_std, y_full)


# In[60]:


# Get feature names
feature_names = encoded_df.drop(columns=['planet_type', 'discovery_year']).columns.tolist()

# Get class names
planet_type_cat = df["planet_type"].astype("category")
class_names = list(planet_type_cat.cat.categories)
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# Get coefficients for each class
coefficients = final_logistic_model.coef_ 
intercepts = final_logistic_model.intercept_

print(f"\nCoefficient matrix shape: {coefficients.shape}")
print(f"Features: {feature_names}")

# Analyze feature importance across all classes
# Use mean absolute coefficient as overall importance measure
feature_importance = np.mean(np.abs(coefficients), axis=0)
print(coefficients)


# In[65]:


# Create DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': feature_importance
}).sort_values('Coefficient', ascending=False)

print(importance_df.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'])
plt.xlabel('Mean Absolute Coefficient', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance for Planet Type Classification', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[66]:


# Show coefficients for each class
for class_idx, class_name in enumerate(class_names):
    print(f"\nClass {class_idx}: {class_name}")
    class_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients[class_idx, :]
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(class_coefs.to_string(index=False))

# Visualize coefficients heatmap
plt.figure(figsize=(12, 8))
plt.imshow(coefficients.T, aspect='auto', cmap='RdBu_r', vmin=-coefficients.max(), vmax=coefficients.max())
plt.colorbar(label='Coefficient Value')
plt.yticks(range(len(feature_names)), feature_names, fontsize=16)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=16)
plt.xlabel('Planet Type')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients Heatmap')
plt.tight_layout()
plt.show()

