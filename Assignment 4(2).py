import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load the data
file_path = r"C:\Users\Ocean\Downloads\turnover.csv"  # Make sure this path is correct
data = pd.read_csv(file_path)

# Create dummy variables for categorical columns
data_dummies = pd.get_dummies(data, drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = data_dummies.drop(columns=['Months_active'])  # Independent variables
y = data_dummies['Months_active']  # Dependent variable

# Step 1: Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Set up the Ridge regression model
ridge_reg = Ridge()

# Step 3: Define the alpha values for grid search
alpha_values = np.logspace(-6, 6, 13)  # Searching over a wide range of alpha values

# Step 4: Set up GridSearchCV with 5-fold cross-validation
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Step 5: Fit the grid search model to find the best alpha
grid_search.fit(X_scaled, y)

# Step 6: Get the best alpha value and corresponding performance (MAD)
best_alpha = grid_search.best_params_['alpha']
best_mad = -grid_search.best_score_

print(f"Best alpha value found: {best_alpha}")
print(f"Best MAD (Mean Absolute Deviation) for Ridge Regression: {best_mad:.2f}")
