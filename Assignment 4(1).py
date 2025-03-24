import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load the data
file_path = r"C:\Users\Ocean\Downloads\turnover.csv"  # Make sure to set the correct path
data = pd.read_csv(file_path)

# Create dummy variables for categorical columns
data_dummies = pd.get_dummies(data, drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = data_dummies.drop(columns=['Months_active'])  # Independent variables
y = data_dummies['Months_active']  # Dependent variable

# ==================== Step 1: Linear Regression with Cross-Validation ====================
# Initialize the linear regression model
lin_reg = LinearRegression()

# Use KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform 5-fold cross-validation and compute the Mean Absolute Deviation (MAD)
lin_reg_mad_scores = cross_val_score(lin_reg, X, y, cv=kf, scoring='neg_mean_absolute_error')

# Convert negative MAD to positive values and calculate the average
average_lin_reg_mad = -np.mean(lin_reg_mad_scores)
print(f"Average MAD for Linear Regression: {average_lin_reg_mad:.2f}")

