import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Load the data
file_path = r"C:\Users\Ocean\Downloads\turnover.csv"  # Make sure the file path is correct
data = pd.read_csv(file_path)

# Create dummy variables for categorical columns
data_dummies = pd.get_dummies(data, drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = data_dummies.drop(columns=['Months_active'])  # Independent variables
y = data_dummies['Months_active']  # Dependent variable

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== Step 1: Neural Network with MLPRegressor ====================
# Initialize the MLPRegressor model with the desired hyperparameters
mlp_model = MLPRegressor(hidden_layer_sizes=(512, 512, 512, 512, 512),  # 5 hidden layers with 512 nodes
                         activation='relu',  # ReLU activation function
                         solver='adam',  # Adam optimizer
                         max_iter=100,  # Number of epochs
                         batch_size=16,  # Batch size
                         random_state=42)

# Use KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform 5-fold cross-validation and compute the Mean Absolute Deviation (MAD)
mlp_mad_scores = cross_val_score(mlp_model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')

# Convert negative MAD to positive and calculate the average
average_mlp_mad = -np.mean(mlp_mad_scores)
print(f"Average MAD for Neural Network (MLPRegressor): {average_mlp_mad:.2f}")

# ==================== Step 2: Linear Regression ====================
# Initialize the linear regression model
lin_reg = LinearRegression()

# Evaluate the linear regression model using 5-fold cross-validation with MAE
lin_reg_mad_scores = cross_val_score(lin_reg, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')

# Convert negative MAD to positive and calculate the average
average_lin_reg_mad = -np.mean(lin_reg_mad_scores)
print(f"Average MAD for Linear Regression: {average_lin_reg_mad:.2f}")

# ==================== Comparison ====================
print("\nComparison of Models:")
print(f"Neural Network (MLPRegressor) MAD: {average_mlp_mad:.2f}")
print(f"Linear Regression MAD: {average_lin_reg_mad:.2f}")
