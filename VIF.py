import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data (replace with the correct path if needed)
file_path = r"C:\Users\Ocean\Downloads\turnover.csv"
data = pd.read_csv(file_path)

# Ensure all data is numeric (coerce non-numeric values to NaN)
data = data.apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
data = data.dropna()

# Create dummy variables for categorical columns
data_dummies = pd.get_dummies(data, drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = data_dummies.drop(columns=['Months_active'])  # Independent variables
y = data_dummies['Months_active']  # Dependent variable

# Add constant to the independent variables (for the intercept in the regression)
X = sm.add_constant(X)

# Step 2: Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print("\nOLS Regression Results:")
print(model.summary())

# Step 3: Check for multicollinearity using Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print VIF data
print("\nVariance Inflation Factors (VIF):")
print(vif_data)
