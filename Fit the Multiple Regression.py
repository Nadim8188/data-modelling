import pandas as pd
import statsmodels.api as sm

# Load the data (replace with the correct file path if needed)
file_path = r"C:\Users\Ocean\Downloads\turnover.csv"  # Change this if necessary
data = pd.read_csv(file_path)

# Create dummy variables for categorical columns
data_dummies = pd.get_dummies(data, drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = data_dummies.drop(columns=['Months_active'])  # Independent variables
y = data_dummies['Months_active']  # Dependent variable

# Add constant to the independent variables (for the intercept in the regression)
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print("\nOLS Regression Results:")
print(model.summary())

# Extract coefficients and p-values
coefficients = model.params
p_values = model.pvalues

# Identify the variable with the strongest relationship
strongest_variable = coefficients[1:].abs().idxmax()  # Get the variable with the largest absolute coefficient
strongest_coefficient = coefficients[strongest_variable]
strongest_p_value = p_values[strongest_variable]

# Print the strongest relationship and details
print(f"\nThe variable with the strongest relationship with Months_active is {strongest_variable} with a coefficient of {strongest_coefficient:.3f} and a p-value of {strongest_p_value:.3f}")
