import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
file_path = '/mnt/data/turnover.csv'
data = pd.read_csv(file_path)

# Convert categorical variables to dummy variables
data_dummies = pd.get_dummies(data, columns=['Disciplined', 'Social_drinker', 'Social_smoker'], drop_first=True)

# Separate dependent and independent variables
X = data_dummies.drop(columns=['Months_active'])
y = data_dummies['Months_active']

# Add a constant to the model (for the intercept)
X = add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression summary
print(model.summary())

# Calculate VIF for each independent variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF data
vif_data = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data)
