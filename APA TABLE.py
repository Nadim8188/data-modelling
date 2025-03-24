import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ace_tools as tools

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

# Collect the relevant data for the APA table
coefficients = model.params
std_errors = model.bse
t_values = model.tvalues
p_values = model.pvalues
ci_lower = model.conf_int()[0]
ci_upper = model.conf_int()[1]

# Prepare the data for the APA table
apa_table_data = {
    "Predictor": [
        "Constant", "Distance from Work", "Age", "Children", "Pets", 
        "Weight", "Height", "BMI", "Absent Hours", "Disciplined (Yes)", 
        "Social Drinker (Yes)", "Social Smoker (Yes)"
    ],
    "B": coefficients.round(2),
    "SE B": std_errors.round(2),
    "t": t_values.round(2),
    "p-value": p_values.round(3),
    "95% CI Lower": ci_lower.round(2),
    "95% CI Upper": ci_upper.round(2)
}

# Convert to DataFrame for better visualization
apa_table = pd.DataFrame(apa_table_data)

# Display the table to the user
tools.display_dataframe_to_user(name="APA Table - Regression Results", dataframe=apa_table)

# Output the APA table
print(apa_table)
