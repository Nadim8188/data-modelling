import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('turnover.csv')

# Select the relevant columns
columns_of_interest = ['Months_active', 'Distance_from_work', 'Age', 'Children']
data_selected = data[columns_of_interest]

# Create scatterplot matrix
sns.pairplot(data_selected)
plt.suptitle("Scatterplot Matrix of Employee Data", y=1.02)  # Add title
plt.show()
