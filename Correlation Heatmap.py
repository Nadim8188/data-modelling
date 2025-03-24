import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
data = pd.read_csv('turnover.csv')

# Select relevant columns
columns_of_interest = ['Months_active', 'Distance_from_work', 'Age', 'Children']
data_selected = data[columns_of_interest]

# Compute correlation matrix
corr_matrix = data_selected.corr()

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
