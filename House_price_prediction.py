# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Step 2: Load the dataset
california_data = fetch_california_housing()

# Step 3: Convert the dataset to a Pandas DataFrame for easier manipulation
data = pd.DataFrame(california_data.data, columns=california_data.feature_names)
data['MedHouseVal'] = california_data.target  # Add the target column (median house value)

# Step 4: Display the first few rows of the dataset
data.head()

# Step 1: Check for missing values in the dataset
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Step 2: If there are missing values, decide how to handle them
# For now, let's assume there are no missing values in this dataset

from sklearn.preprocessing import StandardScaler

# Step 1: Initialize the StandardScaler
scaler = StandardScaler()

# Step 2: Fit and transform the feature columns (excluding the target column)
feature_columns = data.columns[:-1]  # All columns except the last one (MedHouseVal)
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Step 3: Display the first few rows of the normalized data
data.head()

