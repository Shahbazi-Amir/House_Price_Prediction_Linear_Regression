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



from sklearn.model_selection import train_test_split

# Step 1: Separate features (X) and target (y)
X = data.drop(columns=['MedHouseVal'])  # Features (all columns except the target)
y = data['MedHouseVal']  # Target (the last column)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



from sklearn.linear_model import LinearRegression

# Step 1: Create an instance of the Linear Regression model
model = LinearRegression()

# Step 2: Train the model using the training data
model.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 4: Display the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 5: Display the first few predicted values
print("First 5 predicted values:", y_pred[:5])



from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# Step 2: Calculate R²
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)


