# Project Report: Predicting House Prices with Linear Regression

## **Project Objective**
The goal of this project is to learn how to use regression models for predicting numerical values. Here, house prices are predicted based on various features such as size, age of the house, location, and more.

---

## **Steps Performed**

### **1. Data Loading**
- The [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) was used.
- The data was converted into a Pandas DataFrame for easier management and analysis.

### **2. Data Preprocessing**
- The dataset was checked for missing values, and no null values were found.
- The data was standardized using `StandardScaler` to ensure all features have the same scale.

### **3. Data Splitting**
- The data was divided into two sets:
  - **Training Set (80%)**: Used for training the model.
  - **Testing Set (20%)**: Used for evaluating the model's performance.

### **4. Model Training**
- A linear regression model (`LinearRegression`) was trained on the training data.
- The model learned the coefficients and intercept.

### **5. Model Evaluation**
- The following evaluation metrics were calculated:
  - **RMSE**: `0.745` (average prediction error).
  - **R²**: `0.575` (the model explains 57.5% of the data variance).

---

## **Results and Analysis**

### **Results**
- The linear regression model predicted house prices with an average error of `0.745`.
- The R² value (`0.575`) indicates that the model explains approximately 57.5% of the data variance.

### **Model Performance Analysis**
- **Strengths**:
  - The model is simple and fast.
  - Standardizing the data improved the model's performance.
- **Weaknesses**:
  - The R² value shows that the model cannot explain all data variations.
  - The RMSE indicates that the model's error is still significant.

### **Suggestions for Improvement**
1. Use more advanced models like Random Forest or Gradient Boosting.
2. Add new features (e.g., distance from the city center or school quality).
3. Tune hyperparameters to improve performance.
4. Analyze the data for hidden patterns or outliers.

---

## **Conclusion**
This project demonstrated that a linear regression model can serve as a good starting point for predicting house prices. However, for higher accuracy, using more advanced models and improving feature engineering is recommended.