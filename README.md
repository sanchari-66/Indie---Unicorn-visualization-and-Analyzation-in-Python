# Indie---Unicorn-visualization-and-Analyzation-in-Python

Indian Unicorn Startups Analysis & Random Forest Modeling
Project Overview

This project analyzes the Indian Unicorn Startups dataset (2023) and demonstrates data exploration, visualization, and regression modeling using Random Forest. The goal is to predict a company’s valuation based on features such as sector, location, entry valuation, and year of entry.


**Dataset**

Source: **Indian Unicorn Startups dataset (June 2023, updated)**

Key columns:

Company – Name of the company

Sector – Industry sector

Location – Headquarters location

Entry Valuation ($B) – Valuation at the time of entry

Valuation ($B) – Current valuation

Entry – Entry year information



**Dataset preprocessing included:**

Removing spaces and standardizing capitalization for Location

Encoding categorical variables (Sector, Location, Company) into numeric features using one-hot encoding



**Libraries Used**

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn, ggplot2 (R style)

Machine Learning: scikit-learn

RandomForestRegressor – Predict valuations

GridSearchCV – Hyperparameter tuning

train_test_split – Split dataset into training and testing sets

Metrics: r2_score, mean_squared_error



**Workflow***
1. **Data Preprocessing**

Removed spaces and standardized text in Location column.

Extracted Year from Entry column.

Converted categorical columns to numeric using one-hot encoding.

Split data into features (X) and target (y).

2. **Exploratory Data Analysis (EDA)**

Bar plots: Companies per sector, average valuation by sector

Histograms: Companies entering over time

Scatter plots: Current vs. entry valuation (with conditional labeling if valuation doubled or exceeded threshold)

Correlation heatmap: Numerical features correlation.
4. **Model Building**

Random Forest Regressor was trained on the processed data.

Hyperparameter tuning performed using GridSearchCV with parameters:

max_features (like R's mtry)

min_samples_leaf (like R's min.node.size)

Cross-validation (5-fold) was used to select the best hyperparameters.
Best parameters were extracted from grid_search.best_params_.

5. **Model Evaluation**

Predictions made on test data:

final_predictions = grid_search.predict(X_test)


Performance metrics:

RMSE

R² score

Visualized Predicted vs Actual Valuation using scatter plot.

6. **Model Export**

Saved the trained Random Forest model for future use:
Notes


Preprocessing is crucial for Random Forest as categorical variables need numeric encoding.

Cross-validation ensures the model generalizes well to unseen data.

Visualizations help identify patterns such as sectors with higher valuations or trends over time.

**References**

**scikit-learn RandomForestRegressor**

**GridSearchCV Documentation**



+------------------------+
|   Indian Unicorn Data  |
+------------------------+
            |
            v
+------------------------+
|      Data Cleaning     |
| - Remove spaces        |
| - Standardize text     |
| - Extract Year         |
+------------------------+
            |
            v
+------------------------+
|   Feature Encoding     |
| - One-hot encoding     |
| - Categorical → numeric|
+------------------------+
            |
            v
+------------------------+
|   Train-Test Split     |
| - X_train, X_test      |
| - y_train, y_test      |
+------------------------+
            |
            v
+------------------------+
|  Model Selection       |
| - Random Forest        |
| - Hyperparameter Tuning|
|   (GridSearchCV)       |
+------------------------+
            |
            v
+------------------------+
|  Model Evaluation      |
| - Predictions          |
| - RMSE & R²            |
| - Predicted vs Actual  |
+------------------------+
            |
            v
+------------------------+
|  Model Export          |
| - Save as .pkl         |
| - Optional: PMML       |
+------------------------+
