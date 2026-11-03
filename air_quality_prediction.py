# =====================================================
# CA2 MACHINE LEARNING PROJECT
# AIR QUALITY PREDICTION USING 5 REGRESSION MODELS
# Dataset: CPCB Tamil Nadu 2015
# Target: RSPM/PM10
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: Load dataset
df = pd.read_csv("cpcb_dly_aq_tamil_nadu-2015.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# STEP 2: Data Preprocessing
df['Sampling Date'] = pd.to_datetime(
    df['Sampling Date'], dayfirst=True, errors='coerce'
)

columns = ['SO2', 'NO2', 'RSPM/PM10']

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['RSPM/PM10', 'Sampling Date'])

df['SO2'] = df['SO2'].fillna(df['SO2'].median())
df['NO2'] = df['NO2'].fillna(df['NO2'].median())

df['Month'] = df['Sampling Date'].dt.month

print("\nCleaned Dataset Shape:")
print(df.shape)


# STEP 3: Sort data by date and split chronologically
df = df.sort_values('Sampling Date')

split_point = int(len(df) * 0.80)

train_data = df.iloc[:split_point]
test_data = df.iloc[split_point:]

X_train_simple = train_data[['SO2']]
X_test_simple = test_data[['SO2']]

X_train = train_data[['SO2', 'NO2', 'Month']]
X_test = test_data[['SO2', 'NO2', 'Month']]

y_train = train_data['RSPM/PM10']
y_test = test_data['RSPM/PM10']


# STEP 4: Create models
models = {
    "Simple Linear Regression": LinearRegression(),
    "Multiple Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    ),
    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),
    "Support Vector Regression": make_pipeline(
        StandardScaler(),
        SVR(kernel='linear')
    )
}


# STEP 5: Train models and calculate performance
results = []

for name, model in models.items():

    if name == "Simple Linear Regression":
        model.fit(X_train_simple, y_train)
        predictions = model.predict(X_test_simple)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, mse, rmse, r2])

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))


# STEP 6: Model comparison
results_df = pd.DataFrame(
    results,
    columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score']
)

print("\nFINAL MODEL COMPARISON:")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)


# GRAPH 1: RMSE Comparison
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['RMSE'])
plt.title("RMSE Comparison of Regression Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


# GRAPH 2: Actual vs Predicted for Best Model
best_model_name = results_df.loc[
    results_df['RMSE'].idxmin(), 'Model'
]

print("\nBest Model based on RMSE:", best_model_name)

best_model = models[best_model_name]

if best_model_name == "Simple Linear Regression":
    best_predictions = best_model.predict(X_test_simple)
else:
    best_predictions = best_model.predict(X_test)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_predictions)
plt.xlabel("Actual RSPM/PM10")
plt.ylabel("Predicted RSPM/PM10")
plt.title("Actual vs Predicted Air Quality Values")
plt.tight_layout()
plt.show()

print("\nProject Execution Completed!")
