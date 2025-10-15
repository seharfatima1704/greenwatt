# ================================================
# GreenWatt Energy Solutions - Power Output Prediction
# Author: (SEHAR FATIMA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Load dataset ---
df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully ✅")
print("Shape of data:", df.shape)
print("\nSample records:")
print(df.head())

# --- Basic info ---
print("\nData Info:")
print(df.info())

# --- Check for missing values ---
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop columns with too many nulls if any
df = df.dropna()

# --- Descriptive statistics ---
print("\nStatistical summary:")
print(df.describe().T)

# --- Correlation analysis ---
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()

# --- Target variable ---
target_col = 'Target'  
if target_col not in df.columns:
    target_col = df.columns[-1]  

# --- Feature selection ---
X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --- Scale numerical features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Random Forest ---
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("RMSE:", round(rf_rmse, 3))
print("R² Score:", round(rf_r2, 3))

# --- Model 2: Gradient Boosting ---
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)

gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_r2 = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Results:")
print("RMSE:", round(gb_rmse, 3))
print("R² Score:", round(gb_r2, 3))

# --- Compare results ---
results = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'RMSE': [rf_rmse, gb_rmse],
    'R² Score': [rf_r2, gb_r2]
})
print("\nModel Performance Comparison:")
print(results)

# --- Feature importance plot ---
plt.figure(figsize=(10,5))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()

# --- Save best model ---
import joblib
joblib.dump(rf, "greenwatt_rf_model.pkl")
print("\nBest model saved as 'greenwatt_rf_model.pkl' ✅")

# --- Summary ---
print("\nProject completed successfully!")
print("Use the saved model to predict power output for new turbine data.")
