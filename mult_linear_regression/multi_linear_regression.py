import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress the feature names warning
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# Seperate features and target variable


# Load data
df = pd.read_csv("../data/kc_house_data.csv")

# Select features and target
X = df[["sqft_living", "bedrooms", "bathrooms"]]
y = df["price"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

# Make predictions for metrics
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate MSE and RMSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
rmse_test = root_mean_squared_error(y_test, y_pred_test)

print(f"Training R²: {r2_train:.4f}")
print(f"Testing R²: {r2_test:.4f}")
print(f"\nTraining MSE: ${mse_train:,.2f}")
print(f"Testing MSE: ${mse_test:,.2f}")
print(f"\nTraining RMSE: ${rmse_train:,.2f}")
print(f"Testing RMSE: ${rmse_test:,.2f}")

# Why these metrics are valuable:

# RMSE of ~$254,000 (training) and ~$272,000 (testing) tells you that on average, your model's predictions are off by about $254k-272k. 

# This is more intuitive than R² since it's in dollar terms.

# MSE is useful for comparing models mathematically (larger errors are penalized more heavily due to squaring).

# R² = 0.51 tells you that your model explains 51% of the variance in house prices.

# The small difference between training and testing metrics (RMSE: $254k vs $272k) suggests your model generalizes well with minimal overfitting.





# Model coefficients
coefficients = pd.Series(
    model.coef_,
    index=X.columns,
    name="Coefficient"
)
# What are coefficients? (the rules your machine learned)

# Each number tells the machine:

# > “If this thing changes, how does the price change?”
print("\nModel Coefficients:")
print(coefficients)

print(f"\nIntercept: {model.intercept_:.2f}")

# If square footage increases by 1 sqft, price increases by about $287.63.
# If bedrooms increase by 1, price decreases by about $31,631.57, holding other factors constant.
# If bathrooms increase by 1, price increases by about $50,000, holding other factors constant.

# So if the sqft_living is 1500, bedrooms is 3, and bathrooms is 2, the predicted price would be:
example_sqft = 1500
example_bedrooms = 3
example_bathrooms = 2
example_features = np.array([[example_sqft, example_bedrooms, example_bathrooms]])
predicted_price = model.predict(example_features)
print(f"\nPredicted Price for example house: ${predicted_price[0]:,.2f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted (Training)
axes[0, 0].scatter(y_train, y_pred_train, alpha=0.5, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'Training Set: Actual vs Predicted (R² = {r2_train:.2f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Testing)
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price')
axes[0, 1].set_ylabel('Predicted Price')
axes[0, 1].set_title(f'Testing Set: Actual vs Predicted (R² = {r2_test:.2f})')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals (Training)
residuals_train = y_train - y_pred_train
axes[1, 0].scatter(y_pred_train, residuals_train, alpha=0.5, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Price')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Training Set: Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals (Testing)
residuals_test = y_test - y_pred_test
axes[1, 1].scatter(y_pred_test, residuals_test, alpha=0.5, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Price')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Testing Set: Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('../outputs/plots/multi_linear_regression_results.png', dpi=300, bbox_inches='tight')
# print("\nPlot saved to: ../outputs/plots/multi_linear_regression_results.png")
plt.show()

