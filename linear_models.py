# Import pandas to load and manipulate the dataset
import pandas as pd

# Import LinearRegression to build a basic regression model
from sklearn.linear_model import LinearRegression

# Load the cleaned housing dataset from a CSV file
df = pd.read_csv("data/kc_house_data_clean.csv")


# -----------------------------
# PART 3: Simple Linear Model
# -----------------------------

# Select a single x-feature (sqft_living) as the input variable
# This represents the size of the house
X = df[["sqft_living"]]

# Select the y-target variable we want to predict
# This represents the house price
y = df["price"]

# Create a Linear Regression model instance
model = LinearRegression()

# Train (fit) the model using the feature and target data
model.fit(X, y)

# Print the R² score of the model
# R² = 0.49 means sqft_living alone explains about 49% of the variation in price
print("R² using sqft_living:", model.score(X, y))
# Explanation of R² score:
# Input (X): sqft_living
# Output (y): price
# R² tells you how informative sqft_living is by itself.
# An R² of 0.49 means:
# Almost half of price variability is explained by living area alone
# This confirms sqft_living is a strong predictor
# But also shows other features matter (location, grade, bathrooms, etc.)