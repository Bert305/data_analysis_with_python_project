import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/kc_house_data_clean.csv")


# PART 3: Linear Models


# Question 6 equivalent
X = df[["sqft_living"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

print("R² using sqft_living:", model.score(X, y)) # score is 0.49 meaning 49% variance explained

# Input (X): sqft_living

# Output (y): price

# R² tells you how informative sqft_living is by itself.

# An R² of 0.49 means:

# Almost half of price variability is explained by living area alone

# This confirms sqft_living is a strong predictor

# But also shows other features matter (location, grade, bathrooms, etc.)