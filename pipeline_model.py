import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


# PART 4: Pipeline Model


FEATURES = [
    "floors", "waterfront", "lat", "bedrooms",
    "sqft_basement", "view", "bathrooms",
    "sqft_living15", "sqft_above", "grade", "sqft_living"
]

df = pd.read_csv("data/kc_house_data_clean.csv")

X = df[FEATURES]
y = df["price"]

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression())
])

pipe.fit(X, y)
print("Pipeline R²:", pipe.score(X, y)) # got 0.752 meaning 75.2% variance explained
