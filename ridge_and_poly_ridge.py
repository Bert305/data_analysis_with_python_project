import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# PART 5: Ridge and Polynomial Ridge Models


FEATURES = [
    "floors", "waterfront", "lat", "bedrooms",
    "sqft_basement", "view", "bathrooms",
    "sqft_living15", "sqft_above", "grade", "sqft_living"
]

df = pd.read_csv("data/kc_house_data_clean.csv")

X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=1
)

# Question 9: Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
print("Ridge R² (test):", ridge.score(X_test, y_test))

# Question 10: Polynomial + Ridge
poly_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", Ridge(alpha=0.1))
])

poly_ridge.fit(X_train, y_train)
print("Poly(2)+Ridge R² (test):", poly_ridge.score(X_test, y_test))
