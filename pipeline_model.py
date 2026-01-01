# Import pandas to load and work with tabular (CSV) data
import pandas as pd

# Import Pipeline to chain preprocessing and modeling steps together
from sklearn.pipeline import Pipeline

# Import tools to scale features and create polynomial (nonlinear) features
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Import Linear Regression model
from sklearn.linear_model import LinearRegression


# -----------------------------
# PART 4: Pipeline Model
# -----------------------------

# Define which columns (X-features) the model will use as inputs
FEATURES = [
    "floors",            # Number of floors in the house
    "waterfront",        # Whether the house has a waterfront view (0 or 1)
    "lat",               # Latitude (location information)
    "bedrooms",          # Number of bedrooms
    "sqft_basement",     # Square footage of the basement
    "view",              # Quality of the view
    "bathrooms",         # Number of bathrooms
    "sqft_living15",     # Living space of nearby houses
    "sqft_above",        # Square footage above ground
    "grade",             # Overall construction and design quality
    "sqft_living"        # Total living space
]

# Load the cleaned housing dataset from a CSV file into a DataFrame
df = pd.read_csv("data/kc_house_data_clean.csv")

# Select the feature columns from the DataFrame (input variables)
X = df[FEATURES]

# Select the Y-target variable (what we want to predict)
y = df["price"]

# Create a machine learning pipeline
pipe = Pipeline([ 
    # Step 1: Scale features so they are on the same numeric scale
    ("scale", StandardScaler()),
    # Step 2: Generate polynomial features to capture nonlinear relationships
    ("poly", PolynomialFeatures(include_bias=False)), 
    # Step 3: Apply linear regression to the transformed features
    ("model", LinearRegression())
])

# Train (fit) the pipeline using the feature data (X) and target prices (y)
pipe.fit(X, y)

# Print the R² score, which shows how much variance in price the model explains
print("Pipeline R²:", pipe.score(X, y))  # 0.752 means 75.2% of price variance is explained

