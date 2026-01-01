import pandas as pd
from pathlib import Path

# PART 1: Load and clean the dataset


DATA_PATH = Path("data/kc_house_data.csv") # Original dataset path
OUT_PATH = Path("data/kc_house_data_clean.csv") # Cleaned dataset output path

def main():
    df = pd.read_csv(DATA_PATH)

    # Drop unnecessary columns
    for col in ["id", "date", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Handle missing values
    for col in ["bedrooms", "bathrooms"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    df.to_csv(OUT_PATH, index=False)
    print("Cleaned dataset saved.")
    print(df.dtypes) # show data types
    print(df.head()) # show first few rows

if __name__ == "__main__":
    main()
