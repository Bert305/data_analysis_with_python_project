import pandas as pd
from pathlib import Path

# PART 1: Load and clean the dataset


DATA_PATH = Path("data/kc_house_data.csv") # Original dataset path
OUT_PATH = Path("data/kc_house_data_clean.csv") # Cleaned dataset output path

def main():
    df = pd.read_csv(DATA_PATH) # Load dataset

    # Drop unnecessary columns
    for col in ["id", "date", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Handle missing values of bedrooms and bathrooms by filling with mean replacing NaNs/nulls
    for col in ["bedrooms", "bathrooms"]:
        # counts how many null/NaN values exist in that column
        if df[col].isnull().sum() > 0:
            # if any missing values,fill NaNs with mean value of that column
            df[col] = df[col].fillna(df[col].mean())
            

    # df.to_csv(OUT_PATH, index=False) # Save cleaned dataset
    # print("Cleaned dataset saved.")
    print(df.dtypes) # show data types
    print(df.head()) # show first few rows
    print(df.describe()) # show summary of cleaned data

if __name__ == "__main__":
    main()
