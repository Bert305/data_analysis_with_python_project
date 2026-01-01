import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# PART 2 EDA: Exploratory Data Analysis

# Create outputs directory if it doesn't exist
os.makedirs("outputs/plots", exist_ok=True)

df = pd.read_csv("data/kc_house_data_clean.csv") # read cleaned data

# Boxplot: waterfront vs price
sns.boxplot(x="waterfront", y="price", data=df) # create boxplot
plt.title("Price vs Waterfront") # add title
plt.savefig("outputs/plots/waterfront_boxplot.png") # save plot
plt.close() # close plot

# Regression plot: sqft_above vs price
sns.regplot(x="sqft_above", y="price", data=df)
plt.title("Sqft Above vs Price") # add title
plt.savefig("outputs/plots/sqft_above_regplot.png") # save plot
plt.close() # close plot

print("EDA plots saved.")  # print confirmation
