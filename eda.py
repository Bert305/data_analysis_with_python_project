import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# PART 2 EDA: Exploratory Data Analysis

# Create outputs directory if it doesn't exist
# os.makedirs("outputs/plots", exist_ok=True)

df = pd.read_csv("data/kc_house_data_clean.csv") # read cleaned data

# Boxplot: waterfront vs price
sns.boxplot(x="waterfront", y="price", data=df) # create boxplot
plt.title("Price vs Waterfront") # add title
# plt.savefig("outputs/plots/waterfront_boxplot.png") # save plot
plt.show() # display plot
plt.close() # close plot
# Outputs 2 boxplots for waterfront=0 and waterfront=1
# Explanation:
# min     = smallest value (excluding outliers)
# Q1      = 25th percentile (first quartile)
# median  = 50th percentile (middle value)
# Q3      = 75th percentile (third quartile)
# max     = largest value (excluding outliers)

# YouTube explanation of boxplots:
# https://www.youtube.com/watch?v=nV8jR8M8C74

# Example (with prices)
# Imagine house prices like this:

# min     = $120,000
# Q1      = $350,000
# median  = $520,000
# Q3      = $780,000
# max     = $1,200,000


# That means:

# 25% of houses cost less than $350k
# 50% cost less than $520k
# 75% cost less than $780k
# Most houses fall between $350k–$780k


# Regression plot: sqft_above vs price
sns.regplot(x="sqft_above", y="price", data=df)
plt.title("Sqft Above vs Price") # add title
# plt.savefig("outputs/plots/sqft_above_regplot.png") # save plot
plt.show() # display plot
plt.close() # close plot
# Explanation:
# Each dot is a coordinate pair (sqft_above, price)
# for one house in your dataset. 
# The regression line then shows the overall trend through all those coordinate points.





print("EDA plots saved.")  # print confirmation
