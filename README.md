# Data Analysis with Python Project

A comprehensive data analysis project using Python for predictive modeling and exploratory data analysis on the King County house sales dataset.

## Project Overview

This project demonstrates a complete data science workflow including data cleaning, exploratory data analysis, and building multiple regression models to predict house prices.

## Project Structure

```
data_analysis_with_python_project/
├── data/
│   ├── kc_house_data.csv          # Raw dataset
│   └── kc_house_data_clean.csv    # Cleaned dataset
├── mult_linear_regression/
│   └── multi_linear_regression.py # Enhanced multiple linear regression with visualizations
├── outputs/
│   └── plots/                     # Generated visualizations
├── load_and_clean.py              # Data loading and cleaning script
├── eda.py                         # Exploratory data analysis
├── linear_models.py               # Linear regression models
├── ridge_and_poly_ridge.py        # Ridge and polynomial regression
├── pipeline_model.py              # Machine learning pipeline
└── requirements.txt               # Project dependencies
```

## Features

- **Data Cleaning**: Handles missing values, outliers, and data preprocessing
- **Exploratory Data Analysis**: Generates insightful visualizations including boxplots and regression plots
- **Linear Models**: Implements simple and multiple linear regression
- **Enhanced Multiple Linear Regression**: 
  - Comprehensive model evaluation with R², MSE, and RMSE metrics
  - 4-panel visualization showing actual vs predicted prices and residual plots
  - Performance comparison between training and testing sets
- **Advanced Models**: Ridge regression and polynomial feature engineering
- **ML Pipeline**: Complete scikit-learn pipeline for model training and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bert305/data_analysis_with_python_project.git
cd data_analysis_with_python_project
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in the following order:

1. **Load and clean data**:
```bash
python load_and_clean.py
```

2. **Exploratory data analysis**:
```bash
python eda.py
```

3. **Train linear models**:
```bash
python linear_models.py
```

4. **Train multiple linear regression with visualizations**:
```bash
cd mult_linear_regression
python multi_linear_regression.py
```
   This script will:
   - Train a multiple linear regression model using sqft_living, bedrooms, and bathrooms
   - Display R², MSE, and RMSE metrics for both training and testing sets
   - Generate a 4-panel visualization showing actual vs predicted prices and residual plots
   - Save the plot to `outputs/plots/multi_linear_regression_results.png`

5. **Train ridge regression models**:
```bash
python ridge_and_poly_ridge.py
```

6. **Run ML pipeline**:
```bash
python pipeline_model.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Dataset

The project uses the King County House Sales dataset, which contains house sale prices and various features such as:
- Square footage
- Number of bedrooms and bathrooms
- Waterfront location
- Year built
- And more...

## Output

The project generates:
- Cleaned dataset in `data/kc_house_data_clean.csv`
- Visualization plots in `outputs/plots/`
  - `multi_linear_regression_results.png`: 4-panel visualization with actual vs predicted and residual plots
- Model performance metrics printed to console:
  - **R² Score**: Measures proportion of variance explained (0.51 for multi-linear regression)
  - **MSE**: Mean Squared Error in squared dollar units
  - **RMSE**: Root Mean Squared Error (~$254k training, ~$272k testing for multi-linear regression)
  - Model coefficients and intercept values

## License

This project is open source and available for educational purposes.