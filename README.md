# Data Analysis with Python Project

A comprehensive data analysis project using Python for predictive modeling and exploratory data analysis on the King County house sales dataset. Inspired from the IBM Course Data Analysis with Python.

# Notes from IBM Course Data Analysis with Python
https://docs.google.com/document/d/1CF4naOy_uOlzKLxhNeE9y0XEBT6vzXqSGcK1sfN2jIM/edit?usp=sharing

## Project Overview

This project demonstrates a complete data science workflow including data cleaning, exploratory data analysis, and building multiple regression models to predict house prices.

## Project Structure

```
data_analysis_with_python_project/
├── data/
│   ├── kc_house_data.csv          # Raw dataset
│   └── kc_house_data_clean.csv    # Cleaned dataset
├── linear_regression_vs_random_forest/
│   ├── main.py                    # LR vs RF model comparison
│   ├── about_this_project.md      # Project explanation
│   ├── explain_dataset.md         # Dataset details
│   ├── LR_vs_RF_Research.md       # Research notes
│   ├── Plot_Explained.md          # Visualization explanation
│   └── step_by_step_deatils.md    # Step-by-step guide
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
- **Linear Regression vs Random Forest**: 
  - Model comparison on Delaney solubility dataset (1,144 compounds)
  - Side-by-side evaluation of Linear Regression and Random Forest Regressor
  - Performance metrics comparison (R², MSE, RMSE)
  - Scatter plot visualization showing actual vs predicted values
  - Educational documentation explaining supervised learning and regression tasks
- **Enhanced Multiple Linear Regression**: 
  - Comprehensive model evaluation with R², MSE, and RMSE metrics
  - 4-panel visualization showing actual vs predicted prices and residual plots
  - Performance comparison between training and testing sets
  - Example prediction demonstration with real house features
  - Detailed coefficient interpretation and explanation
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

4. **Compare Linear Regression vs Random Forest**:
```bash
cd linear_regression_vs_random_forest
python main.py
```
   This script will:
   - Load the Delaney solubility dataset (1,144 compounds)
   - Train both Linear Regression and Random Forest Regressor models
   - Display performance metrics (R², MSE, RMSE) for both models
   - Generate scatter plot comparing actual vs predicted values
   - Show model comparison results (LR performed slightly better with R²=0.77 vs RF R²=0.76)

5. **Train multiple linear regression with visualizations**:
```bash
cd mult_linear_regression
python multi_linear_regression.py
```
   This script will:
   - Train a multiple linear regression model using sqft_living, bedrooms, and bathrooms
   - Display R², MSE, and RMSE metrics for both training and testing sets
   - Show model coefficients with detailed interpretation
   - Demonstrate prediction with an example house (1500 sqft, 3 bedrooms, 2 bathrooms)
   - Generate a 4-panel visualization showing actual vs predicted prices and residual plots

6. **Train ridge regression models**:
```bash
python ridge_and_poly_ridge.py
```

7. **Run ML pipeline**:
```bash
python pipeline_model.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Datasets

### King County House Sales Dataset
Used in most scripts for predicting house prices. Contains house sale prices and various features such as:
- Square footage
- Number of bedrooms and bathrooms
- Waterfront location
- Year built
- And more...

### Delaney Solubility Dataset
Used in the Linear Regression vs Random Forest comparison. Contains:
- 1,144 compounds with structural information
- Water solubility data (logS target variable)
- Molecular descriptors (MolLogP, MolWt, NumRotatableBonds, AromaticProportion)
- Benchmark dataset for supervised learning regression tasks

## Output

The project generates:
- Cleaned dataset in `data/kc_house_data_clean.csv`
- Visualization plots in `outputs/plots/`
  - `multi_linear_regression_results.png`: 4-panel visualization with actual vs predicted and residual plots
  - Actual vs Predicted scatter plot from LR vs RF comparison
- Model performance metrics printed to console:
  - **Linear Regression vs Random Forest (Solubility Dataset)**:
    - LR Test R²: 0.7706 | MSE: 0.9991 | RMSE: 1.0070
    - RF Test R²: 0.7584 | MSE: 1.0521 | RMSE: 1.0282
    - Linear Regression performed slightly better on this dataset
  - **Multi-Linear Regression (House Price Dataset)**:
    - R² Score: 0.51 (proportion of variance explained)
    - MSE: Mean Squared Error in squared dollar units
    - RMSE: ~$254k training, ~$272k testing
    - Model coefficients with interpretation (e.g., each sqft adds ~$306 to price)
    - Example prediction for a sample house (~$382k for 1500 sqft, 3 bed, 2 bath)

## License

This project is open source and available for educational purposes.