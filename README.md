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

4. **Train ridge regression models**:
```bash
python ridge_and_poly_ridge.py
```

5. **Run ML pipeline**:
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
- Model performance metrics printed to console

## License

This project is open source and available for educational purposes.