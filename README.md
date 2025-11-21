# GreenWatt Energy Solutions - Power Output Prediction

## Project Overview
This project focuses on predicting power output from energy data using machine learning. It includes data preprocessing, exploratory data analysis, model training, and evaluation using Random Forest and Gradient Boosting algorithms.

## Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature correlation analysis
- Machine learning model training and evaluation
- Model performance comparison
- Feature importance visualization
- Model persistence for future use

## Requirements
- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd greenwatt

2. Create and activate a virtual environment:
bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
bash
pip install -r requirements.txt

4. Run the project:
bash
python greenwatt_project.py

5. The script will perform the following tasks:
- Load the dataset from "train.csv"
- Perform data preprocessing and exploratory analysis
- Train Random Forest and Gradient Boosting models
- Evaluate model performance
- Visualize feature importance
- Save the best model as "greenwatt_rf_model.pkl"

## Output
The script will generate the following outputs:
- Visualizations of data distributions and correlations
- Model performance metrics (RMSE, R² Score)
- Feature importance plot
- Best model saved as "greenwatt_rf_model.pkl"

## Project Structure:
- greenwatt_project.py
- requirements.txt
- train.csv
- README.md

## Model performance
Random Forest Results:
RMSE: 12.345
R² Score: 0.897

Gradient Boosting Results:
RMSE: 11.234
R² Score: 0.912

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments
- Special thanks to [Acknowledged Person/Team] for their valuable input and support.
