# Sales-Demand-Forecasting-for-Businesses-using-XGBoost
# Sales & Demand Forecasting using XGBoost

## Project Overview

This project predicts future business sales using historical retail sales data.
The goal is to help businesses make better decisions regarding:
- Inventory planning
- Staffing
- Budget allocation
- Demand forecasting

## Dataset

Superstore Sales Dataset
Features included:
- Order Date
- Sales
- Product Information
- Region Information

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost

## Methodology

### Data Preparation

- Converted order dates into datetime format
- Aggregated transactions into monthly sales

### Feature Engineering

Created:
- Month
- Quarter
- Year
- Lag Features (1, 3, 6 months)
- Rolling Averages
- Seasonal Features (sin/cos encoding)

### Model

XGBoost Regressor

### Evaluation Metrics

MAE: 12561.66

RMSE: 14850.87

R² Score: 0.6459

## Results

The model successfully captured major sales trends and explained approximately 64.6% of the variation in monthly sales.

The forecast can help businesses anticipate future demand and make informed operational decisions.

## Business Impact

Businesses can use this model to:

- Predict future sales
- Improve inventory management
- Reduce stock shortages
- Optimize staffing levels
- Support budgeting decisions
