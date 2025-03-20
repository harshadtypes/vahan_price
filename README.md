# Car Price Prediction

## Overview

This project uses machine learning to predict car selling prices based on various features such as fuel type, seller type, transmission type, and present price. The dataset is processed, analyzed, and visualized before training a Linear Regression model.

## Dataset

The dataset used in this project contains various attributes of used cars, including:

- **Car_Name**: The name of the car model

- **Year**: Manufacturing year of the car

- **Selling_Price**: The price at which the car is sold (target variable)

- **Present_Price**: The current ex-showroom price of the car

- **Kms_Driven**: The distance the car has been driven in kilometers

- **Fuel_Type**: Type of fuel used (Petrol, Diesel, CNG)

- **Seller_Type**: Indicates if the seller is an individual or a dealer

- **Transmission**: Type of transmission (Manual or Automatic)

- **Owner**: Number of previous owners

## Features & Functionality

### Data Preprocessing

- Loads the dataset using Pandas.

- Checks for missing values and data types.

- Encodes categorical variables:

  - **Manual encoding** for `Fuel_Type`.

  - **One-hot encoding** for `Seller_Type` and `Transmission`.

- Splits the data into features (X) and target variable (y).

- Standardizes numerical features using `StandardScaler`.

### Data Visualization

- **Categorical Data Analysis**

Visualizes the distribution of `Fuel_Type`, `Seller_Type`, and `Transmission` with respect to `Selling_Price` using bar charts and seaborn bar plots.

- **Correlation Heatmap**

  - Displays the relationship between numerical features.

- **Regression Plot**

  - Shows the correlation between Present_Price and Selling_Price.

### Model Training & Evaluation

- Splits the dataset into **training (70%)** and **test (30%)** sets.

- Trains a Linear Regression model using sklearn.linear_model.LinearRegression.

- Evaluates the model using:

  - **Mean Absolute Error (MAE)**

  - **Mean Squared Error (MSE)**

  - **R² Score**

- Compares predicted and actual prices using a seaborn regression plot.

## How to Use

1. Clone the repository:

2. Navigate to the project directory:

3. Install dependencies:

4. Run the script:

## Dependencies

Ensure you have the following Python libraries installed:

- pandas

- numpy

- seaborn

- matplotlib

- scikit-learn

You can install them using:

## Results

- The model provides insights into which features influence car prices the most.

- The trained Linear Regression model predicts car prices with reasonable accuracy.

- The visualizations help in understanding key trends in the dataset.
