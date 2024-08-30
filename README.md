# Insurance AI Algorithm

This project is a web-based dashboard for insurance company employees to manage customer data, calculate risk factors, and estimate insurance premiums based on uploaded datasets. The dashboard is designed to be visually appealing, professional, and user-friendly.

## Features

- **Employee Login Display**: The dashboard shows the logged-in employee's ID.
- **CSV Upload**: Employees can upload CSV files containing customer data and give each dataset a unique name.
- **Data Processing**: The backend processes the data to calculate risk factors and insurance premiums.
- **Results Display**: The processed results are displayed in a table, showing the dataset name, risk factor, and calculated insurance price.
- **Responsive Design**: The dashboard uses Bootstrap for a clean, responsive layout.
- **AI-Powered Predictions**: The backend leverages an advanced AI model trained to predict customer risk factors and calculate corresponding insurance premiums based on user-provided data.

## AI Algorithm Details

### Model Overview

The AI algorithm used in this project is a machine learning model based on the Gradient Boosting Regressor. The model is trained to predict the risk factor for an insurance customer, which is then used to calculate the insurance premium. 

### Key Components:

- **Preprocessing**: 
  - Numeric features are scaled using `StandardScaler` and transformed using polynomial features to capture non-linear relationships.
  - Categorical features are one-hot encoded to allow the model to interpret them effectively.
  
- **Model**:
  - **Gradient Boosting Regressor**: A powerful ensemble learning method that builds trees sequentially, with each new tree aiming to reduce the errors of the previous ones. This model is well-suited for predicting risk factors in a complex, non-linear dataset.
  - Hyperparameters are tuned using `GridSearchCV` to optimize model performance.

### Training Process

1. **Data Splitting**: The dataset is split into training and testing sets to ensure the model generalizes well to unseen data.
2. **Feature Engineering**: Numeric and categorical features are preprocessed as described above.
3. **Model Training**: The Gradient Boosting Regressor is trained on the preprocessed data.
4. **Model Evaluation**: The model's performance is evaluated using R² score, ensuring it accurately predicts the risk factor.

### Prediction and Premium Calculation

- The model predicts the risk factor for each customer based on their data.
- The insurance premium is calculated using a simple linear formula:
  ```python
  premium = base_premium + (risk_multiplier * risk_factor)

