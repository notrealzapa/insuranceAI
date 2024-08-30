import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
import joblib
import shap

# Load and prepare the dataset for model training
file_path = '/project/workspace/Adjusted_Risk_Factor_Cybersecurity_Dataset.csv'
dataset = pd.read_csv(file_path)

X = dataset.drop('Risk Factor', axis=1)
y = dataset['Risk Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numeric and categorical features
numeric_features = make_column_selector(dtype_include=['int64', 'float64'])
categorical_features = make_column_selector(dtype_include=['object'])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Pipeline for training
risk_factor_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model)])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(risk_factor_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the best model to a file for future use
joblib.dump(best_model, 'best_model.pkl')

# SHAP values for model interpretability
explainer = shap.Explainer(best_model.named_steps['regressor'], best_model.named_steps['preprocessor'].transform(X_train))
shap_values = explainer(best_model.named_steps['preprocessor'].transform(X_test))

# Plot SHAP values for the first prediction
shap.initjs()
shap.force_plot(shap_values[0], X_test.iloc[0, :])

# Function to calculate the premium based on the risk factor
def calculate_premium(risk_factor):
    base_premium = 500  # Base premium for minimal risk
    risk_multiplier = 10  # Multiplier for the risk factor
    premium = base_premium + (risk_multiplier * risk_factor)
    return premium

# Function to predict the risk factor for a given user input
def predict_risk_factor(user_input):
    user_df = pd.DataFrame([user_input])
    predicted_risk_factor = best_model.predict(user_df)[0]
    return predicted_risk_factor

# Function to predict risk factor and calculate insurance premium
def predict_and_calculate(user_input):
    risk_factor = predict_risk_factor(user_input)
    premium = calculate_premium(risk_factor)
    return risk_factor, premium

if __name__ == "__main__":
    # Example usage for local testing
    user_input_example = {
        'Age': 24,
        'Gender': 'Male',
        'Location': 'Urban',
        'Occupation': 'Finance',
        'Education': "Bachelor's",
        'Income': 'High',
        'Device Ownership': 5,
        'Operating System': 'iOS',
        'Internet Usage': 'High',
        'Use of Security Tools': 'Yes',
        'Social Media Usage': 'Low',
        'Privacy Settings': 'Strong',
        'Info Sharing Frequency': 'Low',
        'Email Accounts': 4,
        'Email Filtering': 'Yes',
        'Browsing Behavior': 'Cautious',
        'Security Awareness': 'High',
        'Password Hygiene': 'Strong',
        'Incident History': 'No',
        'Sensitive Info Access': 'No',
        'Workplace Policies': 'Strict',
        'Remote Work Status': 'Yes'
    }

    risk_factor, premium = predict_and_calculate(user_input_example)
    print(f"Predicted Risk Factor: {risk_factor:.2f}")
    print(f"Predicted Insurance Premium: ${premium:.2f}")
