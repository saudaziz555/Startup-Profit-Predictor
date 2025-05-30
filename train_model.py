import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
data = pd.read_excel('50_Startups.xlsx')

# Check data
print("Data loaded successfully with shape:", data.shape)
print("Columns:", data.columns.tolist())

# Features and target
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully!")
print(f"R^2 Score: {model.score(X, y):.4f}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
