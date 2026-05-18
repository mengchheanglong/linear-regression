# fix_model.py
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

print("--- Generating a locally compatible dummy model file ---")

# Create dummy matrix data representing structural training inputs
# (Modify these features if your dataset uses different columns!)
X = np.array([
    [1500, 3, 2, 2005],
    [2000, 4, 2.5, 2015],
    [2500, 4, 3, 2020],
    [1100, 2, 1, 1998]
])
y = np.array([210000, 315000, 420000, 150000])

# Fit a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save it explicitly over the existing models/ path
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "house_price_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Success! New model binary written to {model_path}")
print("Your local environment can now read this model without STACK_GLOBAL errors.")