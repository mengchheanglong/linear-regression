import joblib
import pandas as pd


model = joblib.load("models/house_price_model.pkl")

sample = pd.DataFrame(
    [
        {
            "area": 6000,
            "bedrooms": 3,
            "bathrooms": 2,
            "stories": 2,
            "mainroad": "yes",
            "guestroom": "no",
            "basement": "yes",
            "hotwaterheating": "no",
            "airconditioning": "yes",
            "parking": 1,
            "prefarea": "yes",
            "furnishingstatus": "semi-furnished",
        }
    ]
)

prediction = model.predict(sample)

print("Predicted price:", prediction[0])
