import requests
import streamlit as st

API_URL = "http://localhost:5000/predict"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("Enter the house details below and click **Predict Price** to get an estimate.")

st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        options=["furnished", "semi-furnished", "unfurnished"],
    )

with col2:
    mainroad = st.selectbox("Main Road Access", options=["yes", "no"])
    guestroom = st.selectbox("Guest Room", options=["yes", "no"])
    basement = st.selectbox("Basement", options=["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", options=["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", options=["yes", "no"])
    prefarea = st.selectbox("Preferred Area", options=["yes", "no"])

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            predicted = result.get("predicted_price", "N/A")
            st.success("Prediction complete!")
            st.metric(
                label="Estimated House Price",
                value=f"${predicted:,.2f}" if isinstance(predicted, (int, float)) else predicted,
            )
        else:
            error_detail = response.json().get("error", response.text)
            st.error(f"API Error ({response.status_code}): {error_detail}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the Flask API. "
            "Make sure the backend is running with: `python src/app.py`"
        )
    except requests.exceptions.Timeout:
        st.error("The request timed out. The API may be overloaded. Please try again.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
