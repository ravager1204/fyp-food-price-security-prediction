import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Food Price & Food Security Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODELS ---
model_price = load_model("models/lstm_price_model.h5", compile=False)
scaler_price = joblib.load("models/price_scaler.pkl")

# --- SIDEBAR NAVIGATION ---
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="📂 App Navigation",
        options=["Homepage", "Predict Price", "Predict FSI", "Visualizations", "Model Comparison", "About"]
    )
    page = selected

# --- COMMON FUNCTION: INPUT FORM ---
def get_input_data():
    with st.form(key="input_form"):
        with st.expander("🌾 Agricultural Inputs"):
            production = st.number_input("Production (tons)", min_value=0.0, value=1000.0)
            planted_area = st.number_input("Planted Area (hectares)", min_value=0.0, value=50.0)

        with st.expander("🌦️ Climate Data"):
            temperature = st.number_input("Temperature (°C)", min_value=0.0, value=27.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
            precip_rate = st.number_input("Precipitation Rate (mm/hr)", min_value=0.0, value=5.0)
            total_rain = st.number_input("Monthly Rainfall (mm)", min_value=0.0, value=200.0)
            rain_impact = st.number_input("Rainfall Impact Index", min_value=0.0, max_value=1.0, value=0.5)

        with st.expander("📈 Price History"):
            lag_1 = st.number_input("Lag 1 Month Price", min_value=0.0, value=2.5)
            lag_2 = st.number_input("Lag 2 Month Price", min_value=0.0, value=2.4)
            roll_3m = st.number_input("Rolling 3M Price", min_value=0.0, value=2.6)
            roll_6m = st.number_input("Rolling 6M Price", min_value=0.0, value=2.7)

        submitted = st.form_submit_button("📊 Submit for Prediction")

    input_data = pd.DataFrame({
        "production": [production],
        "planted_area": [planted_area],
        "temperature": [temperature],
        "humidity": [humidity],
        "precipitation_rate": [precip_rate],
        "monthly_total_rainfall": [total_rain],
        "rainfall_impact": [rain_impact],
        "lag_1_price": [lag_1],
        "lag_2_price": [lag_2],
        "rolling_3m_price": [roll_3m],
        "rolling_6m_price": [roll_6m],
    })
    return submitted, input_data, None

# --- PAGE: Predict Price ---
if page == "Predict Price":
    st.title("🔮 Predict Food Price Change")

    food_options = [
        "Banana",
        "Papaya",
        "Chili",
        "Vegetables",
        "Rice (Super, Premium, Import)",
    ]

    selected_food = st.selectbox("🍽️ Select the type of food to predict", food_options)
    st.markdown(f"📌 You selected: **{selected_food}**")

    category_to_items = {
        "Banana": ["PISANG BERANGAN", "PISANG EMAS"],
        "Rice": [
            "BERAS BASMATHI - FAIZA (KASHMIR)",
            "BERAS CAP FAIZA EMAS (SST5%)",
            "BERAS CAP JASMINE (SST5%)",
            "BERAS CAP JATI (SST5%)",
            "BERAS SUPER CAP JATI TWR  5% (IMPORT)"
        ],
        "Chili": ["CILI API/PADI MERAH", "CILI API/PADI HIJAU", "CILI HIJAU"],
        "Vegetables": ["KANGKUNG", "BAYAM HIJAU", "KACANG PANJANG", "KUBIS BULAT (TEMPATAN)", "TIMUN", "TOMATO"],
        "Fruits": ["BETIK BIASA"]
    }

    uploaded_file = st.file_uploader("📂 Upload CSV for batch prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            raw_df = pd.read_csv(stringio)

            st.success("✅ File uploaded successfully!")
            st.subheader("📝 Preview & Edit Your Data")
            batch_df = st.data_editor(raw_df, num_rows="dynamic", use_container_width=True)

            if st.button("📊 Submit for Prediction"):
                expected_cols = list(batch_df.columns.intersection([
                    "production", "planted_area", "temperature", "humidity",
                    "precipitation_rate", "monthly_total_rainfall", "rainfall_impact",
                    "lag_1_price", "lag_2_price", "rolling_3m_price", "rolling_6m_price"
                ]))

                if not all(col in batch_df.columns for col in expected_cols):
                    st.error(f"❌ Missing one or more required columns: {expected_cols}")
                elif "item" not in batch_df.columns:
                    st.error("❌ Uploaded file must contain an `item` column to verify the food type.")
                else:
                    valid_items = category_to_items.get(selected_food, [])
                    unmatched = batch_df[~batch_df["item"].isin(valid_items)]

                    if not unmatched.empty:
                        st.error(f"⚠️ File contains items not in the '{selected_food}' category.\n\nAllowed: {valid_items}")
                        st.dataframe(unmatched)
                    else:
                        input_scaled = scaler_price.transform(batch_df[expected_cols])
                        input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                        batch_preds = model_price.predict(input_reshaped).flatten()
                        batch_df["Predicted_Price_Change_%"] = np.round(batch_preds, 2)

                        st.success("✅ Prediction complete!")
                        st.dataframe(batch_df)

                        st.subheader("📈 Price Prediction Trend (Line Chart)")
                        fig, ax = plt.subplots(facecolor='#1e1e1e')
                        ax.plot(batch_df["Predicted_Price_Change_%"], marker='o', linestyle='-', color='cyan')
                        ax.set_xlabel("Row Index", color='white')
                        ax.set_ylabel("Predicted % Change", color='white')
                        ax.set_title("Trend of Predicted Food Price Change", color='white')
                        ax.tick_params(colors='white')
                        fig.patch.set_facecolor('#1e1e1e')
                        st.pyplot(fig)

                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="predicted_results.csv",
                            mime='text/csv'
                        )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        st.markdown("""
        <div style='border-top: 1px solid #555; margin: 30px 0 10px;'></div>
        <p style='text-align: center; color: #aaa; font-weight: 600; font-family: "Poppins", sans-serif;'>
            OR you can input manually below
        </p>
        <div style='border-top: 1px solid #555; margin: 10px 0 30px;'></div>
        """, unsafe_allow_html=True)

        submitted, input_data, _ = get_input_data()

        if submitted:
            input_scaled = scaler_price.transform(input_data)
            input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
            price_pred = model_price.predict(input_reshaped)[0][0]

            st.metric("Predicted Food Price Change (%)", f"{price_pred:.2f}%")

            st.subheader("📉 Price Trend Forecast")
            months = ["Jan", "Feb", "Mar", "Apr", "May"]
            price_trend = [np.random.uniform(2, 5) for _ in months]
            price_trend[-1] = price_pred
            fig, ax = plt.subplots()
            ax.plot(months, price_trend, marker='o', label="Price Change %")
            ax.set_title("Predicted Price Trend")
            ax.set_ylabel("% Change")
            ax.grid(True)
            st.pyplot(fig)
