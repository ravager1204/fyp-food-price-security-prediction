import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Food Price & Food Security Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS STYLING ---
st.markdown(
    """
    <style>
        /* Load Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Apply Google Font */
        html, body, .stApp {
            background-color: #1e1e1e;
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
        }

        /* Number Input */
        .stNumberInput > div > input {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 8px;
        }

        /* Text Input and File Uploader */
        .stTextInput > div > input,
        .stFileUploader > div > div {
            background-color: #333333 !important;
            color: #ffffff !important;
            border-radius: 8px;
        }

        /* DataFrame display */
        .stDataFrame {
            background-color: #2b2b2b !important;
            border-radius: 8px;
        }

        /* Buttons or other common components */
        .css-1cpxqw2, .css-qrbaxs, .css-1v0mbdj {
            background-color: #444444 !important;
            color: #ffffff !important;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)




# Load models once
model_price = load_model("models/lstm_price_model.h5", compile=False)
model_fsi = joblib.load("models/rf_fsi_model.pkl")


# --- SIDEBAR NAVIGATION ---
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="📂 App Navigation",
        options=["Dashboard", "Predict Price", "Predict FSI", "Visualizations", "Model Comparison", "About"]
    )
    page = selected

# --- COMMON FUNCTION: INPUT FORM ---
def get_input_data():
    """
    Returns:
        submitted (bool): Form submission status.
        input_data (DataFrame): Inputs compiled into a pandas DataFrame.
        uploaded_file: Optional CSV file for batch predictions.
    """
    with st.form(key="input_form"):
        # --- Agricultural Inputs ---
        with st.expander("🌾 Agricultural Inputs"):
            production = st.number_input("Production (tons)", min_value=0.0, value=1000.0)
            planted_area = st.number_input("Planted Area (hectares)", min_value=0.0, value=50.0)

        # --- Climate Data ---
        with st.expander("🌦️ Climate Data"):
            temperature = st.number_input("Temperature (°C)", min_value=0.0, value=27.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
            precip_rate = st.number_input("Precipitation Rate (mm/hr)", min_value=0.0, value=5.0)
            total_rain = st.number_input("Monthly Rainfall (mm)", min_value=0.0, value=200.0)
            rain_impact = st.number_input("Rainfall Impact Index", min_value=0.0, max_value=1.0, value=0.5)

        # --- Price History ---
        with st.expander("📈 Price History"):
            lag_1 = st.number_input("Lag 1 Month Price", min_value=0.0, value=2.5)
            lag_2 = st.number_input("Lag 2 Month Price", min_value=0.0, value=2.4)
            roll_3m = st.number_input("Rolling 3M Price", min_value=0.0, value=2.6)
            roll_6m = st.number_input("Rolling 6M Price", min_value=0.0, value=2.7)

        uploaded_file = st.file_uploader("📂 Or upload a CSV file for batch prediction", type=["csv"])
        submitted = st.form_submit_button("Submit")
    
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
    return submitted, input_data, uploaded_file

# --- PAGE: Dashboard ---
if page == "Dashboard":
    st.title("🥦 Dashboard")
    st.write(
        """
        Welcome to the Food Price & Food Security Predictor Dashboard!
        
        **Project Overview:**
        - **Objective:** Predict percentage change in food prices and the food security index (FSI) using climate and food production data in Malaysia.
        - **Inputs:** Climate data (rainfall, temperature, humidity), food production metrics, and food price history.
        - **Models:** Uses multiple regression models including Random Forest, XGBoost, LSTM, and SVR.
        
        ## 🔍 Why Do We Use the Same Inputs for Price & FSI?
        Both food prices and the Food Security Index (FSI) are influenced by similar drivers:

        - 🌾 **Agricultural Inputs:** Production and planted area directly affect supply levels, influencing both market prices and food availability.
        - 🌦️ **Climate Data:** Variables like rainfall, temperature, and humidity impact crop health, affecting harvests and prices, and potentially threatening food security.
        - 📈 **Price History:** Lagged and rolling prices indicate affordability trends which reflect economic access to food — a core element of food security.

        Using the same inputs simplifies user interaction while ensuring consistent, holistic forecasting.

        Explore the sidebar to navigate to different functionalities.
        """
    )

# --- PAGE: PREDICT PRICE ---
elif page == "Predict Price":
    st.title("🔮 Predict Food Price Change")
    st.info("""
    👉 To predict food prices, please enter:

    - 🌾 **Production**: The amount of food (e.g., rice, vegetables, poultry) harvested or produced. This reflects the supply level available in the market.
    - 🌾 **Planted Area**: The total land area used to cultivate food crops, measured in hectares. More area generally means higher production potential.
    - 🌦️ **Temperature & Humidity**: Monthly average temperature (°C) and humidity (%) that affect plant growth conditions.
    - 🌧️ **Precipitation Rate & Monthly Rainfall**: How much rain is received, and its cumulative impact on soil moisture and crop health.
    - 💧 **Rainfall Impact Score**: A preprocessed value showing how favorable or damaging the rainfall is for the crops.
    - 📉 **Lag Price (1–2 months ago)**: The actual market price of the food item 1 and 2 months prior. This helps the model detect momentum or seasonal trends.
    - 📊 **Rolling Prices**: Average prices over the last 3 and 6 months. This smooths out short-term fluctuations to reveal longer-term patterns.
    """)

    food_options = [
        "Banana",
        "Papaya",
        "Chili",
        "Vegetables",
        "Rice (Super, Premium, Import)",
    ]
    selected_food = st.selectbox("🍽️ Select the type of food to predict", food_options)
    st.markdown(f"📌 You selected: **{selected_food}**")

    st.markdown("""
    <p style='font-size:18px; font-weight:600; color:#f5f5f5;
              font-family: "Poppins", sans-serif; margin-bottom: 5px;'>
        📂 Upload CSV for batch prediction
    </p>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="", type=["csv"])

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

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)

        if "item" not in raw_df.columns:
            st.error("❌ Uploaded file must contain an `item` column to verify the food type.")
        else:
            valid_items = category_to_items.get(selected_food, [])
            unmatched = raw_df[~raw_df["item"].isin(valid_items)]

            if not unmatched.empty:
                st.error(f"⚠️ File contains items not in the '{selected_food}' category.\n\nAllowed: {valid_items}")
                st.dataframe(unmatched)
            else:
                st.success("✅ All items match the selected food category.")

    st.markdown("""
    <div style='border-top: 1px solid #555; margin: 30px 0 10px;'></div>
    <p style='text-align: center; color: #aaa; font-weight: 600; font-family: "Poppins", sans-serif;'>
        OR you can also input manually below
    </p>
    <div style='border-top: 1px solid #555; margin: 10px 0 30px;'></div>
    """, unsafe_allow_html=True)

    submitted, input_data, _ = get_input_data()

    if submitted:
        st.subheader("📊 Prediction Result")

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

    if uploaded_file:
        st.subheader("📂 Batch Prediction Result")
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.subheader("📝 Preview & Edit Your Data")
            batch_df = st.data_editor(raw_df, num_rows="dynamic", use_container_width=True)

            expected_cols = list(input_data.columns)

            if not all(col in batch_df.columns for col in expected_cols):
                st.error(f"⚠️ Uploaded CSV must contain these columns: {expected_cols}")
            elif "item" not in batch_df.columns:
                st.error("❌ Uploaded file must include an `item` column to identify food types.")
            else:
                valid_items = category_to_items.get(selected_food, [])
                unmatched = batch_df[~batch_df["item"].isin(valid_items)]

                if not unmatched.empty:
                    st.error(f"❌ File contains items not in the '{selected_food}' category.")
                    st.dataframe(unmatched)
                else:
                    batch_scaled = scaler_price.transform(batch_df[expected_cols])
                    batch_reshaped = batch_scaled.reshape((batch_scaled.shape[0], 1, batch_scaled.shape[1]))

                    batch_preds = model_price.predict(batch_reshaped).flatten()
                    batch_df["Predicted_Price_Change_%"] = np.round(batch_preds, 2)

                    st.success("✅ Predictions complete!")
                    st.dataframe(batch_df)

                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name="batch_price_predictions.csv",
                        mime='text/csv'
                    )

                    st.subheader("📈 Price Prediction Trend (Line Chart)")
                    fig, ax = plt.subplots(facecolor='#1e1e1e')
                    ax.plot(batch_df["Predicted_Price_Change_%"], marker='o', linestyle='-', color='cyan')
                    ax.set_xlabel("Row Index", color='white')
                    ax.set_ylabel("Predicted % Change", color='white')
                    ax.set_title("Trend of Predicted Food Price Change", color='white')
                    ax.tick_params(colors='white')
                    fig.patch.set_facecolor('#1e1e1e')

                    if "item" in batch_df.columns:
                        ax.set_xticks(range(len(batch_df)))
                        ax.set_xticklabels(batch_df["item"], rotation=45, ha='right')

                    st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")



# --- PAGE: Predict FSI ---
elif page == "Predict FSI":
    st.title("🔮 Predict Food Security Index (FSI)")
    #st.warning("⚠️ Models are not loaded. Dummy prediction values will be used for demonstration.")
    st.info("""
    👉 To predict the Food Security Index (FSI), please input:

    - 🌾 **Production**: The quantity of food available. Higher production improves supply and lowers scarcity risk.
    - 🌾 **Planted Area**: Indicates potential sustainability and capacity to produce food in the region.
    - 🌦️ **Temperature, Humidity, Rainfall**: Climate-related conditions that affect how easily food can be grown and distributed.
    - 💧 **Rainfall Impact Score**: Indicates whether rainfall was supportive or damaging to crops.
    - 💸 **Lagged Prices**: Market prices 1–2 months ago. Higher prices reduce food affordability.
    - 📊 **Rolling Prices (3M & 6M)**: Average food prices in the past 3 and 6 months — a key indicator of long-term affordability and access to food.
    """)
    
    # Display the input form.
    submitted, input_data, uploaded_file = get_input_data()
    st.subheader("📋 Input Summary")
    st.dataframe(input_data)
    
    if submitted:
        st.subheader("📊 Prediction Result")
        fsi_pred = model_fsi.predict(input_data)[0]
        st.metric("Predicted Food Security Index", f"{fsi_pred:.3f}")
        
        # --- Visualization: FSI Trend Forecast ---
        st.subheader("📉 FSI Trend Forecast")
        months = ["Jan", "Feb", "Mar", "Apr", "May"]
        fsi_trend = [np.random.uniform(0.5, 1.0) for _ in months]
        fsi_trend[-1] = fsi_pred  # Ensure the last month reflects the prediction.
        fig, ax = plt.subplots()
        ax.plot(months, fsi_trend, marker='o', label="FSI")
        ax.set_title("Predicted FSI Trend")
        ax.set_ylabel("FSI")
        ax.grid(True)
        st.pyplot(fig)
    
    # --- Batch Prediction Preview ---
    if uploaded_file:
        st.subheader("📂 Batch Prediction Result")
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Validate columns
            expected_cols = list(input_data.columns)
            if not all(col in batch_df.columns for col in expected_cols):
                st.error(f"⚠️ Uploaded CSV must contain these columns: {expected_cols}")
            else:
                # Predict
                batch_preds = model_fsi.predict(batch_df[expected_cols])
                batch_df["Predicted_FSI"] = np.round(batch_preds, 3)

            st.success("✅ Predictions complete!")
            st.dataframe(batch_df)

            # Download button
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="batch_fsi_predictions.csv",
                mime='text/csv'
            )

            # 📉 FSI Distribution (Light Theme)
            st.subheader("📉 Predicted FSI Distribution (Histogram)")
            fig, ax = plt.subplots()
            ax.hist(batch_df["Predicted_FSI"], bins=10, color='orange', edgecolor='black')
            ax.set_xlabel("FSI")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Predicted Food Security Index")
            st.pyplot(fig)



        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# --- PAGE: Visualizations ---
elif page == "Visualizations":
    st.title("📊 Visualizations")
    st.write("Explore visual insights on food price trends and food security predictions.")
    
    # --- Example Visualization: Price Trend ---
    st.subheader("Predicted Price Trend")
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    price_trend = [np.random.uniform(2, 5) for _ in months]
    fig, ax = plt.subplots()
    ax.plot(months, price_trend, marker='o')
    ax.set_title("Predicted Price Trend Over Months")
    ax.set_ylabel("Price Change (%)")
    ax.grid(True)
    st.pyplot(fig)
    
    # Additional visualizations like interactive feature importance charts can be added here.

# --- PAGE: Model Comparison ---
elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    st.write("Compare the performance of different models used in this project. (The values below are dummy values for demonstration.)")
    
    # --- Dummy Model Performance Metrics ---
    data = {
        "Model": ["Random Forest", "XGBoost", "LSTM", "SVR"],
        "Food Price MAPE": [np.random.uniform(5, 15) for _ in range(4)],
        "Food Price RMSE": [np.random.uniform(0.5, 2.0) for _ in range(4)],
        "FSI MAE": [np.random.uniform(0.1, 0.5) for _ in range(4)],
        "FSI RMSE": [np.random.uniform(0.1, 0.5) for _ in range(4)]
    }
    performance_df = pd.DataFrame(data)
    st.dataframe(performance_df)

# --- PAGE: About ---
elif page == "About":
    st.title("ℹ️ About")
    st.write(
        """
        **Project Title:** Predictive Modeling of Food Prices and Food Security Based on Climate Data
        
        **Objective:**  
        Build machine learning models that predict:
        - The percentage change in food prices.
        - The food security index (FSI).
        This is based on climate and food production data in Malaysia.
        
        **Data Sources:**  
        - **Climate Data:** Aggregated values like temperature, humidity, and rainfall.
        - **Food Production Data:** Crop type, production volume, and planted area.
        - **Price Data:** Median monthly food prices for items like Rice, Vegetables, Chicken & Eggs, Seafood, Palm oil, and Fruits.
        
        **Models Used:**  
        - **Baseline:** Random Forest.  
        - **Others:** XGBoost, LSTM, and potentially SVR.
        
        **Evaluation Metrics:**  
        - **Food Price:** MAPE and RMSE.  
        - **FSI:** MAE and RMSE.
        
        **UI Functionality:**  
        - **Dashboard:** Provides an overview and project description.
        - **Predict Price & Predict FSI:** Separate pages for individual predictions.
        - **Visualizations:** Includes trend charts and, potentially, feature importance.
        - **Model Comparison:** Compare performance metrics across different models.
        
        This application is built with Streamlit to demonstrate our approach toward predictive modeling for sustainable food systems.
        """
    )
