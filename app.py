import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import io
#import seaborn as sns

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

        /* Base Page Styling */
        html, body, .stApp {
            background-color: #0e1117;
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
        }

        /* Main container and sidebar */
        .block-container, .css-1d391kg { 
            background-color: #0e1117 !important;
        }

        /* Top Header Bar */
        header[data-testid="stHeader"] {
            background-color: #0e1117 !important;
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

        /* Buttons */
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
scaler_price = joblib.load("models/price_scaler.pkl")

import os
from datetime import datetime

def save_prediction(pred_type, category, item, input_df, prediction, file_path="data/predictions.csv"):
    """
    Save prediction results into a CSV for visualization and logging.

    Args:
        pred_type (str): "Price" or "FSI"
        category (str): e.g. "Banana"
        item (str): e.g. "PISANG BERANGAN"
        input_df (pd.DataFrame): DataFrame with 1 row of input features
        prediction (float): prediction result
        file_path (str): where to store the CSV
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    record = input_df.copy()
    record["prediction_type"] = pred_type
    record["food_category"] = category
    record["item"] = item
    record["prediction_result"] = prediction
    record["timestamp"] = datetime.today().strftime("%Y-%m-%d")
    
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        updated = pd.concat([existing, record], ignore_index=True)
    else:
        updated = record

    updated.to_csv(file_path, index=False)



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

        #uploaded_file = st.file_uploader("📂 Or upload a CSV file for batch prediction", type=["csv"])
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
    return submitted, input_data, uploaded_file

# --- PAGE: Homepage ---
if page == "Homepage":
    st.title("🥦 Homepage")
    st.write(
        """
        Welcome to the Food Price & Food Security Predictor Homepage!
        
        **Project Overview:**
        - **FYP Title:** Predictive Modeling of Food Prices and Food Security Based on Agriculture Climate Data.
        - **Objective:** Predict median price (RM) of food prices and the food security index (FSI) using climate and food production data in Malaysia.
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

    # 📘 How Our Model Understands Food Price Dynamics
    st.markdown("### 📘 How Our Model Understands Food Price Dynamics")

    try:
        from PIL import Image
        image = Image.open("data/correlation_matrix.png")
        st.image(image, caption="Correlation Matrix of Key Features", use_container_width=True)

        with st.expander("ℹ️ How to Read This Chart"):
            st.markdown("""
            - **+1.0** = Strong positive relationship  
            - **-1.0** = Strong negative relationship  
            - Dark red cells show high positive correlation; blue cells show inverse relationships.
            - Example: High correlation between `lag_1_price` and `median_price` shows past prices are strong predictors of future prices.
            """)
    except Exception as e:
        st.warning(f"⚠️ Unable to display correlation image. Make sure 'data/correlation_matrix.png' exists. Error: {e}")

    # 🛠️ How the System Works
    st.markdown("### 🛠️ How This Prediction System Works")
    st.info("""
    1. **Choose Food Item**: Select from categories like Rice, Vegetables, or Fruits.
    2. **Input Data**: Enter agricultural, climate, and price data manually OR upload a CSV.
    3. **Make Predictions**:
       - 📈 **Price Prediction**: Forecast future median food prices (in RM).
       - 📉 **FSI Prediction**: Forecast Food Security Index based on supply, affordability, and climate impact.
    4. **Understand Results**:
       - Visualize future trends (line charts, distributions).
       - See which factors affect predictions the most (feature importance charts).
    5. **Download Reports**: Save your results for offline analysis or reporting.
    """)

    # 📄 CSV Upload Requirements
    st.markdown("### 📄 CSV Upload Requirements (Feature Descriptions)")
    st.markdown("""
    Below is the list of features your CSV file should contain. Make sure column names match **exactly** and values are numeric where applicable.

    | **Feature Name**          | **Description**                                       |
    |---------------------------|-------------------------------------------------------|
    | `production`              | Crop output (in tons)                                |
    | `planted_area`            | Area of land planted (in hectares)                   |
    | `temperature`             | Monthly average temperature (°C)                     |
    | `humidity`                | Monthly average humidity (%)                         |
    | `precipitation_rate`      | Average rainfall intensity (mm/hr)                   |
    | `monthly_total_rainfall`  | Total monthly rainfall (mm)                          |
    | `rainfall_impact`         | Precomputed score (0–1) of rainfall effect           |
    | `lag_1_price`             | Price 1 month ago                                    |
    | `lag_2_price`             | Price 2 months ago                                   |
    | `rolling_3m_price`        | Avg price over last 3 months                         |
    | `rolling_6m_price`        | Avg price over last 6 months                         |
    """, unsafe_allow_html=True)

# --- PAGE: Predict Price ---
if page == "Predict Price":
    st.title("🔮 Predict Median Food Price ")
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

    with st.expander("ℹ️ Notes on Prediction Results", expanded=False):
        st.markdown("""
        - **Predicted values** reflect the **median food price** or **FSI** for the next month based on input data.
        - The model assumes monthly-level data (e.g., average temperature, rainfall total, etc.).
        - Charts shown are simulated forecasts assuming conditions remain steady.
        - Results are based on historical patterns and should not be considered financial or agricultural advice.
        - All predictions are saved and used for trend visualization later in the app.
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
                        batch_df["Predicted_Median_Price_RM"] = np.round(batch_preds, 2)

                        st.success("✅ Prediction complete!")
                        st.dataframe(batch_df)

                        # Save each prediction to predictions.csv
                        for idx, row in batch_df.iterrows():
                            save_prediction(
                                pred_type="Price",
                                category=selected_food,
                                item=row["item"],
                                input_df=row[expected_cols].to_frame().T,
                                prediction=row["Predicted_Median_Price_RM"]
                            )


                        st.subheader("📊 Price Comparison (Grouped Bar Chart)")
                        import matplotlib.pyplot as plt
                        import numpy as np

                        items = batch_df["item"]
                        x = np.arange(len(items))  # label locations
                        width = 0.25  # width of each bar

                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')

                        bars1 = ax.bar(x - width, batch_df["lag_1_price"], width, label='Lag 1 Price')
                        bars2 = ax.bar(x, batch_df["rolling_3m_price"], width, label='3M Rolling Price')
                        bars3 = ax.bar(x + width, batch_df["Predicted_Median_Price_RM"], width, label='Predicted Price')

                        # Labels and styling
                        ax.set_xlabel('Item', color='white')
                        ax.set_ylabel('Price (RM)', color='white')
                        ax.set_title('Predicted vs Historical Food Prices', color='white')
                        ax.set_xticks(x)
                        ax.set_xticklabels(items, rotation=45, ha="right", color='white')
                        ax.legend()

                        # Set background color and tick color
                        fig.patch.set_facecolor('#1e1e1e')
                        ax.set_facecolor('#1e1e1e')
                        ax.tick_params(colors='white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')

                        st.pyplot(fig)

                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="predicted_median_price.csv",
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

            # Save result to predictions.csv
            save_prediction(
                pred_type="Price",
                category=selected_food,
                item=category_to_items[selected_food][0],  # default 1st item
                input_df=input_data,
                prediction=price_pred
            )


            st.metric("Predicted Median Price (RM)", f"RM {price_pred:.2f}")

            # --- Forecast Chart ---
            st.subheader("📉 Price Trend Forecast")
            forecast_input = input_data.copy()
            price_trend = []
            months = ["Jan", "Feb", "Mar", "Apr", "May"]

            for _ in range(5):
                scaled = scaler_price.transform(forecast_input)
                reshaped = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
                pred = model_price.predict(reshaped)[0][0]
                price_trend.append(pred)

                # Update lag and rolling features
                forecast_input["lag_2_price"] = forecast_input["lag_1_price"]
                forecast_input["lag_1_price"] = forecast_input["rolling_3m_price"]
                forecast_input["rolling_3m_price"] = forecast_input["rolling_6m_price"]
                forecast_input["rolling_6m_price"] = pred


            # Plotting
            fig, ax = plt.subplots()
            ax.plot(months, price_trend, marker='o', label="Median Price (RM)")
            ax.set_title("Predicted Median Price Trend")
            ax.set_ylabel("Price (RM)")
            ax.grid(True)
            st.pyplot(fig)

# --- PAGE: Predict FSI ---
elif page == "Predict FSI":
    st.title("🔮 Predict Food Security Index (FSI)")
    st.info("""
    👉 To predict the Food Security Index (FSI), please input:

    - 🌾 **Production**: The quantity of food available. Higher production improves supply and lowers scarcity risk.
    - 🌾 **Planted Area**: Indicates potential sustainability and capacity to produce food in the region.
    - 🌦️ **Temperature, Humidity, Rainfall**: Climate-related conditions that affect how easily food can be grown and distributed.
    - 💧 **Rainfall Impact Score**: Indicates whether rainfall was supportive or damaging to crops.
    - 💸 **Lagged Prices**: Market prices 1–2 months ago. Higher prices reduce food affordability.
    - 📊 **Rolling Prices (3M & 6M)**: Average food prices in the past 3 and 6 months — a key indicator of long-term affordability and access to food.
    """)

    with st.expander("ℹ️ Notes on Prediction Results", expanded=False):
        st.markdown("""
        - **Predicted values** reflect the **median food price** or **FSI** for the next month based on input data.
        - The model assumes monthly-level data (e.g., average temperature, rainfall total, etc.).
        - Charts shown are simulated forecasts assuming conditions remain steady.
        - Results are based on historical patterns and should not be considered financial or agricultural advice.
        - All predictions are saved and used for trend visualization later in the app.
        """)


    # Food category selection
    food_options = [
        "Banana",
        "Papaya",
        "Chili",
        "Vegetables",
        "Rice (Super, Premium, Import)",
    ]

    selected_food = st.selectbox("🍽️ Select the type of food to evaluate FSI for", food_options)
    st.markdown(f"📌 You selected: **{selected_food}**")

    # Define food category mappings
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

    # Upload CSV
    uploaded_file = st.file_uploader("📂 Upload CSV for batch prediction", type=["csv"])

    # --- BATCH PREDICTION ---
    if uploaded_file is not None:
        try:
            # Read CSV
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            raw_df = pd.read_csv(stringio)

            st.success("✅ File uploaded successfully!")
            st.subheader("📝 Preview & Edit Your Data")
            batch_df = st.data_editor(raw_df, num_rows="dynamic", use_container_width=True)

            # Submit batch for prediction
            if st.button("📊 Submit for Prediction"):
                expected_cols = list(batch_df.columns.intersection([
                    "production", "planted_area", "temperature", "humidity",
                    "precipitation_rate", "monthly_total_rainfall", "rainfall_impact",
                    "lag_1_price", "lag_2_price", "rolling_3m_price", "rolling_6m_price"
                ]))

                # Validate
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
                        # Prediction
                        batch_preds = model_fsi.predict(batch_df[expected_cols])
                        batch_df["Predicted_FSI"] = np.round(batch_preds, 3)

                        # Show results
                        st.success("✅ Prediction complete!")
                        st.dataframe(batch_df)

                        # --- Feature Importance (Same for all rows since Random Forest is not row-specific)
                        st.subheader("📊 Feature Importance (FSI Model - Random Forest)")
                        importances = model_fsi.feature_importances_
                        feat_names = expected_cols
                        
                        importance_df = pd.DataFrame({
                            "Feature": feat_names,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False)
                        
                        st.dataframe(importance_df)
                        
                        # Bar Plot
                        st.subheader("🔍 Visual Breakdown of Feature Impact")
                        fig3, ax3 = plt.subplots(facecolor='#1e1e1e')
                        ax3.barh(importance_df["Feature"], importance_df["Importance"], color='orange')
                        ax3.set_xlabel("Importance", color='white')
                        ax3.set_title("Which Factors Influence FSI Most?", color='white')
                        ax3.tick_params(colors='white')
                        fig3.patch.set_facecolor('#1e1e1e')
                        st.pyplot(fig3)


                        # ✅ Save each prediction to predictions.csv
                        for idx, row in batch_df.iterrows():
                            save_prediction(
                                pred_type="FSI",
                                category=selected_food,
                                item=row["item"],
                                input_df=row[expected_cols].to_frame().T,
                                prediction=row["Predicted_FSI"]
                            )


                        # Chart
                        st.subheader("📉 Predicted FSI Distribution")
                        fig, ax = plt.subplots(facecolor='#1e1e1e')
                        ax.hist(batch_df["Predicted_FSI"], bins=10, color='orange', edgecolor='white')
                        ax.set_xlabel("FSI", color='white')
                        ax.set_ylabel("Frequency", color='white')
                        ax.set_title("Distribution of Predicted FSI", color='white')
                        ax.tick_params(colors='white')
                        fig.patch.set_facecolor('#1e1e1e')
                        st.pyplot(fig)

                        # Download
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="predicted_fsi_results.csv",
                            mime='text/csv'
                        )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    # --- MANUAL INPUT ---
    else:
        st.markdown("""
        <div style='border-top: 1px solid #555; margin: 30px 0 10px;'></div>
        <p style='text-align: center; color: #aaa; font-weight: 600; font-family: "Poppins", sans-serif;'>
            OR you can input manually below
        </p>
        <div style='border-top: 1px solid #555; margin: 10px 0 30px;'></div>
        """, unsafe_allow_html=True)

        # Display the input form
        submitted, input_data, _ = get_input_data()

        if submitted:
            st.subheader("📊 Prediction Result")
            fsi_pred = model_fsi.predict(input_data)[0]
            st.metric("Predicted Food Security Index", f"{fsi_pred:.3f}")

            # --- Feature Importance for Random Forest (FSI)
            st.subheader("📊 Feature Importance (FSI Model - Random Forest)")
            importances = model_fsi.feature_importances_
            feat_names = input_data.columns.tolist()

            importance_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df)

            # Bar plot
            st.subheader("🔍 Visual Breakdown of Feature Impact")
            fig, ax = plt.subplots(facecolor='#1e1e1e')
            ax.barh(importance_df["Feature"], importance_df["Importance"], color='orange')
            ax.set_xlabel("Importance", color='white')
            ax.set_title("Which Factors Influence FSI Most?", color='white')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#1e1e1e')
            st.pyplot(fig)

            # Forecast Chart
            st.subheader("📉 FSI Forecast")
            forecast_input = input_data.copy()
            fsi_trend = []
            months = ["Jan", "Feb", "Mar", "Apr", "May"]
            
            # 📈 Predict 5 future months
            for _ in range(5):
                pred = model_fsi.predict(forecast_input)[0]
                fsi_trend.append(pred)

                # Update lag features
                forecast_input["lag_2_price"] = forecast_input["lag_1_price"]
                forecast_input["lag_1_price"] = forecast_input["rolling_3m_price"]
                forecast_input["rolling_3m_price"] = forecast_input["rolling_6m_price"]
                forecast_input["rolling_6m_price"] = pred 



            fig, ax = plt.subplots()
            ax.plot(months, fsi_trend, marker='o', label="FSI")
            ax.set_title("Predicted Food Security Index Trend")
            ax.set_ylabel("FSI")
            ax.grid(True)
            st.pyplot(fig)

# --- PAGE: Visualizations ---
elif page == "Visualizations":
    st.title("📊 Visualizations")
    st.markdown("Explore predictions you've made using the app — visualize trends and patterns.")

    try:
        chunks = pd.read_csv("data/predictions.csv", chunksize=5000)
        df = pd.concat(chunks, ignore_index=True)
        st.success("✅ Loaded predictions.csv successfully!")
    except FileNotFoundError:
        st.error("❌ predictions.csv not found in /data folder. Please make predictions first.")
    else:
        # Step 1: Prediction type
        prediction_types = df["prediction_type"].unique().tolist()
        selected_type = st.selectbox("📂 Select Prediction Type", prediction_types)

        # Step 2: Food Category
        category_options = df[df["prediction_type"] == selected_type]["food_category"].unique().tolist()
        selected_category = st.selectbox("🍽️ Select Food Category", category_options)

        # Step 3: Item (Optional)
        item_options = df[
            (df["prediction_type"] == selected_type) & 
            (df["food_category"] == selected_category)
        ]["item"].unique().tolist()
        selected_item = st.selectbox("🥕 Select Item (Optional)", ["All"] + item_options)

        # Filter DataFrame
        filtered_df = df[
            (df["prediction_type"] == selected_type) & 
            (df["food_category"] == selected_category)
        ]

        if selected_item != "All":
            filtered_df = filtered_df[filtered_df["item"] == selected_item]

        # Convert timestamp column properly
        filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], errors='coerce')
        filtered_df = filtered_df.sort_values("timestamp")

        # --- Line Chart ---
        st.subheader("📈 Prediction Trend Over Time")

        import matplotlib.dates as mdates

        fig, ax = plt.subplots()
        ax.plot(filtered_df["timestamp"], filtered_df["prediction_result"], marker='o', linestyle='-')
        ax.set_title(f"{selected_type} Prediction Trend")
        
        # Format x-axis nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Example: Jan 2025
        fig.autofmt_xdate()

        ax.set_xlabel("Month / Year")
        ax.set_ylabel("Predicted Median Price (RM)" if selected_type == "Price" else "Predicted FSI")
        ax.grid(True)
        st.pyplot(fig)

        st.caption("Shows how predictions vary over time based on the timestamp of each prediction.")
        with st.expander("ℹ️ What does this chart mean?"):
            st.write(f"""
                This line chart visualizes the trend of {selected_type} predictions over time.
                
                - **X-axis**: Represents months and years.
                - **Y-axis**: Shows the predicted values.
                - Helps you track if {selected_type.lower()} is rising, falling, or stable over time.
            """)

        # --- Histogram ---
        st.subheader("📊 Distribution of Predictions")
        
        # Define bins dynamically (based on min-max prediction values)
        min_pred = filtered_df["prediction_result"].min()
        max_pred = filtered_df["prediction_result"].max()
        bin_width = 0.5 if selected_type == "Price" else 0.05  # Smaller bins for FSI (0-1 range)
        
        bins = np.arange(min_pred, max_pred + bin_width, bin_width)
        
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered_df["prediction_result"], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_title(f"{selected_type} Distribution")
        ax2.set_xlabel("Predicted Median Price (RM)" if selected_type == "Price" else "Predicted FSI")
        ax2.set_ylabel("Number of Predictions")
        ax2.grid(axis='y')
        st.pyplot(fig2)

        st.caption("Shows how predictions are distributed — are most predictions concentrated around certain values?")
        with st.expander("ℹ️ What does this chart mean?"):
            st.write(f"""
                This histogram displays how many predictions fall into specific value ranges.

                - It helps detect **clusters** (e.g., prices around RM8–RM9) or **spread** (diverse prices).
                - Useful for understanding **market volatility** (for price) or **stability** (for FSI).
            """)

# --- PAGE: Model Comparison ---
elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    st.write("Compare the performance of different models used in this project.")

    try:
        
        data = {
            "Model": ["Random Forest", "XGBoost", "LSTM", "SVR", "LightGBM"],
            "Food Price MAPE": [0.0065, 0.0548, 0.0017, 0.0073, 0.0394],
            "Food Price RMSE (RM)": [0.6095, 1.4920, 0.0356, 0.2951, 1.0896],
            "FSI MAE": [0.0040, 0.0093, 0.0289, 0.0460, 0.0056],
            "FSI RMSE": [0.0136, 0.0233, 0.0990, 0.0629, 0.0153]
        }

        performance_df = pd.DataFrame(data)

        
        st.subheader("📈 Model Performance Summary")
        st.dataframe(performance_df, use_container_width=True)

        
        st.markdown("### 📊 Understanding the Evaluation Metrics")
        st.info("""
        - **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy as a percentage. Lower is better.
        - **RMSE (Root Mean Squared Error)**: Shows how spread out the errors are. Lower values indicate better performance.
        - **MAE (Mean Absolute Error)**: Measures the average magnitude of prediction errors. Lower is better.

        **Interpretation:**
        - For **Food Price Prediction**, we mainly use **MAPE** and **RMSE**.
        - For **Food Security Index (FSI) Prediction**, we mainly use **MAE** and **RMSE**.
        - A lower score generally indicates a more accurate and reliable model.
        """)

    except Exception as e:
        st.error(f"⚠️ Unable to load model comparison results: {e}")

# --- PAGE: About ---
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### 🛠️ Project Title:
    **Predictive Modeling of Food Prices and Food Security Based on Agriculture Climate Data**

    ### 🎯 Objective:
    This project builds machine learning models to predict:
    - **Median Food Prices (RM)** for selected food items in Malaysia.
    - **Food Security Index (FSI)** scores that reflect the accessibility, availability, and stability of food systems.

    ### 🗂️ Data Sources:
    - **Climate Data**: Aggregated monthly data on temperature, humidity, and rainfall.
    - **Food Production Data**: Includes production quantity and planted area for key food items.
    - **Price Data**: Median monthly food prices collected for items like rice, vegetables, fruits, poultry, and seafood.

    ### 🤖 Models Used:
    - **Random Forest** (Baseline Model)
    - **XGBoost**
    - **Long Short-Term Memory (LSTM)**
    - **Support Vector Regression (SVR)**
    - **LightGBM**

    ### 📏 Evaluation Metrics:
    - **Food Price Predictions**: MAPE (Mean Absolute Percentage Error) and RMSE (Root Mean Squared Error)
    - **FSI Predictions**: MAE (Mean Absolute Error) and RMSE

    ### 🧩 System Features:
    - **Homepage**: Project overview and user guidance.
    - **Predict Price**: Predict median prices manually or through batch uploads.
    - **Predict FSI**: Predict Food Security Index scores.
    - **Visualizations**: Interactive charts to explore predictions made in the system.
    - **Model Comparison**: Compare the accuracy and performance of different predictive models.

    ### 🌱 Contribution to Sustainability:
    By forecasting food prices and food security risks, this project aims to support:
    - Better decision-making for food affordability.
    - Proactive planning to ensure food stability under climate uncertainties.
    - Efforts toward achieving **Sustainable Development Goal (SDG) 2: Zero Hunger**.

    ---
    """)

