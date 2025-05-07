import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- PAGE CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Food Price & Security Predictor",
    page_icon="🍚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM THEME (for .streamlit/config.toml) ---
# [theme]
# primaryColor = "#1b4332"
# backgroundColor = "#0e1117"
# secondaryBackgroundColor = "#2d6a4f"
# textColor = "#f5f5f5"
# font = "sans serif"

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📂 App Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Homepage", "Predict Price", "Predict FSI", "Visualizations", "About", "FAQ"]
)

# --- ONBOARDING / HELP ---
if "onboarded" not in st.session_state:
    with st.sidebar.expander("👋 First time here? Start here!", expanded=True):
        st.markdown("""
        - Use the sidebar to navigate between pages.
        - Each page has tooltips and explanations.
        - For help, see the FAQ or About page.
        """)
    st.session_state["onboarded"] = True

# --- SECTION DIVIDER ---
def section_divider():
    st.markdown("<hr style='border:1px solid #4CAF50;'>", unsafe_allow_html=True)

# --- INPUT & FORMS (with grouping, tooltips, validation) ---
def get_input_data():
    with st.form(key="input_form"):
        st.markdown("#### 🌾 Agricultural Inputs")
        col1, col2 = st.columns(2)
        with col1:
            production = st.number_input(
                "Production (tons)", min_value=0.0, value=1000.0,
                help="Total crop output in tons for the selected month."
            )
        with col2:
            planted_area = st.number_input(
                "Planted Area (hectares)", min_value=0.0, value=50.0,
                help="Total area used for planting, in hectares."
            )
        st.markdown("#### 🌦️ Climate Data")
        col3, col4, col5 = st.columns(3)
        with col3:
            temperature = st.number_input(
                "Temperature (°C)", min_value=0.0, max_value=50.0, value=27.0,
                help="Average monthly temperature in degrees Celsius."
            )
        with col4:
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0, max_value=100.0, value=80.0,
                help="Average monthly humidity as a percentage."
            )
        with col5:
            precip_rate = st.number_input(
                "Precipitation Rate (mm/hr)", min_value=0.0, value=5.0,
                help="Average rainfall rate in millimeters per hour."
            )
        col6, col7 = st.columns(2)
        with col6:
            total_rain = st.number_input(
                "Monthly Rainfall (mm)", min_value=0.0, value=200.0,
                help="Total rainfall for the month in millimeters."
            )
        with col7:
            rain_impact = st.number_input(
                "Rainfall Impact Index", min_value=0.0, max_value=1.0, value=0.5,
                help="A score (0–1) indicating how rainfall affects crops."
            )
        st.markdown("#### 📈 Price History")
        col8, col9, col10, col11 = st.columns(4)
        with col8:
            lag_1 = st.number_input(
                "Lag 1 Month Price", min_value=0.0, value=2.5,
                help="Median price of the food item one month ago (in RM)."
            )
        with col9:
            lag_2 = st.number_input(
                "Lag 2 Month Price", min_value=0.0, value=2.4,
                help="Median price of the food item two months ago (in RM)."
            )
        with col10:
            roll_3m = st.number_input(
                "Rolling 3M Price", min_value=0.0, value=2.6,
                help="Average price over the last 3 months (in RM)."
            )
        with col11:
            roll_6m = st.number_input(
                "Rolling 6M Price", min_value=0.0, value=2.7,
                help="Average price over the last 6 months (in RM)."
            )
        submitted = st.form_submit_button("📊 Submit for Prediction")
    # Validation feedback
    errors = []
    if production < 0: errors.append("Production must be non-negative.")
    if planted_area < 0: errors.append("Planted area must be non-negative.")
    if not (0 <= temperature <= 50): errors.append("Temperature must be 0-50°C.")
    if not (0 <= humidity <= 100): errors.append("Humidity must be 0-100%.")
    if not (0 <= rain_impact <= 1): errors.append("Rainfall Impact must be 0-1.")
    if errors:
        for e in errors:
            st.error(e)
        return False, None
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
    return submitted, input_data

# --- HOMEPAGE ---
if page == "Homepage":
    st.title("🍚 Food Price & Security Predictor")
    st.markdown("""
    Welcome! This app predicts food prices and food security using climate and agricultural data.
    Use the sidebar to navigate. For help, see the FAQ or About page.
    """)
    section_divider()
    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. Go to **Predict Price** or **Predict FSI**
    2. Enter your data or upload a CSV
    3. View predictions and download results
    4. Explore trends in the Visualizations tab
    """)

# --- PREDICT PRICE ---
if page == "Predict Price":
    st.title("🔮 Predict Median Food Price")
    section_divider()
    st.info("Fill in the form below. Hover over any input for help.")
    submitted, input_data = get_input_data()
    if submitted and input_data is not None:
        with st.spinner("Generating prediction..."):
            time.sleep(1)  # Simulate processing
            prediction = np.random.uniform(2, 5)
            st.success(f"Predicted Median Price: RM {prediction:.2f}")
            st.balloons()
            st.download_button("Download Result as CSV", input_data.to_csv(index=False), "prediction_result.csv", "text/csv")

# --- PREDICT FSI ---
if page == "Predict FSI":
    st.title("🔮 Predict Food Security Index (FSI)")
    section_divider()
    st.info("Fill in the form below. Hover over any input for help.")
    submitted, input_data = get_input_data()
    if submitted and input_data is not None:
        with st.spinner("Generating FSI prediction..."):
            time.sleep(1)
            prediction = np.random.uniform(0, 1)
            st.success(f"Predicted FSI: {prediction:.2f}")
            st.snow()
            st.download_button("Download Result as CSV", input_data.to_csv(index=False), "fsi_prediction_result.csv", "text/csv")

# --- VISUALIZATIONS ---
if page == "Visualizations":
    st.title("📈 Visualizations")
    section_divider()
    st.markdown("""
    Use the tabs below to explore trends and statistics. For help, see the expanders in each tab.
    """)
    # Placeholder data
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=12, freq="M"),
        "price": np.random.uniform(2, 5, 12),
        "fsi": np.random.uniform(0, 1, 12)
    })
    tab1, tab2, tab3 = st.tabs(["Price Trends", "FSI Trends", "Summary Stats"])
    with tab1:
        st.subheader("Price Trends Over Time")
        with st.expander("How to read this chart?"):
            st.markdown("""
            - X-axis: Date
            - Y-axis: Predicted price (RM)
            - Look for trends, peaks, and dips
            """)
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (RM)")
        st.pyplot(fig)
        st.download_button("Download Data", df.to_csv(index=False), "price_trends.csv", "text/csv")
    with tab2:
        st.subheader("FSI Trends Over Time")
        with st.expander("How to read this chart?"):
            st.markdown("""
            - X-axis: Date
            - Y-axis: Food Security Index (0-1)
            - Higher is better
            """)
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["fsi"], marker="o", color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("FSI")
        st.pyplot(fig)
        st.download_button("Download Data", df.to_csv(index=False), "fsi_trends.csv", "text/csv")
    with tab3:
        st.subheader("Summary Statistics")
        with st.expander("How to read these stats?"):
            st.markdown("""
            - Mean: Average value
            - Std: Standard deviation (spread)
            - Min/Max: Range
            """)
        st.dataframe(df.describe().T)
        st.download_button("Download Stats", df.describe().T.to_csv(), "summary_stats.csv", "text/csv")

# --- ABOUT PAGE ---
if page == "About":
    st.title("ℹ️ About")
    section_divider()
    st.markdown("""
    **Food Price & Security Predictor**
    - Developed for FYP at APU
    - Predicts food prices and food security using climate and agricultural data
    - For more info, contact: your@email.com
    """)

# --- FAQ PAGE ---
if page == "FAQ":
    st.title("❓ FAQ")
    section_divider()
    with st.expander("How do I use this app?"):
        st.markdown("Use the sidebar to navigate. Fill in forms and view results.")
    with st.expander("What data do I need?"):
        st.markdown("You need climate, agricultural, and price history data.")
    with st.expander("Can I download results?"):
        st.markdown("Yes, use the download buttons on each page.")
    with st.expander("Who developed this app?"):
        st.markdown("This app was developed for a Final Year Project at APU.")

# --- FOOTER ---
section_divider()
st.caption("© 2024 Food Price & Security Predictor | Developed as FYP at Asia Pacific University") 