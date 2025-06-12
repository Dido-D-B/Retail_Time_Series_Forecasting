import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import datetime
import base64

# PAGE CONFIGURATION
st.set_page_config(page_title="Retail Forecasting", layout="centered")

# MOBILE READABILITY ADJUSTMENTS
# Improved transparent container styling for main content
st.markdown("""
<style>
/* Target the main content container more reliably */
.main .block-container,
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div,
section[data-testid="stSidebar"] ~ div > div > div > div {
    /* Frosted glass look - more transparent */
    background-color: rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px); /* Safari support */
    
    /* Spacing & rounding - only top corners */
    padding: 2rem !important;
    border-radius: 12px 12px 0 0 !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    
    /* Constrain width & center on wide screens */
    max-width: 960px;
    margin: 1.5rem auto !important;
    
    /* Ensure content is readable */
    position: relative;
    z-index: 1;
}

/* Alternative approach - target the main content area */
.stApp > div > div > div > div:first-child {
            background-color: rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    padding: 2rem !important;
    border-radius: 12px 12px 0 0 !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    max-width: 960px;
    margin: 1.5rem auto !important;
}

/* Most reliable approach - target by class */
div.block-container {
    background-color: rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    padding: 2rem !important;
    border-radius: 12px 12px 0 0 !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    max-width: 960px;
    margin: 1.5rem auto !important;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    .main .block-container,
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div,
    .stApp > div > div > div > div:first-child,
    div.block-container {
        background-color: rgba(0, 0, 0, 0.4) !important;
        color: #f0f0f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
}

/* Ensure sidebar doesn't get the same styling */
section[data-testid="stSidebar"] {
    background: transparent !important;
}

/* Remove any conflicting styles */
.stApp {
    /* Keep your background image styles */
}
</style>
""", unsafe_allow_html=True)

# BACKGROUND IMAGE
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    encoded_img = base64.b64encode(img_data).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("04_Forecast_App/background.png")

# Transparent container Main Content
st.markdown("""
<style>
/* 1) Style the entire main content area (not the sidebar) */
div[data-testid="stAppViewContainer"] > main {
    /* frosted‐glass look */
    background-color: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: saturate(180%) blur(20px);
    /* spacing & rounding */
    padding: 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
    /* constrain width & center on wide screens */
    max-width: 960px;
    margin: 1.5rem auto !important;
}
/* 2) Dark‐mode tweak */
@media (prefers-color-scheme: dark) {
  div[data-testid="stAppViewContainer"] > main {
    background-color: rgba(0, 0, 0, 0.75) !important;
    color: #f0f0f0 !important;
  }
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA & MODEL
df_input = pd.read_csv(
    "04_Forecast_App/df_input_light.csv", parse_dates=["date"]
    )

model = joblib.load(
    "/Users/didodeboodt/Documents/Projects/Retail_Forecasting_Project/03_Models/xgb_best_model.pkl"
    )

model_features = [
    'store_nbr', 'item_nbr', 'onpromotion', 'city', 'state', 'store_type', 'store_cluster',
    'item_family', 'item_class', 'perishable', 'holiday_type', 'holiday_locale',
    'holiday_transferred', 'year', 'quarter', 'month', 'day', 'day_of_week', 'is_weekend',
    'day_of_year', 'week_of_year', 'is_holiday', 'is_transferred_holiday', 'is_actual_holiday',
    'is_additional_event', 'is_transfer_day', 'is_national_event', 'is_local_event',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'unit_sales_3d_avg', 'unit_sales_7d_avg',
    'unit_sales_14d_avg', 'unit_sales_30d_avg', 'unit_sales_7d_std', 'unit_sales_14d_std'
]

# SIDEBAR
item_sales = df_test.groupby('item_nbr')['unit_sales'].sum().sort_values(ascending=False)
sorted_items = item_sales.index.tolist()

with st.sidebar:
    st.header("Forecast Input")
    store_nbr = st.selectbox("Select Store", options=sorted(stores))
    item_nbr = st.selectbox("Select Item", options=sorted_items)
    forecast_date = st.date_input(
        "Select Forecast Date",
        value=pd.to_datetime(dates[0]).date()
    )

# MAIN CONTENT
st.title("Retail Sales Forecasting App")
st.subheader("Predict daily unit sales for a selected store, item, and date.")

st.markdown("""
Welcome to the Retail Sales Forecasting App!  
This tool uses machine learning to predict daily unit sales for a specific store-item-date combination, based on historical patterns and calendar events.

**Project by [Dido De Boodt](https://www.linkedin.com/in/dido-de-boodt/)**  
Built using Python, XGBoost, and Streamlit.  
Special thanks to [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/overview) for the dataset! ❤️ 
""")

with st.expander("ℹ️ About this model"):
    st.markdown("""
    - **Model**: XGBoost Regressor (tree-based)  
    - **Features**: Time-based lags, rolling averages, and calendar signals  
    - **Target**: Daily unit sales  
    - **Training Range**: 2013–2014  
    - **Test RMSE**: ~7.78  
    """)

# PREDICTION LOGIC
row = df_input[
    (df_input['store_nbr'] == store_nbr) &
    (df_input['item_nbr'] == item_nbr) &
    (df_input['date'] == pd.to_datetime(forecast_date))
]

if row.empty:
    st.error("No data available for this combination. Please choose a different store, item, or date.")
else:
    input_data = row[model_features].copy()

    # Convert types
    categorical_cols = [
        'onpromotion', 'city', 'state', 'store_type', 'item_family',
        'holiday_type', 'holiday_locale', 'holiday_transferred',
        'is_transferred_holiday', 'is_actual_holiday', 'is_additional_event',
        'is_transfer_day', 'is_national_event', 'is_local_event'
    ]
    for col in categorical_cols:
        input_data[col] = input_data[col].astype("category")
    input_data['is_holiday'] = input_data['is_holiday'].astype(bool)

    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)[0]

    st.markdown(f"""
        <div style='background-color:#D98B82;padding:20px;border-radius:10px;text-align:center'>
            <h2 style='color:#ffffff;'>Forecasted Sales: <strong>{int(prediction)}</strong> units</h2>
        </div>
    """, unsafe_allow_html=True)

    # Historical Chart
    history = df_test[
        (df_test['store_nbr'] == store_nbr) &
        (df_test['item_nbr'] == item_nbr)
    ].sort_values("date")

    if not history.empty:
        st.markdown("### Historical Sales Trend")
        st.line_chart(history.set_index("date")["unit_sales"])

    # Forecast Chart
    future_dates = pd.date_range(start=pd.to_datetime(forecast_date), periods=10)
    future_rows = df_input[
        (df_input['store_nbr'] == store_nbr) &
        (df_input['item_nbr'] == item_nbr) &
        (df_input['date'].isin(future_dates))
    ].sort_values("date")

    future_input = future_rows[model_features].copy()
    future_input["is_holiday"] = future_input["is_holiday"].astype(bool)
    for col in categorical_cols:
        future_input[col] = future_input[col].astype("category")

    if not future_rows.empty:
        future_preds = model.predict(future_input)
        forecast_df = pd.DataFrame({
            "Date": future_rows["date"].values,
            "Store": store_nbr,
            "Item": item_nbr,
            "Forecasted Sales": future_preds.astype(int)
        })

        st.markdown("### 10-Day Forecast")
        st.markdown("""
        This forecast is based on historical sales trends and model predictions, not real-time inventory or promotions.
        """)
        st.markdown(f"**Forecast Date Range:** {forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}")
        st.line_chart(forecast_df.set_index("Date")["Forecasted Sales"])

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 10-Day Forecast as CSV",
            data=csv,
            file_name=f"forecast_store{store_nbr}_item{item_nbr}.csv",
            mime='text/csv'
        )