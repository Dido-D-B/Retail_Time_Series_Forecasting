# Retail Time Series Forecasting

<img width="758" alt="Screenshot 2025-06-11 at 22 05 35" src="https://github.com/user-attachments/assets/ce9dcdfa-9cfd-4442-990e-b116bc795cd2" />

--- 

Forecasting daily unit sales for an Ecuadorian grocery retailer using advanced time series and machine learning methods.
This end-to-end project covers data preparation, feature engineering, modeling, experiment tracking, app deployment, and presentation.

**Goal**: improving inventory planning and promoting strategies through more accurate sales forecasts.  

## Project Structure

```
├── 00_Data/              # Raw & processed datasets
│   └── Raw_Data/         # CSVs: items, stores, holidays, oil, transactions
├── 01_Notebooks/         # EDA, preprocessing, modeling (XGBoost, LSTM, SARIMA)
├── 02_MLflow/            # Experiment tracking logs
├── 03_Models/            # Saved models (.pkl, .h5)
├── 04_Forecast_App/      # Streamlit app files
├── 05_Presentation/      # Slide deck + executive summary report
├── Utils/                # Utility functions (feature engineering, evaluation)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Project Overview

- **Business context**: Support demand planning for a major Ecuadorian supermarket chain
- **Dataset**: Kaggle’s Corporación Favorita dataset (~3.39M rows, 41 features), can be found [here](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting). 
- **Challenge**: Sparse data (47% zero sales), multiple stores, seasonal & holiday effects
- **Models tested**:
  - XGBoost (best performing, RMSE: 7.77, Bias: 0.173 ✅)
  - LSTM (RMSE: 7.80, lowest MAD & SMAPE ✅)
  - SARIMA (interpretable baseline, RMSE: 7.89)
- **Outcome**: Deployed a Streamlit app to forecast sales by store-item-date, with 10-day projections and historical visualizations  

## Methods & Workflow

### Preprocessing & Feature Engineering

- Cleaned anomalies, filled missing dates
- Created time features (day_of_week, is_weekend, day_of_year, month)
- Added lag features (1, 7, 14, 28 days)
- Built rolling averages & std (7d, 14d, 30d)
- Added holiday/event indicators (national, local, transferred)  ￼

### Modeling

- **XGBoost** → best balance of speed, interpretability, accuracy
- **LSTM** → captured long-term patterns, strong with SMAPE, sensitive to tuning
- **SARIMA** → interpretable, but lower performance on sparse data

**Evaluation metrics**: RMSE, MAD, SMAPE, Bias

## Results

<img width="738" height="346" alt="Screenshot 2025-09-25 at 21 25 52" src="https://github.com/user-attachments/assets/7ad0af52-650a-406b-a0b3-0d3093b5608f" />

## Forecast App

Built with Streamlit to make forecasts accessible for decision-makers. Check out the app [here](https://retailforecastingapp-nwuukzuxmjkoqed66zcxcu.streamlit.app/).

**Features**:

- Select store, item, and date → forecast generated
- 10-day forecast with historical trend plot
- Download option for results
- Backend powered by the trained XGBoost model

![Screenshot 2025-06-13 at 03 06 14](https://github.com/user-attachments/assets/f4955f3c-60c3-46a6-963e-aa27d834c8bc)

## Key Takeaways

- Engineered time-based features gave the biggest boost in predictive power
- Tree-based ML models (XGBoost) outperformed both deep learning (LSTM) and statistical baselines (SARIMA) on sparse tabular data
- LSTM showed potential with larger datasets and hyperparameter tuning
- Deployed app bridges technical modeling with real business usability  ￼

Getting Started

1. Clone the repo

```
git clone https://github.com/Dido-D-B/Retail-Forecasting.git
cd Retail-Forecasting
```

2. Install dependencies

```
pip install -r requirements.txt
```

3.	Run notebooks for EDA & modeling

```
jupyter notebook 01_Notebooks/
```

4.	Launch the forecast app

```
streamlit run 04_Forecast_App/forecastapp.py
```

## Acknowledgments

- Dataset: Corporación Favorita Grocery Sales Forecasting (Kaggle)
- Tools: Python, Streamlit, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, MLflow
- Created by [Dido De Boodt](https://www.linkedin.com/in/dido-de-boodt/) as a project for Masterschool
