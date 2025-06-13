# Retail Time Series Forecasting Project

<img width="758" alt="Screenshot 2025-06-11 at 22 05 35" src="https://github.com/user-attachments/assets/ce9dcdfa-9cfd-4442-990e-b116bc795cd2" />


This project focuses on forecasting daily unit sales for a large Ecuadorian grocery chain using time series modeling and machine learning. The goal is to improve inventory planning and business insights by predicting future sales based on historical patterns and calendar events.

## Project Structure

- `00_Data/`: Raw, processed, and engineered data
- `01_Notebooks/`: Exploratory data analysis, modeling notebooks
- `02_MLflow/`: Model tracking and experiment logging
- `03_Models/`: Saved models (XGBoost, LSTM, SARIMA)
- `04_Forecast_App/`: Streamlit app files (background image, model, and UI)
- `05_Presentation/`: Slide deck and reports
- `Utils/`: Utility scripts

## Models Used

- **XGBoost Regressor** (final model)
- LSTM (deep learning baseline)
- ARIMA/SARIMA (classical baselines)

## Evaluation Metrics

- RMSE  
- Bias  
- SMAPE  
- MAD, rMAD

## Forecast App

A deployed Streamlit app allows users to select a store, item, and date to get a forecast + 10-day projection. See the [App Repo](https://github.com/Dido-D-B/RetailForecasting_App) and the [Deployed App](https://retailforecastingapp-nwuukzuxmjkoqed66zcxcu.streamlit.app/).

![Screenshot 2025-06-13 at 03 06 14](https://github.com/user-attachments/assets/f4955f3c-60c3-46a6-963e-aa27d834c8bc)


## Requirements

Install packages with:

```bash
pip install -r requirements.txt

## Presentation

You can find the slide deck [here](https://docs.google.com/presentation/d/1_6X_8SS0RpAmwbGZBB6WTW7p0CXxH4zm5st-aIMQjsU/edit?usp=sharing)
