
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
import os
import gc

def load_data_in_chunks(file_path, chunk_size=10**6):
    """
    Load a large CSV file in chunks and combine it into one DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        chunk_size (int): Number of rows to read per chunk (default = 1,000,000).

    Returns:
        pd.DataFrame: Combined DataFrame with all loaded data.
    """
    print('\nStep 1: Loading pre-filtered training data in chunks...')
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        del chunk  # free memory
    print('\nStep 2: Combining chunks...')
    df_train = pd.concat(chunks, ignore_index=True)
    print(f"✅ Combined dataset: {df_train.shape[0]:,} rows")
    print('\nStep 3: Cleanup...')
    del chunks
    gc.collect()
    print('✅ Data loading complete!')
    return df_train

def summarize_data(df, target_col='unit_sales'):
    """
    Print summary statistics of the dataset.

    Args:
        df (pd.DataFrame): The dataset to summarize.
        target_col (str): Name of the target column (default = 'unit_sales').
    """
    print(f'Dataset shape: {df.shape}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'Number of stores: {df["store_nbr"].nunique()}')
    print(f'Number of items: {df["item_nbr"].nunique()}')
    print(f"Average daily {target_col}: {df[target_col].mean():.2f}")
    print(f"Zero {target_col} percentage: {(df[target_col] == 0).mean() * 100:.2f}%")

def log_xgboost_run(
    model,
    X_train, y_train,
    X_test, y_test,
    run_name="xgboost_run",
    experiment_name="Retail Forecasting",
    plot_sample_size=30,
    notes="",
    tracking_uri="file:///content/mlruns"
):
    """
    Train an XGBoost model, evaluate it, and log everything to MLflow.

    Args:
        model: An instance of an XGBoost model (e.g., XGBRegressor).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        run_name (str): Name for this MLflow run.
        experiment_name (str): MLflow experiment name.
        plot_sample_size (int): Number of predictions to plot.
        notes (str): Optional notes to log with the run.
        tracking_uri (str): Path to MLflow tracking URI (default is local folder).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Plot actual vs predicted
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values[:plot_sample_size], label='Actual')
        plt.plot(y_pred[:plot_sample_size], label='Predicted')
        plt.title(f"{run_name} (Sample: {plot_sample_size})")
        plt.legend()
        plt.tight_layout()
        plot_path = "eval_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        if notes:
            with open("notes.txt", "w") as f:
                f.write(notes)
            mlflow.log_artifact("notes.txt")

        signature = infer_signature(X_test, y_pred)
        input_example = X_test.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        for file in [plot_path, "notes.txt"]:
            if os.path.exists(file):
                os.remove(file)

        print(f"✅ MLflow run logged: {run_name}")

def plot_weekly_aggregation(y_true, y_pred, date_index, title="Weekly Aggregated Forecast vs Actual"):
    """
    Plot weekly-aggregated actual vs predicted values.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        date_index (array-like): Corresponding datetime values (same length as y_true).
        title (str): Plot title.
    """

    # Set style for better looking plots
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette('Set2')
    plt.rcParams['font.size'] = 10

    df_eval = pd.DataFrame({
        'date': pd.to_datetime(date_index),
        'actual': y_true,
        'predicted': y_pred
    })

    df_weekly = df_eval.groupby(df_eval['date'].dt.to_period('W').dt.start_time).agg({
        'actual': 'sum',
        'predicted': 'sum'
    }).reset_index().rename(columns={'date': 'week_start'})

    plt.figure(figsize=(12, 6))
    plt.plot(df_weekly['week_start'], df_weekly['actual'], label='Actual Weekly Sales', linewidth=2)
    plt.plot(df_weekly['week_start'], df_weekly['predicted'], label='Predicted Weekly Sales', linestyle='--', linewidth=2)
    plt.xlabel('Week')
    plt.ylabel('Unit Sales')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def evaluate_forecast(y_true, y_pred, model=None, X_test=None, features=None, model_type="xgboost"):
    """
    Evaluate time series model performance and optionally print model/feature insights.
    
    Parameters:
        y_true: Actual values
        y_pred: Predicted values
        model: Trained model (optional)
        X_test: Test features (optional, needed for accuracy)
        features: Feature names (optional, needed for feature importance)
        model_type: "xgboost", "lstm", "sarima", "arima"

    Returns:
        metrics: dict of evaluation scores
        feature_importance: DataFrame (optional)
    """
    epsilon = 1e-10  # To avoid division by zero

    bias = np.mean(y_pred - y_true)
    mad = np.mean(np.abs(y_true - y_pred))
    rmad = mad / np.mean(y_true)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Accuracy score only for models with .score()
    accuracy = None
    if model and hasattr(model, "score") and X_test is not None:
        try:
            accuracy = model.score(X_test, y_true)
        except:
            accuracy = None

    metrics = {
        "Bias": bias,
        "MAD": mad,
        "rMAD": rmad,
        "SMAPE": smape,
        "RMSE": rmse
    }
    if accuracy is not None:
        metrics["Accuracy"] = accuracy

    print("Time Series Evaluation Metrics")
    print("=" * 40)
    for k, v in metrics.items():
        if k == "SMAPE":
            print(f"{k:>8}: {v:.2f}%")
        else:
            print(f"{k:>8}: {v:.4f}")

    # Model Info
    if X_test is not None:
        print(f"\nModel Info:")
        print(f"Number of features: {X_test.shape[1]}")
        print(f"Test samples      : {len(X_test):,}")

    # Feature Importance (for tree-based models)
    feature_importance = None
    if model_type == "xgboost" and model and hasattr(model, "feature_importances_") and features is not None:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    return metrics, feature_importance
