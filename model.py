import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data(path="data/crime_data.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    return df

def train_predict(df):
    # Crimes per year
    yearly = df.groupby("Year").size().reset_index(name="Crimes")

    # Train model
    X = yearly["Year"].values.reshape(-1,1)
    y = yearly["Crimes"].values

    model = LinearRegression()
    model.fit(X, y)

    # Future years prediction
    future_years = np.array([2024, 2025, 2026]).reshape(-1,1)
    preds = model.predict(future_years)

    return yearly, future_years.flatten(), preds
