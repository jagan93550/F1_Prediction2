# src/preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    base_dir = Path(__file__).resolve().parents[1] / "data"

    results = pd.read_csv(base_dir / "results.csv")
    races = pd.read_csv(base_dir / "races.csv")
    drivers = pd.read_csv(base_dir / "drivers.csv")
    circuits = pd.read_csv(base_dir / "circuits.csv")

    return results, races, drivers, circuits


def merge_data():
    results, races, drivers, circuits = load_data()

    # Clean missing values
    results = results.replace(r"\\N", np.nan, regex=True)

    df = results.merge(races, on="raceId")
    df = df.merge(drivers, on="driverId")
    df = df.merge(circuits, on="circuitId")

    # Convert numeric
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

    df = df.dropna(subset=["positionOrder", "grid"])

    return df


def feature_engineering(df):
    df = df.sort_values(by=["driverId", "raceId"])

    # Driver form
    df["avg_pos_last5"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Constructor form
    df["constructor_avg_pos"] = (
        df.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    mean_pos = df["positionOrder"].mean()

    df["avg_pos_last5"] = df["avg_pos_last5"].fillna(mean_pos)
    df["constructor_avg_pos"] = df["constructor_avg_pos"].fillna(mean_pos)

    # Target
    df["target"] = 1 / df["positionOrder"]

    return df