# src/predict.py

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocess import merge_data, feature_engineering


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def load_model():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


def get_race_id(circuit_name):
    """Resolve a user-provided circuit name to the latest raceId **with results**.

    Matches against race name, circuit name, and circuitRef, but only returns
    a race that actually has entries in results.csv to avoid empty race data
    later in the pipeline.
    """

    races = pd.read_csv(DATA_DIR / "races.csv")
    circuits = pd.read_csv(DATA_DIR / "circuits.csv")
    results = pd.read_csv(DATA_DIR / "results.csv")

    # Avoid column name clashes (both tables have a `name` column)
    df = races.merge(circuits, on="circuitId", suffixes=("_race", "_circuit"))

    # Restrict to races that actually appear in the results dataset
    df = df.merge(results[["raceId"]].drop_duplicates(), on="raceId", how="inner")

    # Normalise text columns for case-insensitive matching
    df["name_race"] = df["name_race"].str.lower()
    df["name_circuit"] = df["name_circuit"].str.lower()
    df["circuitRef"] = df["circuitRef"].str.lower()

    query = circuit_name.strip().lower()

    matched = df[
        df["name_race"].str.contains(query, na=False)
        | df["name_circuit"].str.contains(query, na=False)
        | df["circuitRef"].str.contains(query, na=False)
    ]

    if matched.empty:
        raise ValueError(f"Circuit '{circuit_name}' not found or has no results data")

    # Take the most recent race (by year) at this circuit that has results
    latest = matched.sort_values("year").iloc[-1]

    return latest["raceId"]


def prepare_race_input(race_id):
    df = merge_data()
    df = feature_engineering(df)

    race_df = df[df["raceId"] == race_id].copy()

    if race_df.empty:
        raise ValueError(f"No data for raceId {race_id}")

    return race_df


def predict_by_circuit(circuit_name):
    model = load_model()

    race_id = get_race_id(circuit_name)

    race_df = prepare_race_input(race_id)

    X = race_df[[
        "raceId",
        "driverId",
        "constructorId",
        "grid",
        "avg_pos_last5",
        "constructor_avg_pos"
    ]]

    scores = model.predict(X)

    race_df["pred_score"] = scores

    ranked = race_df.sort_values("pred_score", ascending=False)

    top3 = ranked.head(3)[["driverId", "pred_score"]]

    drivers = pd.read_csv("data/drivers.csv")

    top3 = top3.merge(
        drivers[["driverId", "forename", "surname"]],
        on="driverId",
        how="left"
    )

    top3["name"] = top3["forename"] + " " + top3["surname"]

    return top3[["name", "pred_score"]]