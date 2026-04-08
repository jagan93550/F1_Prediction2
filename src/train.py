# src/train.py

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import merge_data, feature_engineering


def train():
    df = merge_data()
    df = feature_engineering(df)

    X = df[[
        "raceId",
        "driverId",
        "constructorId",
        "grid",
        "avg_pos_last5",
        "constructor_avg_pos"
    ]]

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained & saved!")


if __name__ == "__main__":
    train()