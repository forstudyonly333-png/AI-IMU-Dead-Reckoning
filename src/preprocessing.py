import pandas as pd
import numpy as np


def preprocess_imu(df):

    df = df.copy()

    # Remove missing values
    df = df.dropna()

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Calculate time difference
    df["dt"] = df["timestamp"].diff()

    # First sample
    df["dt"] = df["dt"].fillna(
        df["dt"].median()
    )

    # Avoid invalid time steps
    df["dt"] = df["dt"].clip(
        lower=0.001,
        upper=0.1
    )

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/demo_imu.csv"
    )

    df = preprocess_imu(df)

    df.to_csv(
        "data/processed/clean_imu.csv",
        index=False
    )

    print(df.head())
    print("\nPreprocessing complete!")