"""
Week 2: Data Preprocessing & Pipeline for Logistics Analysis

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def execute_logistics_pipeline(raw_csv_path: str) -> pd.DataFrame:
    """Executes an end-to-end data cleaning, imputation, outlier treatment,

    encoding, and scaling pipeline on raw logistics data.
    """
    # ---------------------------------------------------------
    # Step 1: Data Ingestion & Structural Cleaning
    # ---------------------------------------------------------
    print("Step 1: Ingesting data and performing structural cleaning...")
    df = pd.read_csv(raw_csv_path)

    # Standardize column headers to lowercase snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicate primary keys
    df = df.drop_duplicates(subset=["shipment_id"]).copy()

    # Convert datetime columns
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

    # ---------------------------------------------------------
    # Step 2: Handling Missing Data
    # ---------------------------------------------------------
    print("Step 2: Imputing missing values using median operational logic...")
    # Remove erroneous zero or negative distance readings
    if "distance_km" in df.columns:
        df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan
        df["distance_km"] = df.groupby("shipping_mode")[
            "distance_km"
        ].transform(lambda x: x.fillna(x.median()))

    if "freight_cost_usd" in df.columns:
        df["freight_cost_usd"] = df.groupby("shipping_mode")[
            "freight_cost_usd"
        ].transform(lambda x: x.fillna(x.median()))

    # Impute missing delivery dates using calculated median transit duration
    if "delivery_date" in df.columns and "order_date" in df.columns:
        transit_medians = (
            df["delivery_date"] - df["order_date"]
        ).dt.total_seconds() / 86400
        median_days = transit_medians.median()
        df["delivery_date"] = df["delivery_date"].fillna(
            df["order_date"] + pd.Timedelta(days=median_days)
        )

    # ---------------------------------------------------------
    # Step 3: Outlier Treatment (IQR Winsorization)
    # ---------------------------------------------------------
    print("Step 3: Detecting and capping outliers (Winsorization)...")
    target_cols = [
        c for c in ["package_weight_kg", "freight_cost_usd"] if c in df.columns
    ]
    for col in target_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap extreme outliers instead of eliminating rows
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # ---------------------------------------------------------
    # Step 4: Categorical Encoding
    # ---------------------------------------------------------
    print("Step 4: One-hot encoding categorical variables...")
    categorical_cols = [
        c for c in ["shipping_mode", "warehouse_block"] if c in df.columns
    ]
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_array = encoder.fit_transform(df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index,
        )

        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # ---------------------------------------------------------
    # Step 5: Feature Normalization (Standard Scaling)
    # ---------------------------------------------------------
    print("Step 5: Normalizing continuous numerical features...")
    scaler = StandardScaler()
    num_cols = [
        c
        for c in ["distance_km", "freight_cost_usd", "package_weight_kg"]
        if c in df.columns
    ]
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    print("Preprocessing completed successfully!")
    return df


if __name__ == "__main__":
    # Example execution entrypoint
    print("Logistics Data Preprocessing Module Ready.")