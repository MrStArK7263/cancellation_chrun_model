"""
preprocessing.py
----------------
Feature-engineering code shared by the notebook AND the API.
Keeping it in ONE file means training and prediction use exactly the same logic
(this avoids "training-serving skew").
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MONTH_TO_QUARTER = {
    "Jan": "Q1", "Feb": "Q1", "Mar": "Q1",
    "Apr": "Q2", "May": "Q2", "Jun": "Q2",
    "Jul": "Q3", "Aug": "Q3", "Sep": "Q3",
    "Oct": "Q4", "Nov": "Q4", "Dec": "Q4",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the cleaned raw columns and returns the DataFrame with new features added."""
    df = df.copy()

    # '-' in DeviceType means "no device information"
    df["DeviceType"] = df["DeviceType"].replace("-", "Unknown")

    # --- ratios / differences ---
    df["UsageChange"] = df["CurrentPeriodUsage"] - df["PriorPeriodUsage"]
    df["UsageChangePct"] = df["UsageChange"] / (df["PriorPeriodUsage"] + 1)
    df["FeePerUsage"] = df["BaseFee"] / df["CurrentPeriodUsage"]
    df["PenaltiesPerTenure"] = df["NumberOfPenalties"] / (df["CustomerTenure"] + 1)
    df["TicketsPerTenure"] = df["ServiceTickets"] / (df["CustomerTenure"] + 1)

    # --- interaction feature ---
    df["PenaltyFeeInteraction"] = df["NumberOfPenalties"] * df["BaseFee"]

    # --- binning (fixed edges so train and prediction behave the same) ---
    df["TenureGroup"] = pd.cut(
        df["CustomerTenure"],
        bins=[-np.inf, 80, 100, 120, np.inf],
        labels=["New", "Growing", "Established", "Loyal"],
    ).astype(str)

    # --- flags / date based ---
    df["HasDeviceInfo"] = (df["DeviceType"] != "Unknown").astype(int)
    df["StartQuarter"] = df["StartMonth"].map(MONTH_TO_QUARTER)
    return df


def build_preprocessor(num_cols, cat_cols):
    """Scale numeric columns, one-hot encode categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
        ]
    )


def build_pipeline(num_cols, cat_cols, model):
    """Full pipeline: raw columns -> feature engineering -> scaling/encoding -> model."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features)),
            ("prep", build_preprocessor(num_cols, cat_cols)),
            ("model", model),
        ]
    )
