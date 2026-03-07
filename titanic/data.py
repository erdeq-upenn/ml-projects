import numpy as np
import pandas as pd
import seaborn as sns
from typing import NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Column order after encoding (matches X array indices)
FEATURE_COLS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "family_size", "is_alone", "has_cabin", "fare_per_person", "age_x_class", "embarked_Q", "embarked_S"]
COL = {name: i for i, name in enumerate(FEATURE_COLS)}


class DataSplit(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_titanic() -> DataSplit:
    df = sns.load_dataset("titanic")

    # Compute has_cabin before split — derived from raw deck column, no stat leakage
    df["has_cabin"] = df["deck"].notna().astype(int)

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "has_cabin", "embarked"]
    df = df[features + ["survived"]].copy()

    df["sex"] = (df["sex"] == "male").astype(int)

    # Feature engineering on full df (pure row-wise arithmetic, no leakage)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Split before stat-derived imputation to prevent leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["survived"])

    # Compute imputation stats from train only, apply to both
    age_median = train_df["age"].median()
    fare_median = train_df["fare"].median()
    embarked_mode = train_df["embarked"].mode()[0]

    for split in (train_df, test_df):
        split["age"] = split["age"].fillna(age_median)
        split["fare"] = split["fare"].fillna(fare_median)
        split["embarked"] = split["embarked"].fillna(embarked_mode)

    # Compute features that depend on imputed age/fare (row-wise only, no leakage)
    for split in (train_df, test_df):
        split["fare_per_person"] = (split["fare"] / split["family_size"]).clip(lower=0)
        split["age_x_class"] = split["age"] * split["pclass"]

    # Explicit categories ensure stable column schema regardless of data slice
    for split in (train_df, test_df):
        split["embarked"] = pd.Categorical(split["embarked"], categories=["C", "Q", "S"])

    train_df = pd.get_dummies(train_df, columns=["embarked"], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=["embarked"], drop_first=True)

    # Reorder columns to exactly match FEATURE_COLS
    output_cols = FEATURE_COLS + ["survived"]
    train_df = train_df[output_cols]
    test_df = test_df[output_cols]

    X_train = train_df.drop("survived", axis=1).values.astype(np.float32)
    y_train = train_df["survived"].values.astype(np.float32)
    X_test = test_df.drop("survived", axis=1).values.astype(np.float32)
    y_test = test_df["survived"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return DataSplit(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)