import numpy as np
import pandas as pd
import seaborn as sns
from typing import NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Column order after encoding (matches X array indices)
FEATURE_COLS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked_Q", "embarked_S"]
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

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[features + ["survived"]].copy()

    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    df["sex"] = (df["sex"] == "male").astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    X = df.drop("survived", axis=1).values.astype(np.float32)
    y = df["survived"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return DataSplit(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
