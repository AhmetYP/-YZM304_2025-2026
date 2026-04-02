import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from YZM304_Proje1.src.config import CSV_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


def split_data(df):
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler