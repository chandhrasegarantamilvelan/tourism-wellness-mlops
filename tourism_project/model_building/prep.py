"""
Data preparation script.

Loads the registered dataset, does a bit of cleaning, drops the columns we
don't want to train on, and splits everything into train / test sets. The four
split files are written to the repo root so the workflow can hand them to the
model-training job as an artifact.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"

# columns that are just identifiers / row numbers - no predictive value
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]


def main():
    df = pd.read_csv(DATA_PATH)

    # drop the index column and the customer id if they are present
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # --- small bit of cleaning on the categorical values ---
    # 'Fe Male' is just a typo for 'Female'
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 'Unmarried' and 'Single' mean the same thing here, so merge them
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # separate the target from the features
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # stratified split so the 80/20 class balance is kept in both sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # save the splits at the repo root - the workflow uploads these as an artifact
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}  |  Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}  |  ytest: {ytest.shape}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
