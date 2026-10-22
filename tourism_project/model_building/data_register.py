"""
Data registration / validation script.

This runs as the first job in the pipeline. It reads the dataset that lives
inside the repo (tourism_project/data/tourism.csv), confirms that every column
we expect is actually present, and prints a short summary so we have a record
of what the pipeline started with. If a required column is missing it exits
with a non-zero status so the workflow fails early instead of training on bad
data.
"""

import os
import sys
import pandas as pd

# the dataset is committed inside the repo, so the path is relative to the repo root
DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

# every column we rely on later in prep / training
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

TARGET = "ProdTaken"


def main():
    # make sure the file is actually there before doing anything else
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # check that none of the expected columns are missing
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print("ERROR: the dataset is missing expected columns:", missing)
        sys.exit(1)

    # if we got here the schema is fine - print a quick summary for the logs
    print("Dataset registered successfully.")
    print("-" * 50)
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nTarget distribution ('{TARGET}'):")
    print(df[TARGET].value_counts())
    print("\nTarget distribution (proportion):")
    print(df[TARGET].value_counts(normalize=True).round(3))
    print("-" * 50)
    print("Validation passed: all expected columns are present.")


if __name__ == "__main__":
    main()
