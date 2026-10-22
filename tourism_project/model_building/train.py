"""
Model training script with experiment tracking.

Loads the train/test splits produced by the prep job, builds a preprocessing +
XGBoost pipeline, tunes it with GridSearchCV, logs the tuned parameters and
evaluation metrics to MLflow, and saves the best pipeline into the deployment
folder so the workflow can commit it back to the repo for the Streamlit app.
"""

import os
import joblib
import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)
from xgboost import XGBClassifier

# where the trained model gets saved (the workflow commits this path back to the repo)
DEPLOY_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_tourism_model_v1.joblib")

# the prep job dropped these as features; list the ones we feed the model
NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]


def main():
    # --- load the splits produced by the previous job (they sit at the repo root) ---
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")

    # --- preprocessing: impute then scale numerics, impute then one-hot the categoricals ---
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    # the target is imbalanced (~81% / 19%), so weight the positive class
    neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
    scale_pos_weight = round(neg / pos, 2)

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # --- hyperparameters to tune ---
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
    }

    # f1 is a sensible target given the class imbalance
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)

    # --- log everything to MLflow (local file store - no server needed) ---
    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run():
        grid.fit(Xtrain, ytrain)
        best = grid.best_estimator_

        # evaluate the tuned model on the held-out test set
        preds = best.predict(Xtest)
        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1": f1_score(ytest, preds),
            "cv_best_f1": grid.best_score_,
        }

        # log the tuned parameters and the metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metrics(metrics)

        print("Best parameters:", grid.best_params_)
        print("\nTest metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report:\n", classification_report(ytest, preds))

        # --- save the best pipeline so the workflow can commit it ---
        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        print("Saved best model to", MODEL_PATH)


if __name__ == "__main__":
    main()
