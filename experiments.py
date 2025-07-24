import os
import joblib
import requests
import tempfile
import zipfile
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from mlp import MLPClassifier
from oeadaboost import OEABClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from adabord import ADABORDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from dlordinal.metrics import amae, mmae
from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    cohen_kappa_score,
)


def stream_tocuco_datasets(tmp_tocuco_path, seeds=5):
    with open(os.path.join(tmp_tocuco_path, "train_masks.pkl"), "rb") as train_masks_binary:
        train_masks = joblib.load(train_masks_binary)

    tocuco_datasets_path = os.path.join(tmp_tocuco_path, "data")

    for dataset_name in os.listdir(tocuco_datasets_path):
        dataset = pd.read_csv(os.path.join(tocuco_datasets_path, dataset_name))
        for seed in range(seeds):
            dataset_name_without_extension = dataset_name.split(".")[0]

            dataset_seed_train_mask = train_masks[f"{dataset_name_without_extension}_seed_{seed}"]
            train = dataset.loc[dataset_seed_train_mask]
            test = dataset.loc[~dataset_seed_train_mask]

            X_train = train.drop(columns=["y"]).to_numpy()
            X_test = test.drop(columns=["y"]).to_numpy()
            y_train = train["y"].to_numpy()
            y_test = test["y"].to_numpy()

            yield (X_train, X_test, y_train, y_test, dataset_name_without_extension, seed)


"""
|!| DISCLAIMER:

This script will take a long time to run, as it will train a model for each dataset in the TOC-UCO dataset, using
a different seed for each dataset.
    
→ The utility of this script is to show how the experiments published in [1] can be reproduced. 

[1] ADABORD: a novel Adaptive Boosting approach for Ordinal classificationa. Rafael Ayllón-Gavilán,
David Guijo-Rubio, Pedro A. Gutiérrez, César Hervás-Martínez, Francisco José Martínez-Estudillo.
"""

# Load TOC-UCO into a temporary directory
url = "https://www.uco.es/grupos/ayrna/datasets/TOC-UCO.zip"
response = requests.get(url)
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(response.content)
tmp_tocuco_path = tempfile.mkdtemp()
zip_ref = zipfile.ZipFile(temp_file.name, "r")
zip_ref.extractall(tmp_tocuco_path)
extracted_files = zip_ref.namelist()
tmp_tocuco_path = os.path.join(tmp_tocuco_path, "TOC-UCO")
print("TOC-UCO loaded into temporary directory:", tmp_tocuco_path)

amae_scorer = make_scorer(amae, greater_is_better=False)
SEEDS = 30
RANDOM_SEARCH_N_ITERS = 20
RANDOM_SEARCH_CV_METHOD = StratifiedKFold(n_splits=3)

adaboost_gini_param_grid = {
    "n_estimators": [50, 100, 250, 500, 1000, 2000],
    "estimator": [
        DecisionTreeClassifier(max_depth=4, criterion="gini"),
        DecisionTreeClassifier(max_depth=8, criterion="gini"),
        DecisionTreeClassifier(max_depth=16, criterion="gini"),
        DecisionTreeClassifier(max_depth=None, criterion="gini"),
    ],
}
adaboost_ogini_param_grid = {
    "n_estimators": [50, 100, 250, 500, 1000, 2000],
    "estimator": [
        DecisionTreeClassifier(max_depth=4, criterion="ogini"),
        DecisionTreeClassifier(max_depth=8, criterion="ogini"),
        DecisionTreeClassifier(max_depth=16, criterion="ogini"),
        DecisionTreeClassifier(max_depth=None, criterion="ogini"),
    ],
}
adaboost_oeab_param_grid = {
    "n_estimators": [3, 5],  # [10, 25, 50, 100, 200],
    "n_hidden": [2, 4],  # [2, 4, 6, 8],
}

MODELS = {
    "Ridge": RandomizedSearchCV(
        RidgeClassifier(class_weight="balanced"),
        scoring=amae_scorer,
        param_distributions={
            "alpha": np.logspace(-3, 3, 10),
            "fit_intercept": [True, False],
        },
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score=("raise"),
    ),
    "XGBoost": RandomizedSearchCV(
        XGBClassifier(),
        scoring=amae_scorer,
        param_distributions={
            "max_depth": [3, 5, 8],
            "n_estimators": [100, 250, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.75, 0.95, 1.0],
            "colsample_bytree": [0.75, 0.95, 1.0],
        },
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score=("raise"),
    ),
    "MLP": RandomizedSearchCV(
        MLPClassifier(class_weight="balanced"),
        scoring=amae_scorer,
        param_distributions={
            "max_iter": [100, 250, 500, 1000],
            "hidden_neurons": [4, 8, 16],
            "hidden_layers": [1, 2, 4],
        },
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score=("raise"),
    ),
    "AdaBoost": RandomizedSearchCV(
        AdaBoostClassifier(),
        scoring=amae_scorer,
        param_distributions=adaboost_gini_param_grid,
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score="raise",
    ),
    "ADABORD": RandomizedSearchCV(
        ADABORDClassifier(),
        scoring=amae_scorer,
        param_distributions=adaboost_ogini_param_grid,
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score="raise",
    ),
    "OEAdaBoost": RandomizedSearchCV(
        OEABClassifier(),
        scoring=amae_scorer,
        param_distributions=adaboost_oeab_param_grid,
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=RANDOM_SEARCH_CV_METHOD,
        n_jobs=1,
        verbose=1,
        error_score="raise",
    ),
}

results = pd.DataFrame()

for X_train, X_test, y_train, y_test, dataset_name, seed in stream_tocuco_datasets(
    tmp_tocuco_path=tmp_tocuco_path, seeds=SEEDS
):
    for model_name, model in MODELS.items():
        print(f"\n→ Fitting: {model_name} on dataset {dataset_name} using seed {seed}")
        # Set the random state for reproducibility
        if hasattr(model, "random_state"):
            model.random_state = seed
        if hasattr(model, "estimator"):
            if hasattr(model.estimator, "random_state"):
                model.estimator.random_state = seed

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        amae_score = amae(y_test, y_pred)
        mmae_score = mmae(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "amae": amae_score,
                        "mmae": mmae_score,
                        "bacc": bacc,
                        "qwk": qwk,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        print(f"Finished: {model_name} on dataset {dataset_name} using seed {seed}")

results.to_csv("results.csv", index=False)
