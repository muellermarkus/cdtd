import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from evaluation.eval_ml_efficiency import calculate_metrics


def train_catboost(X, y, cat_features, task, params, seed=0, logging_level="Verbose"):
    """
    Inspired by https://github.com/yandex-research/tab-ddpm/blob/main/scripts/eval_catboost.py
    """

    if task == "regression":
        catboost = CatBoostRegressor(
            **params,
            eval_metric="RMSE",
            random_seed=seed,
            logging_level=logging_level,
            allow_const_label=True,
        )
        predict = catboost.predict
    else:
        # for mult_class need to get number of classes
        n_classes = len(np.unique(y["train"]))

        catboost = CatBoostClassifier(
            loss_function="MultiClass" if task == "mult_class" else "Logloss",
            **params,
            eval_metric="TotalF1:average=Macro",
            random_seed=seed,
            class_names=[str(i) for i in range(n_classes)]
            if task == "mult_class"
            else ["0", "1"],
            logging_level=logging_level,
            # allow_const_label=True,
        )
        predict = (
            catboost.predict_proba
            if task == "mult_class"
            else lambda x: catboost.predict_proba(x)[:, 1]
        )

    catboost.fit(
        X["train"],
        y["train"],
        eval_set=(X["val"], y["val"]),
        cat_features=cat_features,
        verbose=100,
    )

    y_pred = {k: predict(v) for k, v in X.items()}
    results = {
        split: calculate_metrics(y[split], y_pred[split], task) for split in y.keys()
    }

    return results
