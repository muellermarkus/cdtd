import argparse
import os
import time
from functools import partial

import optuna
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, StratifiedKFold

from data.data_prep import DataWrangler
from train_catboost import train_catboost


def suggest_catboost_params(trial, default_params):
    """
    See https://github.com/yandex-research/tab-ddpm/blob/main/scripts/tune_evaluation_model.py
    """

    params = dict(default_params)
    params["model"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 0.001, 1.0, log=True
    )  # loguniform
    params["model"]["depth"] = trial.suggest_int("depth", 3, 8)
    params["model"]["l2_leaf_reg"] = trial.suggest_float(
        "l2_leaf_reg", 0.1, 10.0
    )  # uniform
    params["model"]["bagging_temperature"] = trial.suggest_float(
        "bagging_temperature", 0.0, 1.0
    )  # uniform
    params["model"]["leaf_estimation_iterations"] = trial.suggest_int(
        "leaf_estimation_iterations", 1, 10
    )

    return params


def tune_catboost(data, labels, cat_features, task, metric, k_folds=5, n_trials=100):
    default_params = OmegaConf.load("configs/catboost/default.yaml")

    def objective(trial, data, labels, cat_features, skf, task, metric, default_params):
        params = suggest_catboost_params(trial, default_params)
        trial.set_user_attr("params", params)

        X = {}
        y = {}
        X["val"] = data["val"]
        y["val"] = labels["val"]

        score = 0.0
        for train_idx, test_idx in skf.split(data["train"], labels["train"]):
            X["train"] = data["train"].loc[train_idx]
            y["train"] = labels["train"][train_idx]
            X["test"] = data["train"].loc[test_idx]
            y["test"] = labels["train"][test_idx]

            fold_results = train_catboost(
                X, y, cat_features, task, params=params["model"], logging_level="Silent"
            )

            score += fold_results["test"][metric]

        return score / skf.get_n_splits()

    if task == "regression":
        skf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    objective = partial(
        objective,
        data=data,
        labels=labels,
        cat_features=cat_features,
        skf=skf,
        task=task,
        metric=metric,
        default_params=default_params,
    )

    direction = "minimize" if task == "regression" else "maximize"
    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_trial.user_attrs["params"]

    return best_params


def tune_detection_catboost(data, labels, cat_features, n_trials=50):
    default_params = OmegaConf.load("configs/catboost/default.yaml")

    def objective(trial, data, labels, cat_features, default_params):
        params = suggest_catboost_params(trial, default_params)
        trial.set_user_attr("params", params)

        results = train_catboost(
            data,
            labels,
            cat_features,
            "bin_class",
            params=params["model"],
            logging_level="Silent",
        )
        return results["val"]["accuracy"]

    objective = partial(
        objective,
        data=data,
        labels=labels,
        cat_features=cat_features,
        default_params=default_params,
    )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_trial.user_attrs["params"]

    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--debug", help="debug mode", action="store_true")
    args = parser.parse_args()
    common_config = OmegaConf.load("configs/common_config.yaml")
    default_config = OmegaConf.load("configs/catboost/default.yaml")

    data_prep = DataWrangler(
        args.dataset,
        None,
        default_config,
        common_config.val_prop,
        common_config.test_prop,
        common_config.seed,
    )

    X_cat_train, X_cont_train, y_train = data_prep.orig_data.get_orig_data(["train"])
    X, y, cat_features = data_prep.prep_for_ML_efficiency(
        X_cat_train, X_cont_train, y_train, catboost=True
    )

    if args.debug:
        training_start_time = time.time()
        _ = train_catboost(
            X, y, cat_features, data_prep.task, params=default_config["model"]
        )
        training_duration = time.time() - training_start_time
        print(f"DEBUG train duration = {(training_duration / 60):.2f}min")
    else:
        metric = "rmse" if data_prep.task == "regression" else "f1"
        tuned_params = tune_catboost(
            X, y, cat_features, data_prep.task, metric, n_trials=100
        )

        # save params
        tuned_cfg_dir = "configs/catboost/tuned"
        if not os.path.exists(tuned_cfg_dir):
            os.makedirs(tuned_cfg_dir)
        OmegaConf.save(tuned_params, f"configs/catboost/tuned/{args.dataset}.yaml")

        print(f"config saved in configs/catboost/tuned/{args.dataset}.yaml")


if __name__ == "__main__":
    main()
