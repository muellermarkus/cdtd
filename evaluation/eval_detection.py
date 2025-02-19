from functools import partial

import numpy as np
import optuna
import pandas as pd
from omegaconf import OmegaConf

from train_catboost import train_catboost
from tune_catboost import suggest_catboost_params


class DetectionScore:
    """Estimates detection score using catboost model."""

    def __init__(self, X_cat, X_cont, y, y_cond, task, tune_model=True):
        self.y_cond = y_cond
        self.task = task
        self.tune_model = tune_model

        # add y onto features if necessary
        if self.y_cond:
            if self.task == "regression":
                X_cont = {
                    split: np.column_stack((y[split], X_cont[split]))
                    for split in ["train", "val", "test"]
                }
            else:
                X_cat = {
                    split: np.column_stack((y[split], X_cat[split]))
                    for split in ["train", "val", "test"]
                }

        train_obs = (
            X_cat["train"].shape[0]
            if X_cat["train"] is not None
            else X_cont["train"].shape[0]
        )
        test_obs = (
            X_cat["test"].shape[0]
            if X_cat["test"] is not None
            else X_cont["test"].shape[0]
        )
        val_obs = 0
        if X_cat["val"] is not None or X_cont["val"] is not None:
            val_obs = (
                X_cat["val"].shape[0]
                if X_cat["val"] is not None
                else X_cont["val"].shape[0]
            )

        # subsample if necessary
        X_cat["train"], X_cont["train"], self.sub_train_obs = self.subsample(
            X_cat, X_cont, "train", train_obs
        )
        X_cat["val"], X_cont["val"], self.sub_val_obs = self.subsample(
            X_cat, X_cont, "val", val_obs
        )
        X_cat["test"], X_cont["test"], self.sub_test_obs = self.subsample(
            X_cat, X_cont, "test", test_obs
        )
        self.num_samples = self.sub_train_obs + self.sub_val_obs + self.sub_test_obs
        self.X_cat = X_cat
        self.X_cont = X_cont

    def prep_data(self, X_cat_gen, X_cont_gen, y_gen):
        if self.y_cond:
            if self.task == "regression":
                X_cont_gen = np.column_stack((y_gen, X_cont_gen))
            else:
                X_cat_gen = np.column_stack((y_gen, X_cat_gen))

        train_idx = range(self.sub_train_obs)
        val_idx = range(self.sub_train_obs, self.sub_train_obs + self.sub_val_obs)
        test_idx = range(
            self.sub_train_obs + self.sub_val_obs,
            self.sub_train_obs + self.sub_val_obs + self.sub_test_obs,
        )
        idx = {"train": train_idx, "val": val_idx, "test": test_idx}

        # construct data with 50% fake 50% real observations
        X = {}
        y = {}
        for split in idx.keys():
            X_fake = pd.concat(
                (
                    pd.DataFrame(X_cat_gen[idx[split]]).astype(int),
                    pd.DataFrame(X_cont_gen[idx[split]]).astype(float),
                ),
                axis=1,
                ignore_index=True,
            )
            X_real = pd.concat(
                (
                    pd.DataFrame(self.X_cat[split]).astype(int),
                    pd.DataFrame(self.X_cont[split]).astype(float),
                ),
                axis=1,
                ignore_index=True,
            )

            # y = 1 if fake, y = 0 if real
            labels_real = np.zeros((X_real.shape[0],))
            labels_fake = np.ones((X_fake.shape[0],))
            X[split] = pd.concat((X_real, X_fake), axis=0, ignore_index=True)
            y[split] = np.concatenate((labels_real, labels_fake))

            # shuffle data
            X[split] = X[split].sample(frac=1, random_state=42)
            y[split] = y[split][X[split].index]

        if self.X_cat["train"] is not None:
            cat_features = list(range(self.X_cat["train"].shape[1]))
        else:
            cat_features = None

        return X, y, cat_features

    def subsample(self, X_cat, X_cont, split, split_obs, max_obs=25000):
        if split_obs > max_obs:
            n_obs = max_obs
            idx = np.random.choice(split_obs, max_obs, replace=False)
            if X_cat[split] is not None:
                X_cat_sub = X_cat[split][idx]
            else:
                X_cat_sub = None

            if X_cont[split] is not None:
                X_cont_sub = X_cont[split][idx]
            else:
                X_cont_sub = None

            return X_cat_sub, X_cont_sub, n_obs

        else:
            n_obs = split_obs

            return X_cat[split], X_cont[split], n_obs

    def tune_detection_model(self, X_cat_gen, X_cont_gen, y_gen, n_trials=50):
        default_params = OmegaConf.load("configs/catboost/default.yaml")

        if self.tune_model:
            data, labels, cat_features = self.prep_data(X_cat_gen, X_cont_gen, y_gen)

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
            self.best_params = study.best_trial.user_attrs["params"]
        else:
            self.best_params = default_params

        return self

    def estimate_score(self, X_cat_gen, X_cont_gen, y_gen):
        data, labels, cat_features = self.prep_data(X_cat_gen, X_cont_gen, y_gen)

        results = train_catboost(
            data,
            labels,
            cat_features,
            "bin_class",
            params=self.best_params["model"],
            logging_level="Silent",
        )

        return results["test"]["accuracy"]
