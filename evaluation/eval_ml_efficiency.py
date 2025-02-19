import copy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


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

    y_pred = predict(X["test"])
    results = calculate_metrics(y["test"], y_pred, task)

    return results


def subsample_trainset(X_cat_train, X_cont_train, y_train, n=50000):
    n_current = len(X_cat_train) if X_cat_train is not None else len(X_cont_train)
    idx = np.random.choice(n_current, n, replace=False)
    X_cat_train_s = X_cat_train[idx] if X_cat_train is not None else None
    X_cont_train_s = X_cont_train[idx] if X_cont_train is not None else None
    y_train_s = y_train[idx] if y_train is not None else None

    return X_cat_train_s, X_cont_train_s, y_train_s


def calculate_metrics(y_true, y_pred, task):
    if task == "regression":
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        r2 = metrics.r2_score(y_true, y_pred)
        result = {"rmse": rmse, "r2": r2}
    else:
        probs = y_pred
        labels = np.round(probs) if task == "bin_class" else probs.argmax(axis=1)
        labels = labels.astype("int64")
        metric_report = metrics.classification_report(y_true, labels, output_dict=True)

        result = {
            "accuracy": metric_report["accuracy"],
            "f1": metric_report["macro avg"]["f1-score"],
        }

        if task == "bin_class":
            try:
                roc_auc = metrics.roc_auc_score(y_true, probs)
            except:
                roc_auc = "error"
            result["roc_auc"] = roc_auc
            result["brier"] = metrics.brier_score_loss(y_true, probs)

    return result


def average_efficiency_results(results):
    models = results[0]["real"].keys()
    avg_results_real = {}
    std_results_real = {}
    avg_results_gen = {}
    std_results_gen = {}

    for model in models:
        df_real = pd.DataFrame(res["real"][model] for res in results)
        df_gen = pd.DataFrame(res["gen"][model] for res in results)
        avg_results_real[model] = df_real.mean(0).to_dict()
        std_results_real[model] = df_real.std(0).to_dict()
        avg_results_gen[model] = df_gen.mean(0).to_dict()
        std_results_gen[model] = df_gen.std(0).to_dict()

    return avg_results_real, avg_results_gen, std_results_real, std_results_gen


def get_ml_efficiency_table(avg_results_real, avg_results_gen):
    output_tbl = PrettyTable()
    models = list(avg_results_real.keys())
    metrics = list(avg_results_gen[models[0]].keys())

    output_tbl.field_names = ["model", *metrics]

    for i, df in enumerate([avg_results_real, avg_results_gen]):
        prefix = "real_"
        if i == 1:
            prefix = "synth_"

        for j, model in enumerate(models):
            if j == len(models) - 1:
                divider = True
            else:
                divider = False

            metric_values = []
            for metric_name, metric_value in df[model].items():
                metric_values.append(round(metric_value, 3))
            output_tbl.add_row([prefix + model, *metric_values], divider=divider)

    return output_tbl


def _split_y_from_X(X):
    y = X[:, 0]
    X_new = X[:, 1:]
    return y, X_new


class MLEfficiencyScores:
    def __init__(
        self, X_cat, X_cont, y, task, n_runs, catboost_cfg, seed=42, max_obs=50000
    ):
        self.X_cat = copy.deepcopy(X_cat)
        self.X_cont = copy.deepcopy(X_cont)
        self.y = copy.deepcopy(y)
        self.task = task
        self.train_obs = (
            self.X_cat["train"].shape[0]
            if self.X_cat["train"] is not None
            else self.X_cont["train"].shape[0]
        )
        self.subsample = self.train_obs > max_obs
        self.num_cat_features = (
            self.X_cat["train"].shape[1] if self.X_cat["train"] is not None else 0
        )
        self.y_cond = self.y["train"] is not None
        self.n_runs = n_runs
        self.catboost_cfg = catboost_cfg
        self.seed = seed

    def prep_data(self, X_cat, X_cont, y, catboost=False):
        """
        Transforms data for the use sklearn models to estimate ML efficiency.
        """
        # load data in place of training data (usually load generated or real train set)
        X_cat_train = X_cat
        X_cont_train = X_cont
        y_train = y

        # get val and test data
        X_cat_val, X_cont_val, y_val = (
            self.X_cat["val"],
            self.X_cont["val"],
            self.y["val"],
        )
        X_cat_test, X_cont_test, y_test = (
            self.X_cat["test"],
            self.X_cont["test"],
            self.y["test"],
        )

        # remove y from X_cat or X_cont if we treated it as a feature during generation
        if not self.y_cond:
            if self.task == "regression":
                if X_cont_train is not None:
                    y_train, X_cont_train = _split_y_from_X(X_cont_train)
                if X_cont_val is not None:
                    y_val, X_cont_val = _split_y_from_X(X_cont_val)
                if X_cont_test is not None:
                    y_test, X_cont_test = _split_y_from_X(X_cont_test)
            else:
                if X_cat_train is not None:
                    y_train, X_cat_train = _split_y_from_X(X_cat_train)
                if X_cat_val is not None:
                    y_val, X_cat_val = _split_y_from_X(X_cat_val)
                if X_cat_test is not None:
                    y_test, X_cat_test = _split_y_from_X(X_cat_test)

        if not catboost:
            cont_enc = MinMaxScaler()
            cat_enc = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )

            if self.X_cat["train"] is not None:
                X_cat_train_trans = cat_enc.fit_transform(X_cat_train)
                X_cat_val_trans = cat_enc.transform(X_cat_val)
                X_cat_test_trans = cat_enc.transform(X_cat_test)
            else:
                X_cat_train_trans, X_cat_val_trans, X_cat_test_trans = None, None, None

            if self.X_cont["train"] is not None:
                X_cont_train_trans = cont_enc.fit_transform(X_cont_train)
                X_cont_val_trans = cont_enc.transform(X_cont_val)
                X_cont_test_trans = cont_enc.transform(X_cont_test)
            else:
                X_cont_train_trans, X_cont_val_trans, X_cont_test_trans = (
                    None,
                    None,
                    None,
                )

        # transform outcome depending on prediction task
        if self.task == "regression":
            y_enc = StandardScaler()
            y_train_trans = y_enc.fit_transform(y_train.reshape(-1, 1))
            y_val_trans = y_enc.transform(y_val.reshape(-1, 1))
            y_test_trans = y_enc.transform(y_test.reshape(-1, 1))
            y_prepped = {
                "train": y_train_trans,
                "val": y_val_trans,
                "test": y_test_trans,
            }
        else:
            y_prepped = {"train": y_train, "val": y_val, "test": y_test}

        if catboost:
            X_train = pd.concat(
                (
                    pd.DataFrame(X_cat_train).astype(int),
                    pd.DataFrame(X_cont_train).astype(float),
                ),
                axis=1,
                ignore_index=True,
            )
            X_val = pd.concat(
                (
                    pd.DataFrame(X_cat_val).astype(int),
                    pd.DataFrame(X_cont_val).astype(float),
                ),
                axis=1,
                ignore_index=True,
            )
            X_test = pd.concat(
                (
                    pd.DataFrame(X_cat_test).astype(int),
                    pd.DataFrame(X_cont_test).astype(float),
                ),
                axis=1,
                ignore_index=True,
            )
            if self.num_cat_features > 0:
                cat_features = list(range(X_cat_train.shape[1]))
            else:
                cat_features = None
        else:
            # combine categorical and continuous features
            X_train = pd.concat(
                (
                    pd.DataFrame(X_cat_train_trans).astype(int),
                    pd.DataFrame(X_cont_train_trans).astype(float),
                ),
                axis=1,
                ignore_index=True,
            ).to_numpy()
            X_val = pd.concat(
                (
                    pd.DataFrame(X_cat_val_trans).astype(int),
                    pd.DataFrame(X_cont_val_trans).astype(float),
                ),
                axis=1,
                ignore_index=True,
            ).to_numpy()
            X_test = pd.concat(
                (
                    pd.DataFrame(X_cat_test_trans).astype(int),
                    pd.DataFrame(X_cont_test_trans).astype(float),
                ),
                axis=1,
                ignore_index=True,
            ).to_numpy()

        X = {"train": X_train, "val": X_val, "test": X_test}

        if catboost:
            return X, y_prepped, cat_features
        else:
            return X, y_prepped

    def _get_catboost_results(self, X, y, cat_features, model_seed):
        return train_catboost(
            X,
            y,
            cat_features,
            self.task,
            params=self.catboost_cfg.model,
            seed=model_seed,
            logging_level="Silent",
        )

    def _get_sklearn_results(self, X, y, model_seed):
        if self.task != "regression":
            models = {
                "forest": RandomForestClassifier(
                    max_depth=12, n_estimators=100, random_state=model_seed
                ),
                "linear": LogisticRegression(max_iter=1000, random_state=model_seed),
            }
        else:
            models = {
                "forest": RandomForestRegressor(
                    max_depth=12, n_estimators=100, random_state=model_seed
                ),
                "linear": Ridge(max_iter=1000, random_state=model_seed),
            }

        results = {}
        for model_key, model in models.items():
            model.fit(X["train"], y["train"])

            if self.task == "regression":
                predict = model.predict
            elif self.task == "mult_class":
                predict = model.predict_proba
            elif self.task == "bin_class":
                predict = lambda f: model.predict_proba(f)[:, 1]  # p(y=1)

            y_pred = predict(X["test"])
            results[model_key] = calculate_metrics(y["test"], y_pred, self.task)

        return results

    def compute_ml_eff_score(self, X_cat_gen, X_cont_gen, y_gen):
        # subsample training data if necessary
        if self.subsample:
            X_cat_train_s, X_cont_train_s, y_train_s = subsample_trainset(
                self.X_cat["train"], self.X_cont["train"], self.y["train"]
            )
        else:
            X_cat_train_s, X_cont_train_s, y_train_s = (
                self.X_cat["train"],
                self.X_cont["train"],
                self.y["train"],
            )

        # data prep
        X_real, y_real = self.prep_data(X_cat_train_s, X_cont_train_s, y_train_s)
        X_real_catboost, y_real_catboost, cat_features = self.prep_data(
            X_cat_train_s, X_cont_train_s, y_train_s, catboost=True
        )

        X_gen_prepped, y_gen_prepped = self.prep_data(X_cat_gen, X_cont_gen, y_gen)
        X_gen_catboost, y_gen_catboost, _ = self.prep_data(
            X_cat_gen, X_cont_gen, y_gen, catboost=True
        )

        # estimate ml efficiency for n_runs
        results = {"real": [], "gen": []}
        for i in range(self.n_runs):
            model_seed = self.seed + i
            res_real = self._get_sklearn_results(X_real, y_real, model_seed)
            res_real["catboost"] = self._get_catboost_results(
                X_real_catboost, y_real_catboost, cat_features, model_seed
            )

            res_gen = self._get_sklearn_results(
                X_gen_prepped, y_gen_prepped, model_seed
            )
            res_gen["catboost"] = self._get_catboost_results(
                X_gen_catboost, y_gen_catboost, cat_features, model_seed
            )

            results["real"].append(res_real)
            results["gen"].append(res_gen)

        # get average performance across models and per model
        ml_eff_score = {}
        avg_results_per_model = {}
        models = results["real"][0].keys()
        metrics = results["real"][0][list(models)[0]].keys()
        n_metrics = len(metrics)
        for state in ["real", "gen"]:
            avg = np.zeros((self.n_runs, n_metrics))
            avg_per_model = {}
            for model in models:
                df = pd.DataFrame(res[model] for res in results[state])
                avg += np.array(df)
                avg_per_model[model] = df.mean(0).to_dict()
            avg /= len(models)
            ml_eff_score[state] = pd.DataFrame(avg, columns=metrics)
            avg_results_per_model[state] = avg_per_model

        return ml_eff_score, avg_results_per_model
