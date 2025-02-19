import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from prettytable import PrettyTable
from sklearn import model_selection, preprocessing
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OrdinalEncoder

from experiments.utils import set_seeds

from .data_utils import FastTensorDataLoader, OriginalData
from .dequantizer import Dequantizer


def _add_y_to_X(y, X, split):
    return np.column_stack((y[split], X[split]))


def _split_y_from_X(X):
    y = X[:, 0]
    X_new = X[:, 1:]
    return y, X_new


class DataWrangler(object):
    """
    Data preparation for pre-processing data and post-processing generated data.

    """

    def __init__(self, dataset, logdir, config, val_prop, test_prop, seed):
        self.data_name = dataset
        self.data_config = OmegaConf.load(f"configs/data/{self.data_name}.yaml")
        self.task = self.data_config.task
        self.target = self.data_config.target
        self.target_is_int = False
        if self.task == "regression":
            self.target_is_int = self.data_config.target_is_int
        self.cat_features = list(self.data_config.cat_features)
        self.cont_features = (
            list(self.data_config.cont_features)
            + list(self.data_config.int_features)
            + list(self.data_config.dequant_features)
        )
        self.int_features = list(
            self.data_config.int_features
        )  # round these features after generation
        self.dequant_features = list(
            self.data_config.dequant_features
        )  # apply dequantization to these features
        self.dequant_data = config.data.dequant_data
        if not self.dequant_data:
            # instead of dequantization, simply use rounding
            self.int_features = self.int_features + self.dequant_features
            self.dequant_features = []

        self.val_prop = val_prop
        self.test_prop = test_prop
        self.cont_scaler = config.data.cont_scaler
        self.standardize_data = config.data.standardize_data
        self.cat_encoding = config.data.cat_encoding
        self.y_cond = False
        if "y_cond" in config.model.keys():
            self.y_cond = config.model.y_cond
        self.drop_cont_missings = True

        if logdir is not None:
            self.logdir = os.path.join(logdir, "data")
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

        # derive indices for rounding and encoding of cont features
        self.cont_idx = {"int": [], "dequant": [], "non_dequant": []}
        for i, label in enumerate(self.cont_features):
            if label in self.int_features:
                self.cont_idx["int"].append(i)
                self.cont_idx["non_dequant"].append(i)
            elif label in self.dequant_features:
                self.cont_idx["dequant"].append(i)
            else:
                self.cont_idx["non_dequant"].append(i)
        self.cont_idx = {k: np.array(self.cont_idx[k]) for k in self.cont_idx.keys()}

        self.data = self.preprocess_data(seed)

    def preprocess_data(self, seed):
        set_seeds(seed)

        if self.data_name == "bank":
            sep = ";"
        else:
            sep = ","

        df = pd.read_csv(
            os.path.join("data", "raw_data", self.data_name, self.data_config.csv_file),
            sep=sep,
        )

        # code missings
        df = df[[self.target] + self.cat_features + self.cont_features]
        df = df.replace(r" ", np.nan)
        df = df.replace(r"?", np.nan)
        df = df.replace(r"", np.nan)

        if self.data_name == "diabetes":
            df[self.target] = np.where(df[self.target] == "NO", 0, 1)
            enc = OrdinalEncoder()
            df["age"] = enc.fit_transform(df["age"].to_numpy().reshape(-1, 1))
        elif self.data_name == "diabetes_mult_class":
            enc = OrdinalEncoder()
            df["age"] = enc.fit_transform(df["age"].to_numpy().reshape(-1, 1))
        elif self.data_name == "lending":
            enc = OrdinalEncoder()
            df["sub_grade"] = enc.fit_transform(
                df["sub_grade"].to_numpy().reshape(-1, 1)
            )
            enc = OrdinalEncoder()
            df["grade"] = enc.fit_transform(df["grade"].to_numpy().reshape(-1, 1))
        elif self.data_name == "nmes":
            cats = [["poor", "average", "excellent"]]
            enc = OrdinalEncoder(categories=cats)
            df["health"] = enc.fit_transform(df["health"].to_numpy().reshape(-1, 1))
        elif self.data_name == "covertype":
            df[self.target] = np.where(df[self.target] == 2, 1, 0)

        # remove rows with missings in targets
        idx_target_nan = df[self.target].isna().to_numpy().nonzero()[0]
        df.drop(labels=idx_target_nan, axis=0, inplace=True)

        # for categorical features, replace missings with 'empty
        df[self.cat_features] = df[self.cat_features].fillna("empty")

        # for continuous data, drop missing or set to some value and use masking during training
        if self.drop_cont_missings:
            df.dropna(inplace=True)
        else:
            df[self.cont_features] = df[self.cont_features].fillna(-9999999)

        # ensure correct types
        X_cat = df[self.cat_features].to_numpy().astype("str")
        X_cont = df[self.cont_features].to_numpy().astype("float")
        y = df[self.target].to_numpy()

        # compute proportion of data required to achieve val_prop
        prop = self.val_prop / (1 - self.test_prop)

        # train, validation, test split
        stratify = None if self.task == "regression" else y
        X_cat_train, X_cat_test, X_cont_train, X_cont_test, y_train, y_test = (
            model_selection.train_test_split(
                X_cat,
                X_cont,
                y,
                test_size=self.test_prop,
                stratify=stratify,
                random_state=42,
            )
        )
        if self.val_prop > 0:
            stratify = None if self.task == "regression" else y_train
            X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = (
                model_selection.train_test_split(
                    X_cat_train,
                    X_cont_train,
                    y_train,
                    stratify=stratify,
                    test_size=prop,
                    random_state=42,
                )
            )
        else:
            X_cat_val, X_cont_val, y_val = None, None, None

        X_cat = {"train": X_cat_train, "val": X_cat_val, "test": X_cat_test}
        X_cont = {"train": X_cont_train, "val": X_cont_val, "test": X_cont_test}
        y = {"train": y_train, "val": y_val, "test": y_test}

        # combine y and X based on y_cond and task
        if self.y_cond:
            # condition model on y
            assert self.task != "regression"
            self.y_int_enc = preprocessing.LabelEncoder()
            self.y_int_enc.fit(
                np.concatenate(list(y[k] for k in y.keys() if y[k] is not None))
            )
            y = {
                k: self.y_int_enc.transform(v) if v is not None else None
                for k, v in y.items()
            }
            self.num_y_classes = len(self.y_int_enc.classes_)
        else:
            if self.task == "regression":
                # include y into X_cont such that y = X_cont[:,0]
                X_cont = {
                    split: np.column_stack((y[split], X_cont[split]))
                    if X_cont[split] is not None
                    else None
                    for split in X_cont.keys()
                }
                # adjust indices used for encoding later on
                self.cont_idx["int"] += 1
                self.cont_idx["dequant"] += 1
                self.cont_idx["non_dequant"] += 1
                self.cont_features = [self.target] + self.cont_features
                if self.target_is_int:
                    self.cont_idx["int"] = np.insert(self.cont_idx["int"], 0, 0).astype(
                        int
                    )
                    self.cont_idx["non_dequant"] = np.insert(
                        self.cont_idx["non_dequant"], 0, 0
                    ).astype(int)
            else:
                # include y into X_cat such that y = X_cat[:,0]
                X_cat = {
                    split: np.column_stack((y[split], X_cat[split]))
                    if X_cat[split] is not None
                    else None
                    for split in X_cat.keys()
                }
                self.cat_features = [self.target] + self.cat_features
            y = {"train": None, "val": None, "test": None}
            self.num_y_classes = None

        # store number of features to construct model later on
        self.num_cat_features = (
            X_cat["train"].shape[1] if X_cat["train"] is not None else 0
        )
        self.num_cont_features = (
            X_cont["train"].shape[1] if X_cont["train"] is not None else 0
        )
        self.num_total_features = self.num_cat_features + self.num_cont_features

        # preprocess categorical classes and convert to integers
        if self.num_cat_features > 0:
            self.cat_int_enc = preprocessing.OrdinalEncoder()
            self.cat_int_enc.fit(
                np.concatenate(
                    list(X_cat[k] for k in X_cat.keys() if X_cat[k] is not None)
                )
            )
            X_cat = {
                k: self.cat_int_enc.transform(v) if v is not None else None
                for k, v in X_cat.items()
            }

            # store number of categories per feature to construct model
            # when using ordinal cat encoding only (not onehot)
            self.num_cats = None
            if self.cat_encoding is None:
                num_cats = []
                for i in range(self.num_cat_features):
                    uniq_vals = np.unique(X_cat["train"][:, i])
                    num_cats.append(len(uniq_vals))
                self.num_cats = num_cats
        else:
            self.num_cats = None
            X_cat = {k: None for k in y.keys()}

        if self.num_cont_features == 0:
            X_cont = {k: None for k in y.keys()}

        return OriginalData(X_cat, X_cont, y)

    def get_train_loader(self, batch_size, partition="train"):
        """Transform data and construct train/valid/test loader."""

        batch_size = min(batch_size, self.data.get_train_obs())
        X_cat, X_cont, y = self.data.get_data()

        # transform categorical features again (ordinal encoding, to ensure that order goes from 0 to num categories - 1 in training set)
        if self.num_cat_features > 0:
            if self.cat_encoding is None:
                self.cat_enc = preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=9999
                )
                self.cat_enc.fit(X_cat["train"])
            elif self.cat_encoding == "onehot":
                self.cat_enc = preprocessing.OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float32
                )
                self.cat_enc.fit(X_cat["train"])
            else:
                ValueError(self.cat_encoding)
            X_cat_transformed = {
                k: self.cat_enc.transform(v) if v is not None else None
                for k, v in X_cat.items()
            }

        # transform continuous features that are not dequantized later (so truly continuous or integer-valued but we do not observe all possible values)
        if (len(self.cont_idx["non_dequant"]) > 0) and self.cont_scaler is not None:
            if self.cont_scaler == "quantile":
                self.cont_enc = preprocessing.QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=max(min(self.data.get_train_obs() // 30, 1000), 10),
                    subsample=int(1e9),
                    random_state=42,
                )
            elif self.cont_scaler == "standard":
                self.cont_enc = preprocessing.StandardScaler()
            else:
                ValueError(self.cont_scaler)
            self.cont_enc.fit(X_cont["train"][:, self.cont_idx["non_dequant"]])

        # transform discrete continuous features for which we observe all possible values using dequantization
        if len(self.cont_idx["dequant"]) > 0:
            self.dequantizer = Dequantizer()
            self.dequantizer.fit(X_cont["train"][:, self.cont_idx["dequant"]])

        # apply transformations
        X_cont_transformed = {}
        for set in ["train", "val", "test"]:
            X = np.copy(X_cont[set])
            if len(self.cont_idx["dequant"]) > 0:
                X[:, self.cont_idx["dequant"]] = self.dequantizer.transform(
                    X_cont[set][:, self.cont_idx["dequant"]]
                )
            if (len(self.cont_idx["non_dequant"]) > 0) and self.cont_scaler is not None:
                X[:, self.cont_idx["non_dequant"]] = self.cont_enc.transform(
                    X_cont[set][:, self.cont_idx["non_dequant"]]
                )

                # additionally standardize data (zero mean, unit variance), as sometimes this is not achieved by QuantileTransformer when we have skewed data
                if self.cont_scaler == "quantile":
                    if set == "train":
                        self.X_cont_mean = X[:, self.cont_idx["non_dequant"]].mean(0)
                        self.X_cont_std = X[:, self.cont_idx["non_dequant"]].std(0)
                    X[:, self.cont_idx["non_dequant"]] = (
                        X[:, self.cont_idx["non_dequant"]] - self.X_cont_mean
                    ) / self.X_cont_std
            X_cont_transformed[set] = X

        # construct data loader
        def get_loader(partition):
            X_cat_train = (
                torch.tensor(X_cat_transformed[partition]).long()
                if self.num_cat_features > 0
                else None
            )
            X_cont_train = (
                torch.tensor(X_cont_transformed[partition]).float()
                if self.num_cont_features > 0
                else None
            )
            if self.y_cond:
                y_train = (
                    torch.tensor(y[partition]).long()
                    if self.task != "regression"
                    else torch.tensor(y[partition]).float()
                )
            else:
                y_train = None

            if partition == "train":
                shuffle = True
                drop_last = True
            else:
                shuffle = False
                drop_last = False

            return FastTensorDataLoader(
                X_cat_train,
                X_cont_train,
                y_train,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )

        return get_loader(partition)

    def postprocess_gen_data(self, X_cat_gen, X_cont_gen, y_gen):
        """Apply inverse transformationto generated data points."""

        X_cat_gen = self.cat_enc.inverse_transform(X_cat_gen)

        if (len(self.cont_idx["non_dequant"]) > 0) and self.cont_scaler is not None:
            if self.cont_scaler == "quantile":
                # revert standardization
                X_cont_gen[:, self.cont_idx["non_dequant"]] = (
                    X_cont_gen[:, self.cont_idx["non_dequant"]] * self.X_cont_std
                ) + self.X_cont_mean

            # revert quantile transformation or standardization
            X_cont_gen[:, self.cont_idx["non_dequant"]] = (
                self.cont_enc.inverse_transform(
                    X_cont_gen[:, self.cont_idx["non_dequant"]]
                )
            )

        if len(self.cont_idx["dequant"]) > 0:
            # revert dequantization
            X_cont_gen[:, self.cont_idx["dequant"]] = (
                self.dequantizer.inverse_transform(
                    X_cont_gen[:, self.cont_idx["dequant"]]
                )
            )

        # round integer columns in generated data
        X_cont_gen[:, self.cont_idx["int"]] = np.round(
            X_cont_gen[:, self.cont_idx["int"]]
        ).astype(int)

        return X_cat_gen, X_cont_gen, y_gen

    def get_DCR(self, X_cat_gen, X_cont_gen, y_gen, prefix=None):
        X_cat_real_train, X_cont_real_train, y_real_train = self.data.get_train_data()
        X_cat_real_test, X_cont_real_test, y_real_test = self.data.get_test_data()

        # include y depending on conditioning
        if self.y_cond:
            X_cat_real_train = np.column_stack((y_real_train, X_cat_real_train))
            X_cat_real_test = np.column_stack((y_real_test, X_cat_real_test))
            X_cat_gen = np.column_stack((y_gen, X_cat_gen))

        onehot_enc = preprocessing.OneHotEncoder(sparse_output=False)
        onehot_enc.fit(np.row_stack((X_cat_real_train, X_cat_real_test)))
        X_cat_real_train_ohe = onehot_enc.transform(X_cat_real_train)
        X_cat_real_test_ohe = onehot_enc.transform(X_cat_real_test)
        X_cat_gen_ohe = onehot_enc.transform(X_cat_gen)

        # subsample real and fake data if dataset is very large
        if len(X_cat_real_train_ohe) > 50000:
            idx = np.random.choice(len(X_cat_real_train_ohe), 50000, replace=False)
            X_cat_real_train_ohe = X_cat_real_train_ohe[idx]
            X_cont_real_train = X_cont_real_train[idx]

        if len(X_cat_real_test_ohe) > 50000:
            idx = np.random.choice(len(X_cat_real_test_ohe), 50000, replace=False)
            X_cat_real_test_ohe = X_cat_real_test_ohe[idx]
            X_cont_real_test = X_cont_real_test[idx]

        if len(X_cat_gen_ohe) > 50000:
            idx = np.random.choice(len(X_cat_gen_ohe), 50000, replace=False)
            X_cat_gen_ohe = X_cat_gen_ohe[idx]
            X_cont_gen = X_cont_gen[idx]

        assert (
            X_cat_real_train_ohe.shape[1]
            == X_cat_real_test_ohe.shape[1]
            == X_cat_gen_ohe.shape[1]
        )

        # combine all data
        X_real_train = np.column_stack((X_cat_real_train_ohe, X_cont_real_train))
        X_real_test = np.column_stack((X_cat_real_test_ohe, X_cont_real_test))
        X_gen = np.column_stack((X_cat_gen_ohe, X_cont_gen))

        standardizer_real_train = preprocessing.StandardScaler()
        standardizer_real_test = preprocessing.StandardScaler()
        standardizer_gen = preprocessing.StandardScaler()
        standardizer_real_train.fit(X_real_train)
        standardizer_real_test.fit(X_real_test)
        standardizer_gen.fit(X_gen)
        X_real_train_scaled = standardizer_real_train.transform(X_real_train)
        X_real_test_scaled = standardizer_real_train.transform(X_real_test)
        X_gen_scaled = standardizer_gen.transform(X_gen)

        indices_gen, min_distances_gen = pairwise_distances_argmin_min(
            X_gen_scaled, X_real_train_scaled, metric="euclidean"
        )
        indices_test, min_distances_test = pairwise_distances_argmin_min(
            X_real_test_scaled, X_real_train_scaled, metric="euclidean"
        )
        plt.close("all")
        plot_dcr(
            self.logdir,
            min_distances_gen,
            f"gen_{prefix}" if prefix is not None else "gen",
        )
        plot_dcr(
            self.logdir,
            min_distances_test,
            f"real_{prefix}" if prefix is not None else "real",
        )

        output_dict = {
            "fifth_perc_gen": np.percentile(min_distances_gen, 5),
            "fifth_perc_test": np.percentile(min_distances_test, 5),
            "median_gen": np.median(min_distances_gen),
            "median_test": np.median(min_distances_test),
            "mean_gen": np.mean(min_distances_gen),
            "mean_test": np.mean(min_distances_test),
            "min_gen": np.min(min_distances_gen),
            "min_test": np.min(min_distances_test),
            "max_gen": np.max(min_distances_gen),
            "max_test": np.max(min_distances_test),
        }

        return output_dict

    def check_bounds(self, X_cont_gen):
        _, X_cont_train, _ = self.data.get_train_data()
        lower_bounds = X_cont_train.min(axis=0)
        upper_bounds = X_cont_train.max(axis=0)

        output_tbl = PrettyTable()
        labels = [f"feature {i}" for i in range(X_cont_train.shape[1])]
        output_tbl.field_names = ["constraint", *labels]
        output_tbl.add_row(["minimum", *lower_bounds])
        output_tbl.add_row(["< minimum", *np.sum(X_cont_gen < lower_bounds, axis=0)])
        output_tbl.add_row(["maximum", *upper_bounds])
        output_tbl.add_row(["> maximum", *np.sum(X_cont_gen > upper_bounds, axis=0)])

        with open(os.path.join(self.logdir, "bound_checks.txt"), "w") as f:
            f.write(str(output_tbl))

    def save_data(self, X_cat_gen, X_cont_gen, y_gen):
        labels = []
        train_data = []
        gen_data = []

        X_cat, X_cont, y = self.data.get_train_data()

        if y is not None:
            train_data.append(y)
            gen_data.append(y_gen)
            labels = labels + [self.target]
        if X_cat is not None:
            train_data.append(X_cat)
            gen_data.append(X_cat_gen)
            labels = labels + self.cat_features
        if X_cont is not None:
            train_data.append(X_cont)
            gen_data.append(X_cont_gen)
            labels = labels + self.cont_features

        df_train = pd.DataFrame(np.column_stack(train_data), columns=labels)
        df_gen = pd.DataFrame(np.column_stack(gen_data), columns=labels)
        df_train.to_csv(os.path.join(self.logdir, "train_data.csv"), index=False)
        df_gen.to_csv(os.path.join(self.logdir, "gen_data.csv"), index=False)

    def compute_diff_in_means(self, X_cat_gen, X_cont_gen, y_gen):
        X_cat_train, X_cont_train, _ = self.data.get_train_data()

        # standardize so they have same scale
        enc_cont = preprocessing.StandardScaler()
        enc_cont.fit(X_cont_train)
        X_cont_train = enc_cont.transform(X_cont_train)
        X_cont_gen = enc_cont.transform(X_cont_gen)

        # can disregard y: either it does not exist, or we sample it outside of the diffusion model, so means match by construction
        X_gen = np.column_stack((X_cat_gen, X_cont_gen))
        X_train = np.column_stack((X_cat_train, X_cont_train))

        return torch.tensor(np.mean(X_train, 0) - np.mean(X_gen, 0))

    def get_diff_in_means_table(self, X_cat_gen, X_cont_gen, y_gen):
        diff_in_means = self.compute_diff_in_means(X_cat_gen, X_cont_gen, y_gen)

        cat_labels = [f"feature {i} (cat)]" for i in range(X_cat_gen.shape[1])]
        cont_labels = [f"feature {i} (cont)]" for i in range(X_cont_gen.shape[1])]
        labels = cat_labels + cont_labels

        table = PrettyTable()
        table.field_names = ["feature", "mean difference"]
        for i, val in enumerate(diff_in_means):
            table.add_row([labels[i], round(val.item(), 3)])

        with open(os.path.join(self.logdir, "diff_in_means.txt"), "w") as f:
            f.write(str(table))


def get_encoders(cat_encoding, cont_scaler, X_cat_train, X_cont_train):
    # categorical features
    if cat_encoding == "onehot":
        cat_enc = preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        cat_enc.fit(X_cat_train)
    else:
        cat_enc = None

    #  continuous features
    cont_enc = None
    if cont_scaler == "quantile":
        cont_enc = preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X_cont_train.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=42,
        )
    elif cont_scaler == "standard":
        cont_enc = preprocessing.StandardScaler()
    elif cont_scaler == "minmax":
        cont_enc = preprocessing.MinMaxScaler()

    if cont_enc is not None:
        cont_enc.fit(X_cont_train)

    return cat_enc, cont_enc


def plot_dcr(logdir, dcr, prefix):
    ax = sns.histplot(dcr)
    plt.text(
        0.01,
        0.99,
        f"min(DCR) = {torch.tensor(dcr).min().item()} ({(dcr == dcr.min()).sum()} cases)",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    plt.ylabel("Number of Samples")
    plt.xlabel("DCR")
    plt.savefig(os.path.join(logdir, prefix + "_dcr.png"), dpi=300, bbox_inches="tight")
    plt.close()
