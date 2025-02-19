import copy

import numpy as np
from dython.nominal import associations
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, wasserstein_distance
from sklearn import preprocessing


class SimilarityScores:
    def __init__(self, X_cat, X_cont, y, task):
        """
        X_cat, X_cont, y are dictionaries containing the train, valid, test splits.
        """

        self.task = task
        self.y_cond = y["train"] is not None

        if self.y_cond:
            if self.task != "regression":
                xs_cat_train = [y["train"], X_cat["train"]]
                xs_cat_train = [x for x in xs_cat_train if x is not None]
                X_cat_train = np.column_stack(xs_cat_train)
                self.X_cont_train = X_cont["train"]
            else:
                xs_cont_train = [y["train"], X_cont["train"]]
                xs_cont_train = [x for x in xs_cont_train if x is not None]
                self.X_cont_train = np.column_stack(xs_cont_train)
                X_cat_train = X_cat["train"]
        else:
            self.X_cont_train = X_cont["train"]
            X_cat_train = X_cat["train"]

        # init pdf of training data for Jensen Shannon divergence
        self.train_pdfs = []
        self.train_labels = []
        self.num_cat_features = X_cat_train.shape[1]
        self.nominal_columns = list(range(self.num_cat_features))
        for col_idx in range(X_cat_train.shape[1]):
            # construct probabilities from categorical values
            real_labels, real_bin_count = np.unique(
                X_cat_train[:, col_idx].astype(int), return_counts=True
            )
            real_pdf = real_bin_count / real_bin_count.sum()
            self.train_labels.append(real_labels)
            self.train_pdfs.append(np.array(real_pdf))

        # prep things for Wasserstein divergence
        self.cont_scaler = preprocessing.MinMaxScaler()
        self.cont_scaler.fit(self.X_cont_train)

        self.sim_test = self.compute_similarity(
            X_cat["test"], X_cont["test"], y["test"]
        )

        # init things for corrleation computation
        if self.y_cond:
            if self.task == "regression":
                xs_train = [X_cat["train"], y["train"], X_cont["train"]]
            elif self.task != "regression":
                xs_train = [y["train"], X_cat["train"], X_cont["train"]]
        else:
            xs_train = [X_cat["train"], X_cont["train"]]

        # filter out if X_cat, X_cont or y are None
        xs_train = [x for x in xs_train if x is not None]
        X_train = np.column_stack(xs_train)

        self.corr_train = self._compute_correlation(X_train)
        self.corr_test_diffs = self.compute_diff_in_corr(
            X_cat["test"], X_cont["test"], y["test"]
        )

    def _compute_correlation(self, X):
        return associations(
            X,
            nominal_columns=self.nominal_columns,
            mark_columns=True,
            nom_nom_assoc="theil",
            num_num_assoc="pearson",
            plot=False,
        )["corr"]

    def compute_diff_in_corr(self, X_cat_gen, X_cont_gen, y_gen):
        if self.y_cond:
            if self.task == "regression":
                xs = [X_cat_gen, y_gen, X_cont_gen]
            elif self.task != "regression":
                xs = [y_gen, X_cat_gen, X_cont_gen]
        else:
            xs = [X_cat_gen, X_cont_gen]

        xs = [x for x in xs if x is not None]
        X_gen = np.column_stack(xs)

        res_gen = self._compute_correlation(X_gen)

        # construct differences in correlations
        diffs = np.array(res_gen - self.corr_train)
        abs_diff_corr = np.abs(diffs)
        l2_norm_diff_corr = np.linalg.norm(diffs)

        return {"abs_diff": abs_diff_corr, "l2_norm": l2_norm_diff_corr}

    def compute_similarity(self, X_cat_gen, X_cont_gen, y_gen):
        if self.y_cond:
            if self.task != "regression":
                xs = [y_gen, X_cat_gen]
                xs = [x for x in xs if x is not None]
                X_cat = np.column_stack(xs)
                X_cont = copy.deepcopy(X_cont_gen)
            else:
                xs = [y_gen, X_cont_gen]
                xs = [x for x in xs if x is not None]
                X_cont = np.column_stack(xs)
                X_cat = copy.deepcopy(X_cat_gen)
        else:
            X_cont = copy.deepcopy(X_cont_gen)
            X_cat = copy.deepcopy(X_cat_gen)

        cat_stat = []
        cont_stat = []
        ks_test = []

        # compute Jensen Shannon divergence for categorical features
        for col_idx in range(self.num_cat_features):
            # construct probabilities from categorical values
            gen_labels, gen_bin_count = np.unique(
                X_cat[:, col_idx].astype(int), return_counts=True
            )
            gen_pdf = gen_bin_count / gen_bin_count.sum()

            # adjust pdf if a class does not appear in pdf of generated data
            adjusted_gen_pdf = []
            if len(self.train_labels[col_idx]) != len(gen_labels):
                for label in self.train_labels[col_idx]:
                    filter = label == gen_labels
                    if any(filter):
                        adjusted_gen_pdf.append(gen_pdf[filter].item())
                    else:
                        adjusted_gen_pdf.append(0.0)
            else:
                adjusted_gen_pdf = gen_pdf

            cat_stat.append(
                jensenshannon(
                    self.train_pdfs[col_idx], np.array(adjusted_gen_pdf), base=2.0
                )
            )

        # compute Wasserstein distance for continuous features, scaled to [0,1]
        X_cont_train_scaled = self.cont_scaler.transform(self.X_cont_train)
        X_cont_scaled = self.cont_scaler.transform(X_cont)

        for col_idx in range(X_cont.shape[1]):
            cont_stat.append(
                wasserstein_distance(
                    X_cont_train_scaled[:, col_idx], X_cont_scaled[:, col_idx]
                )
            )

            # test H0: the samples are drawn from the same distribution
            ks_test.append(kstest(self.X_cont_train[:, col_idx], X_cont[:, col_idx]))

        return {
            "cat_full": cat_stat,
            "cont_full": cont_stat,
            "cat_max": np.max(cat_stat),
            "cont_max": np.max(cont_stat),
            "cat_min": np.min(cat_stat),
            "cont_min": np.min(cont_stat),
            "cat": np.mean(cat_stat),
            "cont": np.mean(cont_stat),
            "ks_test": ks_test,
        }
