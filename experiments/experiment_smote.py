import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.neighbors import NearestNeighbors

from data.data_prep import _split_y_from_X

from .experiment import Experiment
from .utils import set_seeds


class Experiment_SMOTE(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def train(self, **kwargs):
        self.save_train_time(0.0)
        return

    def sample_tabular_data(self, num_samples, **kwargs):
        seed = kwargs.get("seed", None)
        set_seeds(seed)

        train_loader = self.data_wrangler.get_train_loader(1000)
        X_cat_train = train_loader.X_cat
        X_cont_train = train_loader.X_cont

        if self.data_wrangler.task == "regression":
            # pseudo outcome similar to TabDDPM paper
            y_train = np.where(X_cont_train[:, 0] > np.median(X_cont_train[:, 0]), 1, 0)
        else:
            y_train, X_cat_train = _split_y_from_X(X_cat_train)
            y_train = y_train.numpy()

        X_train = np.column_stack((X_cat_train, X_cont_train))

        # setup SMOTE
        frac_samples = num_samples / X_train.shape[0]
        sampling_strategy = {
            k: int((1 + frac_samples) * np.sum(y_train == k))
            for k in np.unique(y_train)
        }
        obs_sum = sum(sampling_strategy.values())
        diff = obs_sum - y_train.shape[0]
        # if too many / too few samples would be drawn, make adjustments to randomly chosen class
        if diff != num_samples:
            c = np.random.choice(list(sampling_strategy.keys()), 1).item()
            sampling_strategy[c] += num_samples - diff
            assert sum(sampling_strategy.values()) - y_train.shape[0] == num_samples

        categorical_features = list(range(X_cat_train.shape[1]))

        nearest_neighbors = NearestNeighbors(
            n_neighbors=self.config.model.k_neighbors, n_jobs=self.config.model.n_jobs
        )
        smote = SMOTENC(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=seed,
            k_neighbors=nearest_neighbors,
        )

        X_gen, y_gen = smote.fit_resample(X_train, y_train)

        # only retain fake data not true samples
        X_gen = X_gen[X_train.shape[0] :]
        y_gen = y_gen[y_train.shape[0] :]
        
        # shuffle generated data
        idx = np.random.permutation(len(y_gen))
        X_gen = X_gen[idx]
        y_gen = y_gen[idx]

        # split into X_cat, X_cont, y, depending on task and conditioning
        if self.data_wrangler.task == "regression":
            y_gen = None
        else:
            if not self.data_wrangler.y_cond:
                X_gen = np.column_stack((y_gen, X_gen))
                y_gen = None

        X_cat_gen = np.int64(X_gen[:, : self.data_wrangler.num_cat_features])
        X_cont_gen = X_gen[:, self.data_wrangler.num_cat_features :]

        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen, X_cont_gen, y_gen
        )
        X_cat_gen = X_cat_gen.astype(int)

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen

    def save_model(self):
        return

    def load_model(self):
        return
