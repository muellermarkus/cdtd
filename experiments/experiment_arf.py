import os
import pickle
import time

import pandas as pd
from arfpy import arf

from .experiment import Experiment
from .utils import set_seeds


class Experiment_ARF(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def train(self, **kwargs):
        save_model = kwargs.get("save_model", False)
        set_seeds(self.seed)

        train_loader = self.data_wrangler.get_train_loader(4096)
        df_cat = pd.DataFrame(train_loader.X_cat).astype("category")
        df_cont = pd.DataFrame(train_loader.X_cont)
        df = pd.concat((df_cat, df_cont), axis=1, ignore_index=True)

        training_start_time = time.time()
        self.model = arf.arf(
            df,
            num_trees=self.config.model.num_trees,
            delta=self.config.model.delta,
            max_iters=self.config.model.max_iters,
            min_node_size=self.config.model.min_node_size,
            random_state=self.seed,
            n_jobs=self.config.model.n_jobs,
        )
        self.model.forde()
        training_duration = time.time() - training_start_time

        if save_model:
            self.save_train_time(training_duration)
            self.save_model()

    def save_model(self):
        checkpoint = {"model": self.model, "data_wrangler": self.data_wrangler}
        with open(os.path.join(self.ckpt_restore_dir, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(os.path.join(self.ckpt_restore_dir, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)
            self.model = checkpoint["model"]
            self.data_wrangler = checkpoint["data_wrangler"]

    def sample_tabular_data(self, num_samples, seed):
        set_seeds(seed)
        df = self.model.forge(n=num_samples)

        # get data into required shape
        X_cat_gen = df.to_numpy()[:, : self.data_wrangler.num_cat_features]
        X_cont_gen = df.to_numpy()[:, self.data_wrangler.num_cat_features :]
        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen, X_cont_gen, None
        )

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen
