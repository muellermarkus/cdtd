import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from synthcity.plugins.core.models.tabular_ddpm import TabDDPM

from .experiment import Experiment
from .utils import set_seeds


class Experiment_TabDDPM(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def train(self, **kwargs):
        save_model = kwargs.get("save_model", False)
        plot_figures = kwargs.get("plot_figures", False)
        set_seeds(self.seed)

        train_loader = self.data_wrangler.get_train_loader(self.config.model.batch_size)
        X_cat_train = train_loader.X_cat
        X_cont_train = train_loader.X_cont
        y_train = train_loader.y
        df = pd.DataFrame(np.column_stack((X_cat_train, X_cont_train)))

        if self.data_wrangler.num_cat_features > 0:
            cat_cols = list(range(X_cat_train.shape[1]))
            cat_counts = list(self.data_wrangler.num_cats)
            model_kwargs = {"cat_cols": cat_cols, "cat_counts": cat_counts}
        else:
            model_kwargs = {}

        if self.data_wrangler.task != "regression" and self.data_wrangler.y_cond:
            self._labels, self._cond_dist = np.unique(y_train, return_counts=True)
            self._cond_dist = self._cond_dist / self._cond_dist.sum()
            self.target_name = "target"
            cond = pd.Series(y_train, index=df.index)
            self.expecting_conditional = True
        else:
            cond = None

        # convert training steps to epochs
        batch_size = min(
            self.config.model.batch_size, self.data_wrangler.data.get_train_obs()
        )
        steps_per_epoch = self.data_wrangler.data.get_train_obs() / batch_size
        epochs = round(self.config.model.train_steps / steps_per_epoch)
        print(f"Training for {epochs} epochs.")

        self.model = TabDDPM(
            n_iter=epochs,
            lr=self.config.model.lr,
            weight_decay=self.config.model.weight_decay,
            batch_size=batch_size,
            num_timesteps=self.config.model.num_timesteps,
            gaussian_loss_type=self.config.model.gaussian_loss_type,
            is_classification=(self.data_wrangler.task != "regression"),
            scheduler=self.config.model.scheduler,
            device=torch.device(self.device),
            callbacks=(),
            log_interval=100,
            model_type=self.config.model.model_type,
            model_params=self.config.model.model_params.copy(),
            dim_embed=self.config.model.dim_embed,
            valid_size=0,
            valid_metric=None,
            verbose=plot_figures,
        )

        training_start_time = time.time()
        self.model.fit(df, cond, **model_kwargs)
        training_duration = time.time() - training_start_time
        self.loss_history = self.model.loss_history
        self.validation_history = self.model.val_history

        if save_model:
            self.save_train_time(training_duration)
            self.save_model()

    def save_model(self):
        checkpoint = {
            "model": self.model,
            "data_wrangler": self.data_wrangler,
            "loss_history": self.loss_history,
        }
        with open(os.path.join(self.ckpt_restore_dir, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(os.path.join(self.ckpt_restore_dir, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)
            self.model = checkpoint["model"]
            self.data_wrangler = checkpoint["data_wrangler"]

    def sample_tabular_data(self, num_samples, seed):
        set_seeds(seed)

        cond = None
        if self.data_wrangler.task != "regression" and self.data_wrangler.y_cond:
            cond = np.random.choice(self._labels, size=num_samples, p=self._cond_dist)

        df = self.model.generate(num_samples, cond=cond)

        # get data into required shape
        X_cat_gen = df.to_numpy()[:, : self.data_wrangler.num_cat_features]
        X_cont_gen = df.to_numpy()[:, self.data_wrangler.num_cat_features :]
        y_gen = cond

        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen, X_cont_gen, y_gen
        )

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen
