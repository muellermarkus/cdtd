import os
import pickle
import time

import numpy as np
from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Discriminator

from .experiment import Experiment
from .utils import get_total_trainable_params, set_seeds


class Experiment_CTGAN(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def train(self, **kwargs):
        save_model = kwargs.get("save_model", False)
        set_seeds(self.seed)

        train_loader = self.data_wrangler.get_train_loader(self.config.model.batch_size)
        X_cat_train = train_loader.X_cat
        X_cont_train = train_loader.X_cont
        X_train = np.column_stack((X_cat_train, X_cont_train))
        categorical_features = list(range(X_cat_train.shape[1]))

        # batch sizes must be multiple of 10 (otherwise PAC does not work)
        batch_size = min(
            self.config.model.batch_size, self.data_wrangler.data.get_train_obs()
        )
        remainder = batch_size % 10
        batch_size -= remainder

        # convert training steps to epochs
        steps_per_epoch = self.data_wrangler.data.get_train_obs() / batch_size
        epochs = round(self.config.model.train_steps / steps_per_epoch)
        print(f"Training for {epochs} epochs.")

        self.model = CTGAN(
            embedding_dim=self.config.model.emb_dim,
            generator_dim=self.config.model.generator_dim,
            discriminator_dim=self.config.model.discriminator_dim,
            generator_lr=self.config.model.generator_lr,
            discriminator_lr=self.config.model.discriminator_lr,
            batch_size=batch_size,
            epochs=epochs,
            cuda=self.config.model.cuda,
            verbose=True,
        )

        training_start_time = time.time()
        self.model.fit(X_train, categorical_features)
        training_duration = time.time() - training_start_time

        # compute total number of parameters
        data_dim = self.model._transformer.output_dimensions
        discriminator = Discriminator(
            data_dim + self.model._data_sampler.dim_cond_vec(),
            self.model._discriminator_dim,
            pac=self.model.pac,
        )
        discriminator_params = get_total_trainable_params(discriminator)
        generator_params = get_total_trainable_params(self.model._generator)
        print(f"Total parameters: {discriminator_params + generator_params}")

        if save_model:
            self.save_model()
            self.save_train_time(training_duration)

    def save_model(self):
        with open(os.path.join(self.ckpt_restore_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(os.path.join(self.ckpt_restore_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def sample_tabular_data(self, num_samples, seed):
        set_seeds(seed)
        gen_data = self.model.sample(num_samples)

        # bring generated data in required format
        X_cat_gen = gen_data[:, : self.data_wrangler.num_cat_features]
        X_cont_gen = gen_data[:, self.data_wrangler.num_cat_features :]
        y_gen = None
        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen, X_cont_gen, y_gen
        )

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen
