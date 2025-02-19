import os
import pickle
import time

import numpy as np
from ctgan import TVAE
from ctgan.synthesizers.tvae import Encoder

from .experiment import Experiment
from .utils import get_total_trainable_params, set_seeds


class Experiment_TVAE(Experiment):
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

        # convert training steps to epochs
        batch_size = min(
            self.data_wrangler.data.get_train_obs(), self.config.model.batch_size
        )
        steps_per_epoch = self.data_wrangler.data.get_train_obs() / batch_size
        epochs = round(self.config.model.train_steps / steps_per_epoch)
        print(f"Training for {epochs} epochs.")

        self.model = TVAE(
            embedding_dim=self.config.model.emb_dim,
            compress_dims=self.config.model.compress_dims,
            decompress_dims=self.config.model.decompress_dims,
            batch_size=batch_size,
            epochs=epochs,
            cuda=self.config.model.cuda,
            verbose=True,
        )

        training_start_time = time.time()
        self.model.fit(X_train, categorical_features)
        training_duration = time.time() - training_start_time

        # calculate total number of parameters
        data_dim = self.model.transformer.output_dimensions
        encoder = Encoder(
            data_dim, self.config.model.compress_dims, self.model.embedding_dim
        )
        encoder_params = get_total_trainable_params(encoder)
        decoder_params = get_total_trainable_params(self.model.decoder)
        print(f"Total parameters: {encoder_params + decoder_params}")

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
