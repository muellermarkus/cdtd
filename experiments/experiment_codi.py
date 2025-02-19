import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from experiments.codi.diffusion_continuous import (
    GaussianDiffusionSampler,
    GaussianDiffusionTrainer,
)
from experiments.codi.diffusion_discrete import MultinomialDiffusion
from experiments.codi.models.tabular_unet import tabularUnet
from experiments.codi.tabular_transformer import GeneralTransformer
from experiments.codi.utils import (
    apply_activate,
    infiniteloop,
    log_sample_categorical,
    make_negative_condition,
    sampling_with,
    training_with,
)

from .experiment import Experiment
from .utils import set_seeds


def warmup_lr(step):
    return min(step, 5000) / 5000


class Experiment_CoDi(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

        self.config = config
        # method-specific data prep
        X_cat_train, X_cont_train, y_train = self.data_wrangler.data.get_train_data()
        train = np.column_stack((X_cat_train, X_cont_train))

        self.config.batch_size = min(
            self.data_wrangler.data.get_train_obs(), self.config.batch_size
        )
        cols_idx = list(np.arange(train.shape[1]))
        self.dis_idx = list(range(self.data_wrangler.num_cat_features))
        self.con_idx = [x for x in cols_idx if x not in self.dis_idx]

        # split continuous and categorical
        train_con = train[:, self.con_idx]
        train_dis = train[:, self.dis_idx]

        self.transformer_con = GeneralTransformer()
        self.transformer_dis = GeneralTransformer()

        self.transformer_con.fit(train_con, [])
        self.transformer_dis.fit(train_dis, self.dis_idx)

        train_con_data = self.transformer_con.transform(train_con)
        train_dis_data = self.transformer_dis.transform(train_dis)
        self.num_class = np.array(self.data_wrangler.num_cats)
        self.train_con_data_shape = train_con_data.shape
        self.train_dis_data_shape = train_dis_data.shape

        self.train_iter_con = DataLoader(
            train_con_data, batch_size=self.config.batch_size
        )
        self.train_iter_dis = DataLoader(
            train_dis_data, batch_size=self.config.batch_size
        )

        # Continuous diffusion model setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.input_size = train_con_data.shape[1]
        config.cond_size = train_dis_data.shape[1]
        config.output_size = train_con_data.shape[1]
        config.encoder_dim = config.encoder_dim_con
        config.nf = config.nf_con
        self.model_con = tabularUnet(config)
        self.trainer = GaussianDiffusionTrainer(
            self.model_con, config.beta_1, config.beta_T, config.T
        ).to(self.device)

        # Multinomial diffusion model setup
        config.input_size = train_dis_data.shape[1]
        config.cond_size = train_con_data.shape[1]
        config.output_size = train_dis_data.shape[1]
        config.encoder_dim = config.encoder_dim_dis
        config.nf = config.nf_dis
        self.model_dis = tabularUnet(config)
        self.trainer_dis = MultinomialDiffusion(
            self.num_class,
            train_dis_data.shape,
            self.model_dis,
            config,
            timesteps=config.T,
            loss_type="vb_stochastic",
        ).to(self.device)

        num_params_con = sum(p.numel() for p in self.model_con.parameters())
        num_params_dis = sum(p.numel() for p in self.model_dis.parameters())
        print(f"Total parameters: {num_params_con + num_params_dis}")
        logging.info("Continuous model params: %d" % (num_params_con))
        logging.info("Discrete model params: %d" % (num_params_dis))

    def train(self, **kwargs):
        save_model = kwargs.get("save_model", False)
        set_seeds(self.seed)

        datalooper_train_con = infiniteloop(self.train_iter_con)
        datalooper_train_dis = infiniteloop(self.train_iter_dis)

        optim_con = torch.optim.Adam(self.model_con.parameters(), lr=self.config.lr_con)
        sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
        optim_dis = torch.optim.Adam(self.model_dis.parameters(), lr=self.config.lr_dis)
        sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)

        total_steps_both = self.config.total_steps_both

        # Start Training
        training_start_time = time.time()
        epoch = 0
        with tqdm(initial=0, total=total_steps_both, disable=(not save_model)) as pbar:
            for step in range(total_steps_both):
                self.model_con.train()
                self.model_dis.train()

                x_0_con = next(datalooper_train_con).to(self.device).float()
                x_0_dis = next(datalooper_train_dis).to(self.device)

                ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
                con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
                    x_0_con,
                    x_0_dis,
                    self.trainer,
                    self.trainer_dis,
                    ns_con,
                    ns_dis,
                    self.transformer_dis,
                    self.config,
                )

                loss_con = con_loss + self.config.lambda_con * con_loss_ns
                loss_dis = dis_loss + self.config.lambda_dis * dis_loss_ns

                optim_con.zero_grad()
                loss_con.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_con.parameters(), self.config.grad_clip
                )
                optim_con.step()
                sched_con.step()

                optim_dis.zero_grad()
                loss_dis.backward()
                torch.nn.utils.clip_grad_value_(
                    self.trainer_dis.parameters(), self.config.grad_clip
                )  # , self.args.clip_value)
                torch.nn.utils.clip_grad_norm_(
                    self.trainer_dis.parameters(), self.config.grad_clip
                )  # , self.args.clip_norm)
                optim_dis.step()
                sched_dis.step()

                if (step + 1) % int(
                    self.data_wrangler.data.get_train_obs() / self.config.batch_size + 1
                ) == 0:
                    logging.info(
                        f"Epoch :{epoch}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}"
                    )
                    logging.info(
                        f"Epoch :{epoch}, CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}"
                    )
                    logging.info(
                        f"Epoch :{epoch}, Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}"
                    )
                    epoch += 1

                pbar.update(1)

        training_duration = time.time() - training_start_time

        if save_model:
            self.save_model()
            self.save_train_time(training_duration)

    def save_model(self):
        ckpt = {
            "model_con": self.model_con.state_dict(),
            "model_dis": self.model_dis.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.ckpt_restore_dir, "ckpt.pt"))

    def load_model(self):
        ckpt = torch.load(os.path.join(self.ckpt_restore_dir, "ckpt.pt"))
        self.model_con.load_state_dict(ckpt["model_con"])
        self.model_dis.load_state_dict(ckpt["model_dis"])
        self.model_con.eval()
        self.model_dis.eval()

    def sample_tabular_data(self, num_samples, seed, batch_size=4096):
        set_seeds(seed)

        net_sampler = GaussianDiffusionSampler(
            self.model_con,
            self.config.beta_1,
            self.config.beta_T,
            self.config.T,
            self.config.mean_type,
            self.config.var_type,
        ).to(self.device)
        trainer_dis = MultinomialDiffusion(
            self.num_class,
            self.train_dis_data_shape,
            self.model_dis,
            self.config,
            timesteps=self.config.T,
            loss_type="vb_stochastic",
        ).to(self.device)

        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        )

        fake_sample = []
        for num_samples in sample_sizes:
            with torch.no_grad():
                x_T_con = torch.randn(num_samples, self.train_con_data_shape[1]).to(
                    self.device
                )
                log_x_T_dis = log_sample_categorical(
                    torch.zeros(
                        num_samples, self.train_dis_data_shape[1], device=self.device
                    ),
                    self.num_class,
                ).to(self.device)
                x_con, x_dis = sampling_with(
                    x_T_con,
                    log_x_T_dis,
                    net_sampler,
                    trainer_dis,
                    self.transformer_con,
                    self.config,
                )
            x_dis = apply_activate(x_dis, self.transformer_dis.output_info)
            sample_con = self.transformer_con.inverse_transform(
                x_con.detach().cpu().numpy()
            )
            sample_dis = self.transformer_dis.inverse_transform(
                x_dis.detach().cpu().numpy()
            )
            sample = np.zeros([num_samples, len(self.con_idx + self.dis_idx)])
            for i in range(len(self.con_idx)):
                sample[:, self.con_idx[i]] = sample_con[:, i]
            for i in range(len(self.dis_idx)):
                sample[:, self.dis_idx[i]] = sample_dis[:, i]
            fake_sample.append(sample)

        X_fake = np.concatenate(fake_sample)
        X_cat_gen = X_fake[:, self.dis_idx]
        X_cont_gen = X_fake[:, self.con_idx]
        y_gen = None

        # round integer-valued continuous features
        if len(self.data_wrangler.cont_idx["int"]) > 0:
            X_cont_gen[:, self.data_wrangler.cont_idx["int"]] = np.round(
                X_cont_gen[:, self.data_wrangler.cont_idx["int"]]
            )

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen
