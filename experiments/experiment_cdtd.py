import logging
import os
import time
from pprint import pformat

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import plotting
from layers.mlp import MLP, TabDDPM_MLP
from layers.train_utils import InverseSquareRootScheduler, LinearScheduler, cycle
from mixed_type_diffusion import MixedTypeDiffusion

from .experiment import Experiment
from .utils import get_total_trainable_params, set_seeds


class Experiment_CDTD(Experiment):
    """Train and evaluate a continuous diffusion model for mixed-type tabular data."""

    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def get_model(self):
        args = self.config.model
        self.categories = self.data_wrangler.num_cats
        self.num_cat_features = self.data_wrangler.num_cat_features
        self.num_cont_features = self.data_wrangler.num_cont_features
        self.num_features = self.data_wrangler.num_total_features
        self.simulate_missings = self.config.training.get("simulate_missings", False)

        # derive proportions for max CE losses at t = 1 for normalization
        self.calibrate_losses = args.calibrate_losses

        if self.calibrate_losses:
            proportions = []
            n_sample = self.train_loader.X_cat.shape[0]
            for i in range(len(self.categories)):
                _, counts = self.train_loader.X_cat[:, i].unique(return_counts=True)
                proportions.append(counts / n_sample)
            self.proportions = proportions
        else:
            self.proportions = None

        if self.config.model.architecture == "mlp":
            score_model = MLP(
                self.num_cont_features,
                args.dim,
                self.categories,
                self.data_wrangler.num_y_classes,
                args.mlp_emb_dim,
                args.mlp_n_layers,
                args.mlp_n_units,
                proportions=self.proportions,
                use_fourier_features=args.use_fourier_features,
                act=args.act,
                feat_spec_cond=args.use_feat_spec_cond,
                time_fourier=args.use_time_fourier,
            )

        elif self.config.model.architecture == "tabddpm":
            score_model = TabDDPM_MLP(
                self.num_cont_features,
                args.dim,
                self.categories,
                self.data_wrangler.num_y_classes,
                args.mlp_emb_dim,
                args.mlp_n_layers,
                args.mlp_n_units,
                proportions=self.proportions,
                use_fourier_features=args.use_fourier_features,
            )

        return MixedTypeDiffusion(
            model=score_model,
            dim=args.dim,
            categories=self.categories,
            num_features=self.num_features,
            task=self.data_wrangler.task,
            sigma_data_cat=args.sigma_data_cat,
            sigma_data_cont=args.sigma_data_cont,
            sigma_min_cat=args.sigma_min_cat,
            sigma_max_cat=args.sigma_max_cat,
            sigma_min_cont=args.sigma_min_cont,
            sigma_max_cont=args.sigma_max_cont,
            calibrate_losses=args.calibrate_losses,
            proportions=self.proportions,
            cat_emb_init_sigma=args.cat_emb_init_sigma,
            timewarp_variant=args.timewarp_variant,
            timewarp_type=args.timewarp_type,
            timewarp_weight_low_noise=args.timewarp_weight_low_noise,
            timewarp_bins=args.timewarp_bins,
            timewarp_decay=args.timewarp_decay,
            cat_bias=args.use_cat_bias,
            simulate_missings=self.simulate_missings,
        )

    def get_optimizer(self):
        config = self.config.optimizer

        if config.name == "adam":
            optimizer = torch.optim.Adam(self.diff_model.parameters(), **config.args)
            if hasattr(config, "gradient_clip_norm"):
                NotImplementedError()
        elif config.name == "adamw":
            optimizer = torch.optim.AdamW(self.diff_model.parameters(), **config.args)
        else:
            raise Exception("Unknown optimizer.")

        return optimizer

    def get_lr_scheduler(self):
        if self.config.training.scheduler == "linear":
            scheduler = LinearScheduler(
                self.config.training.num_steps_train,
                base_lr=self.config.optimizer.args.lr,
                final_lr=1e-6,
                warmup_steps=self.config.training.num_steps_lr_warmup,
                warmup_begin_lr=1e-6,
                anneal_lr=self.config.training.anneal_lr,
            )
        elif self.config.training.scheduler == "inverse_sqrt":
            scheduler = InverseSquareRootScheduler(
                base_lr=self.config.optimizer.args.lr,
                ref_step=self.config.training.ref_step,
                warmup_steps=self.config.training.num_steps_lr_warmup,
                anneal_lr=self.config.training.anneal_lr,
                warmup_begin_lr=1e-6,
            )
        else:
            scheduler = None

        return scheduler

    def train(self, **kwargs):
        plot_figures = kwargs.get("plot_figures", False)
        save_model = kwargs.get("save_model", False)

        # get train data
        self.train_loader = self.data_wrangler.get_train_loader(
            self.config.training.batch_size
        )
        train_iter = cycle(self.train_loader)

        logging.warning("=== Initializing model ===")
        set_seeds(self.seed, cuda_deterministic=True)
        self.diff_model = self.get_model().to(self.device)
        self.diff_model.train()
        print(
            f"total number of trainable parameters = {get_total_trainable_params(self.diff_model)}."
        )
        self.ema_diff_model = ExponentialMovingAverage(
            self.diff_model.parameters(), decay=self.config.optimizer.ema_decay
        )

        # initialize optimizer
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()

        logging.warning("=== Start training... ===")
        config = self.config.training

        # initialize logging
        self.current_step = 0
        if save_model:
            writer = SummaryWriter(os.path.join(self.logdir, "tb"))
            with open(os.path.join(self.logdir, "hparams.txt"), "w") as f:
                f.write(pformat(self.config))

        training_start_time = time.time()
        with tqdm(
            initial=self.current_step,
            total=config.num_steps_train,
            disable=(not save_model),
        ) as pbar:
            while self.current_step < config.num_steps_train:
                is_last_step = self.current_step == (config.num_steps_train - 1)
                self.optimizer.zero_grad()

                inputs = next(train_iter)
                x_cat, x_cont, y_cond = (
                    input.to(self.device) if input is not None else None
                    for input in inputs
                )

                losses, _ = self.diff_model.loss_fn(x_cat, x_cont, y_cond)
                losses["train_loss"].backward()

                # update parameters
                self.optimizer.step()
                self.diff_model.timewarp_cdf.update_ema()
                self.ema_diff_model.update()

                # track metric
                train_dict = self.get_metric_dict(
                    losses["train_loss"],
                    losses["weighted_calibrated"],
                    losses["timewarping"],
                )

                # write to summaryfile
                if save_model:
                    if (
                        self.current_step % config.steps_per_logging == 0
                        or is_last_step
                    ):
                        self.log_fn(writer, self.current_step, train_dict)

                self.current_step += 1
                pbar.update(1)

                # anneal learning rate
                if self.scheduler:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.scheduler(self.current_step)

        # compute training duration
        training_duration = time.time() - training_start_time
        self.save_train_time(training_duration)

        # take EMA of model parameters
        self.ema_diff_model.copy_to()
        self.diff_model.eval()
        
        if plot_figures:
            self.evaluate_loss_over_time(self.train_loader, prefix="train_")

        if save_model:
            self.save_model()
            writer.close()
        logging.info("=== Finished model training. ===")

    def evaluate_loss_over_time(self, data_loader, steps=100, prefix="train_"):
        """Generate figures of loss over time over whole set.

        Args:
            split (str): train, val, test set selection
            steps (int, optional): number of timesteps to form the grid. Defaults to 100.
        """

        u_list = torch.linspace(1e-5, 1, steps, device=self.device)
        X_cat = data_loader.X_cat.clone()
        X_cont = data_loader.X_cont.clone()
        y = data_loader.y.clone() if self.config.model.y_cond else None

        # if full dataset is too large, subsample 50k
        train_obs = self.data_wrangler.data.get_train_obs()
        if train_obs > 50000:
            idx = torch.randperm(X_cat.shape[0], device=X_cat.device)[:50000]
            X_cat = X_cat[idx]
            X_cont = X_cont[idx]
            if y is not None:
                y = y[idx]
            train_obs = 50000

        data_split = torch.split(torch.arange(train_obs), 3000)

        data_dict = {
            "weighted_avg": [],
            "weighted_std": [],
            "unweighted_avg": [],
            "unweighted_std": [],
            "cdf_avg": [],
            "cdf_std": [],
            "sigma": [],
        }

        for u in tqdm(u_list):
            weighted_losses_per_u = []
            unweighted_losses_per_u = []
            cdf_losses_per_u = []
            sigmas = None

            for idx in data_split:
                x_cat = X_cat[idx].to(self.device)
                x_cont = X_cont[idx].to(self.device)

                if self.config.model.y_cond:
                    y_cond = y[idx].to(self.device)
                else:
                    y_cond = None

                batch = x_cat.shape[0] if x_cat is not None else x_cont.shape[0]
                u_tensor = torch.full((batch,), fill_value=u, device=self.device)
                with torch.no_grad():
                    losses, sigmas = self.diff_model.loss_fn(
                        x_cat, x_cont, y_cond, u=u_tensor
                    )

                if self.diff_model.calibrate_losses:
                    weighted_losses_per_u.append(losses["weighted_calibrated"].cpu())
                    unweighted_losses_per_u.append(
                        losses["unweighted_calibrated"].cpu()
                    )
                else:
                    weighted_losses_per_u.append(
                        torch.zeros_like(losses["unweighted"].cpu())
                    )
                    unweighted_losses_per_u.append(losses["unweighted"].cpu())

                cdf_losses_per_u.append(losses["timewarping"].cpu())

            data_dict["sigma"].append(sigmas[0, :].cpu())

            # average over batch dimension
            weighted_loss = torch.cat(weighted_losses_per_u)
            data_dict["weighted_avg"].append(weighted_loss.mean(0))
            data_dict["weighted_std"].append(weighted_loss.std(0))

            unweighted_loss = torch.cat(unweighted_losses_per_u)
            data_dict["unweighted_avg"].append(unweighted_loss.mean(0))
            data_dict["unweighted_std"].append(unweighted_loss.std(0))

            cdf_loss = torch.cat(cdf_losses_per_u)
            data_dict["cdf_avg"].append(cdf_loss.mean(0))
            data_dict["cdf_std"].append(cdf_loss.std(0))

        data_dict = {k: torch.stack(v) for k, v in data_dict.items() if v is not None}
        data_dict["u"] = u_list.detach().cpu()

        plotting.plot_cdf_and_loss(
            self.logdir,
            self.diff_model,
            data_dict,
            self.num_cat_features,
            prefix=prefix,
        )

        plotting.plot_weighting(self.logdir, self.diff_model, prefix=prefix)

    def sample_tabular_data(self, num_samples, **kwargs):
        seed = kwargs.get("seed", None)
        verbose = kwargs.get("verbose", False)

        if seed:
            set_seeds(seed, cuda_deterministic=True)

        if self.config.model.y_cond:
            _, _, y_train = self.data_wrangler.data.get_train_data()
            _, y_counts = np.unique(y_train, return_counts=True)
            y_dist = y_counts / y_counts.sum()
        else:
            y_dist = None

        X_cat_gen, X_cont_gen, y_gen = self.diff_model.sample_data(
            num_samples,
            self.config.training.batch_size_eval,
            y_dist,
            self.config.model.generation_steps,
            verbose=verbose,
        )

        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen.to(torch.long).numpy(),
            X_cont_gen.numpy(),
            y_gen.numpy() if y_gen is not None else None,
        )
        X_cat_gen = X_cat_gen.astype(int)

        return X_cat_gen, X_cont_gen, y_gen

    def get_metric_dict(self, diff_loss, weighted_losses, timewarp_losses):
        scalar_dict = {
            "total_train_loss": diff_loss.detach().mean().item(),
            "timewarp/total_loss": timewarp_losses.detach().mean().item(),
        }

        if weighted_losses is not None:
            scalar_dict["avg_weighted_loss"] = weighted_losses.detach().mean().item()

        return scalar_dict

    def log_fn(self, writer, step, train_dict):
        for metric_name, metric_value in train_dict.items():
            writer.add_scalar(
                "train/{}".format(metric_name), metric_value, global_step=step
            )

    def save_model(self):
        checkpoint = {
            "current_step": self.current_step,
            "diff_model": self.diff_model.state_dict(),
            "data_wrangler": self.data_wrangler,
            "train_loader": self.train_loader,
        }
        torch.save(checkpoint, os.path.join(self.ckpt_restore_dir, "model.pt"))

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.ckpt_restore_dir, "model.pt"))
        self.current_step = checkpoint["current_step"]
        self.train_loader = checkpoint["train_loader"]
        self.data_wrangler = checkpoint["data_wrangler"]
        self.diff_model = self.get_model()
        self.diff_model.load_state_dict(checkpoint["diff_model"])
        self.diff_model.to(self.device)
        self.diff_model.eval()
