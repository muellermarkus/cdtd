import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.tabsyn.tabsyn.diffusion_utils import sample
from experiments.tabsyn.tabsyn.model import Model
from experiments.tabsyn.tabsyn.vae.model import Decoder_model, Encoder_model, Model_VAE
from layers.mlp import TabDDPM_MLP_Cont

from .experiment import Experiment
from .utils import set_seeds


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= idx + 1
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


class Experiment_TabSyn(Experiment):
    def __init__(self, config, exp_path, dataset):
        super().__init__(config, exp_path, dataset)

    def train(self, **kwargs):
        d_numerical = self.data_wrangler.num_cont_features
        categories = self.data_wrangler.num_cats

        train_loader = self.data_wrangler.get_train_loader(
            self.config.batch_size, partition="train"
        )
        val_loader = self.data_wrangler.get_train_loader(
            self.config.batch_size, partition="val"
        )

        # filter out observation in validation set with categories not appearing in training set
        idx_unknown_cat = (val_loader.X_cat == 9999).sum(1).to(torch.bool)
        print(f"filter out {idx_unknown_cat.sum().item()} rows with unknown cat")
        X_test_num = val_loader.X_cont[~idx_unknown_cat].to(self.device)
        X_test_cat = val_loader.X_cat[~idx_unknown_cat].to(self.device)

        ############ TRAIN VAE

        set_seeds(self.seed, cuda_deterministic=True)
        model = Model_VAE(
            self.config.num_layers,
            d_numerical,
            categories,
            self.config.d_token,
            n_head=self.config.n_head,
            factor=self.config.factor,
            bias=True,
        )
        model = model.to(self.device)
        vae_model_save_path = os.path.join(self.ckpt_restore_dir, "vae_model.pt")

        pre_encoder = Encoder_model(
            self.config.num_layers,
            d_numerical,
            categories,
            self.config.d_token,
            n_head=self.config.n_head,
            factor=self.config.factor,
        ).to(self.device)
        pre_decoder = Decoder_model(
            self.config.num_layers,
            d_numerical,
            categories,
            self.config.d_token,
            n_head=self.config.n_head,
            factor=self.config.factor,
        ).to(self.device)

        num_params = sum(p.numel() for p in model.parameters())
        print("the number of parameters", num_params)

        pre_encoder.eval()
        pre_decoder.eval()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.wd
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.95, patience=10, verbose=True
        )

        current_lr = optimizer.param_groups[0]["lr"]
        patience = 0
        batch_size = min(
            self.config.batch_size, self.data_wrangler.data.get_train_obs()
        )
        steps_per_epoch = self.data_wrangler.data.get_train_obs() / batch_size
        num_epochs_vae = round(self.config.train_steps_vae / steps_per_epoch)
        print(f"training VAE model for {num_epochs_vae} epochs...")
        best_train_loss = float("inf")
        beta = self.config.max_beta

        vae_start_time = time.time()
        for epoch in tqdm(range(num_epochs_vae)):
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0
            curr_count = 0

            for batch_cat, batch_num, _ in train_loader:
                model.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)

                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

                loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )

                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count

            model.eval()
            with torch.no_grad():
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(
                    X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]["lr"]

                if new_lr != current_lr:
                    current_lr = new_lr

                train_loss = val_loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                else:
                    patience += 1
                    if patience == 10:
                        if beta > self.config.min_beta:
                            beta = beta * self.config.lambd

        vae_training_duration = time.time() - vae_start_time

        # Saving latent embeddings
        with torch.no_grad():
            torch.save(
                self.data_wrangler,
                os.path.join(self.ckpt_restore_dir, "data_wrangler.pt"),
            )
            torch.save(model.state_dict(), vae_model_save_path)
            pre_encoder.load_weights(model)
            pre_decoder.load_weights(model)

            torch.save(
                pre_encoder.state_dict(),
                os.path.join(self.ckpt_restore_dir, "encoder.pt"),
            )
            torch.save(
                pre_decoder.state_dict(),
                os.path.join(self.ckpt_restore_dir, "decoder.pt"),
            )

            data_split = torch.split(
                torch.arange(train_loader.X_cont.shape[0]), self.config.batch_size
            )
            train_z = []
            for idx in data_split:
                X_train_num = train_loader.X_cont[idx].to(self.device)
                X_train_cat = train_loader.X_cat[idx].to(self.device)
                train_z.append(
                    pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
                )
            train_z = np.concatenate(train_z)

            np.save(os.path.join(self.data_wrangler.logdir, "train_z.npy"), train_z)

            print("Successfully save pretrained embeddings in disk!")

        ############ TRAIN DIFFUSION MODEL

        train_z = torch.tensor(train_z).float()
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.shape
        in_dim = num_tokens * token_dim
        train_z = train_z.view(B, in_dim)

        in_dim = train_z.shape[1]
        mean, std = train_z.mean(0), train_z.std(0)
        train_data = (train_z - mean) / 2

        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        num_epochs_diff = round(self.config.train_steps_diff / steps_per_epoch)
        print(f"training diffusion model for {num_epochs_diff} epochs...")
        # denoise_fn = MLPDiffusion(in_dim, 540).to(self.device)
        denoise_fn = TabDDPM_MLP_Cont(
            in_dim,
            self.config.denoiser.emb_dim,
            self.config.denoiser.n_layers,
            self.config.denoiser.n_units,
        )
        num_params = sum(p.numel() for p in denoise_fn.parameters())
        print("the number of parameters", num_params)

        model = Model(denoise_fn=denoise_fn, hid_dim=train_data.shape[1]).to(
            self.device
        )
        diff_model_save_path = os.path.join(self.ckpt_restore_dir, "diff_model.pt")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=20, verbose=True
        )
        model.train()

        best_loss = float("inf")
        patience = 0

        diff_start_time = time.time()
        for epoch in tqdm(range(num_epochs_diff)):
            batch_loss = 0.0
            len_input = 0
            for batch in train_loader:
                inputs = batch.float().to(self.device)
                loss = model(inputs)

                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(model.state_dict(), diff_model_save_path)
            else:
                patience += 1
                if patience == 500:
                    print("Early stopping")
                    break

        diff_training_duration = time.time() - diff_start_time
        self.save_train_time(diff_training_duration + vae_training_duration)

    @torch.inference_mode()
    def sample_tabular_data(self, num_samples, **kwargs):
        seed = kwargs.get("seed", None)
        if seed:
            set_seeds(seed, cuda_deterministic=True)

        sample_dim = self.train_z.shape[1]

        x_next = sample(
            self.model.denoise_fn_D,
            num_samples,
            sample_dim,
            num_steps=self.config.num_sample_steps,
        )
        x_next = x_next * 2 + self.train_z.mean_val.to(self.device)

        syn_data = x_next.float().cpu().numpy()

        syn_data = syn_data.reshape(syn_data.shape[0], -1, self.token_dim)
        norm_input = self.pre_decoder(torch.tensor(syn_data))
        x_hat_num, x_hat_cat = norm_input

        syn_cat = []
        for pred in x_hat_cat:
            syn_cat.append(pred.argmax(dim=-1))

        syn_num = x_hat_num.detach().cpu().numpy()
        syn_cat = torch.stack(syn_cat).t().cpu().numpy()

        X_cat_gen = np.int64(syn_cat)
        X_cont_gen = syn_num
        y_gen = None

        X_cat_gen, X_cont_gen, y_gen = self.data_wrangler.postprocess_gen_data(
            X_cat_gen, X_cont_gen, y_gen
        )
        X_cat_gen = X_cat_gen.astype(int)

        return X_cat_gen.astype("int64"), X_cont_gen.astype("float64"), y_gen

    def save_model(self):
        return

    def load_model(self):
        self.data_wrangler = torch.load(
            os.path.join(self.ckpt_restore_dir, "data_wrangler.pt")
        )
        embedding_save_path = os.path.join(self.data_wrangler.logdir, "train_z.npy")
        train_z = torch.tensor(np.load(embedding_save_path)).float()
        train_z = train_z[:, 1:, :]
        B, num_tokens, self.token_dim = train_z.shape
        in_dim = num_tokens * self.token_dim
        self.train_z = train_z.view(B, in_dim)
        self.train_z.mean_val = self.train_z.mean(0)

        denoise_fn = TabDDPM_MLP_Cont(
            in_dim,
            self.config.denoiser.emb_dim,
            self.config.denoiser.n_layers,
            self.config.denoiser.n_units,
        )

        self.model = Model(denoise_fn=denoise_fn, hid_dim=self.train_z.shape[1]).to(
            self.device
        )
        self.model.load_state_dict(
            torch.load(os.path.join(self.ckpt_restore_dir, "diff_model.pt"))
        )

        d_numerical = self.data_wrangler.num_cont_features
        categories = self.data_wrangler.num_cats
        self.pre_decoder = Decoder_model(
            self.config.num_layers,
            d_numerical,
            categories,
            self.config.d_token,
            n_head=self.config.n_head,
            factor=self.config.factor,
        )
        decoder_save_path = os.path.join(self.ckpt_restore_dir, "decoder.pt")
        self.pre_decoder.load_state_dict(torch.load(decoder_save_path))
        self.model.eval()
        self.pre_decoder.eval()
