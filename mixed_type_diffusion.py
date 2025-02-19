import torch
import torch.nn.functional as F
from torch import nn
from tqdm.autonotebook import tqdm

from layers.adaptive_weighting import WeightNetwork
from layers.mlp import CatEmbedding
from layers.timewarping import Timewarp, Timewarp_Logistic
from layers.train_utils import low_discrepancy_sampler


class MixedTypeDiffusion(nn.Module):
    def __init__(
        self,
        model,
        dim,
        categories,
        num_features,
        task,
        sigma_data_cat,
        sigma_data_cont,
        sigma_min_cat,
        sigma_max_cat,
        sigma_min_cont,
        sigma_max_cont,
        cat_emb_init_sigma,
        timewarp_variant,
        timewarp_type,
        timewarp_bins,
        timewarp_weight_low_noise=1.0,
        timewarp_decay=0.0,
        calibrate_losses=True,
        proportions=None,
        cat_bias=False,
        simulate_missings=False,
    ):
        super(MixedTypeDiffusion, self).__init__()

        self.task = task
        self.dim = dim
        self.num_features = num_features
        self.num_cat_features = len(categories)
        self.num_cont_features = num_features - self.num_cat_features
        self.num_unique_cats = sum(categories)
        self.categories = categories
        self.calibrate_losses = calibrate_losses
        self.model = model
        self.simulate_missings = simulate_missings

        self.cat_emb = CatEmbedding(dim, categories, cat_emb_init_sigma, bias=cat_bias)
        self.register_buffer("sigma_data_cat", torch.tensor(sigma_data_cat))
        self.register_buffer("sigma_data_cont", torch.tensor(sigma_data_cont))

        if self.calibrate_losses:
            assert proportions is not None
            entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
            self.register_buffer(
                "normal_const",
                torch.cat((entropy, torch.ones((self.num_cont_features,)))),
            )
            self.weight_network = WeightNetwork(1024)
        else:
            self.register_buffer("normal_const", torch.ones((self.num_features,)))

        # timewarping
        self.timewarp_type = timewarp_type
        self.sigma_min_cat = torch.tensor(sigma_min_cat)
        self.sigma_max_cat = torch.tensor(sigma_max_cat)
        self.sigma_min_cont = torch.tensor(sigma_min_cont)
        self.sigma_max_cont = torch.tensor(sigma_max_cont)

        # combine sigma boundaries for transforming sigmas to [0,1]
        sigma_min = torch.cat(
            (
                torch.tensor(sigma_min_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_min_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        sigma_max = torch.cat(
            (
                torch.tensor(sigma_max_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_max_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        self.register_buffer("sigma_max", sigma_max)
        self.register_buffer("sigma_min", sigma_min)

        self.timewarp_variant = timewarp_variant
        if self.timewarp_variant == "cdcd":
            self.timewarp_cdf = Timewarp(
                self.timewarp_type,
                self.num_cat_features,
                self.num_cont_features,
                sigma_min,
                sigma_max,
                decay=timewarp_decay,
                num_bins=timewarp_bins,
            )
        else:
            self.timewarp_cdf = Timewarp_Logistic(
                self.timewarp_type,
                self.num_cat_features,
                self.num_cont_features,
                sigma_min,
                sigma_max,
                weight_low_noise=timewarp_weight_low_noise,
                decay=timewarp_decay,
            )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def diffusion_loss(self, x_cat_0, x_cont_0, cat_logits, cont_preds, sigma):
        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_0.shape

        # cross entropy over categorical features for each individual
        ce_losses = torch.stack(
            [
                F.cross_entropy(cat_logits[i], x_cat_0[:, i], reduction="none")
                for i in range(self.num_cat_features)
            ],
            dim=1,
        )

        # MSE loss over numerical features for each individual
        mse_losses = (cont_preds - x_cont_0) ** 2

        return ce_losses, mse_losses

    def add_noise(self, x_cat_emb_0, x_cont_0, sigma):
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        x_cat_emb_t = x_cat_emb_0 + torch.randn_like(x_cat_emb_0) * sigma_cat.unsqueeze(
            2
        )
        x_cont_t = x_cont_0 + torch.randn_like(x_cont_0) * sigma_cont

        return x_cat_emb_t, x_cont_t

    def loss_fn(self, x_cat, x_cont, y_cond, u=None):
        batch = x_cat.shape[0] if x_cat is not None else x_cont.shape[0]

        # get ground truth data
        x_cat_emb_0 = self.cat_emb(x_cat)
        x_cont_0 = x_cont
        x_cat_0 = x_cat

        # draw u and convert to standard deviations for noise
        with torch.no_grad():
            if u is None:
                u = low_discrepancy_sampler(batch, device=self.device)  # (B,)
            sigma = self.timewarp_cdf(u, invert=True).detach().to(torch.float32)
            u = u.to(torch.float32)
            assert sigma.shape == (batch, self.num_features)

        x_cat_emb_t, x_cont_t = self.add_noise(x_cat_emb_0, x_cont_0, sigma)
        cat_logits, cont_preds = self.precondition(
            x_cat_emb_t, x_cont_t, y_cond, u, sigma
        )
        ce_losses, mse_losses = self.diffusion_loss(
            x_cat_0, x_cont_0, cat_logits, cont_preds, sigma
        )

        # compute EDM weight
        sigma_cont = sigma[:, self.num_cat_features :]
        cont_weight = (sigma_cont**2 + self.sigma_data_cont**2) / (
            (sigma_cont * self.sigma_data_cont) ** 2 + 1e-7
        )

        losses = {}
        losses["unweighted"] = torch.cat((ce_losses, mse_losses), dim=1)

        if self.calibrate_losses:
            losses["unweighted_calibrated"] = losses["unweighted"] / self.normal_const
            weighted_calibrated = (
                torch.cat((ce_losses, cont_weight * mse_losses), dim=1)
                / self.normal_const
            )
            c_noise = torch.log(u.to(torch.float32) + 1e-8) * 0.25
            time_reweight = self.weight_network(c_noise).unsqueeze(1)

            losses["timewarping"] = self.timewarp_cdf.loss_fn(
                sigma.detach(), losses["unweighted_calibrated"].detach()
            )
            weightnet_loss = (
                time_reweight.exp() - weighted_calibrated.detach().mean(1)
            ) ** 2
            losses["weighted_calibrated"] = (
                weighted_calibrated / time_reweight.exp().detach()
            )
            train_loss = (
                losses["weighted_calibrated"].mean()
                + losses["timewarping"].mean()
                + weightnet_loss.mean()
            )

        else:
            losses["weighted_calibrated"] = None
            losses["timewarping"] = self.timewarp_cdf.loss_fn(
                sigma.detach(), losses["unweighted"].detach()
            )
            train_loss = losses["unweighted"].mean() + losses["timewarping"].mean()

        losses["train_loss"] = train_loss

        return losses, sigma

    def precondition(self, x_cat_emb_t, x_cont_t, y_cond, u, sigma):
        """
        Improved preconditioning proposed in the paper "Elucidating the Design
        Space of Diffusion-Based Generative Models" (EDM) adjusted for categorical data
        """

        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        c_in_cat = (
            1 / (self.sigma_data_cat**2 + sigma_cat.unsqueeze(2) ** 2).sqrt()
        )  # batch, num_features, 1
        c_in_cont = 1 / (self.sigma_data_cont**2 + sigma_cont**2).sqrt()
        # c_noise = u.log() / 4
        c_noise = torch.log(u + 1e-8) * 0.25 * 1000

        cat_logits, cont_preds = self.model(
            c_in_cat * x_cat_emb_t,
            c_in_cont * x_cont_t,
            c_noise,
            c=y_cond,
        )

        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_t.shape

        # apply preconditioning to continuous features
        c_skip = self.sigma_data_cont**2 / (sigma_cont**2 + self.sigma_data_cont**2)
        c_out = (
            sigma_cont
            * self.sigma_data_cont
            / (sigma_cont**2 + self.sigma_data_cont**2).sqrt()
        )
        D_x = c_skip * x_cont_t + c_out * cont_preds

        return cat_logits, D_x

    def score_interpolation(self, x_cat_emb_t, cat_logits, sigma, return_probs=False):
        if return_probs:
            # transform logits for categorical features to probabilities
            probs = []
            for logits in cat_logits:
                probs.append(F.softmax(logits.to(torch.float64), dim=1))
            return probs

        def interpolate_emb(i):
            p = F.softmax(cat_logits[i].to(torch.float64), dim=1)
            true_emb = self.cat_emb.get_all_feat_emb(i).to(torch.float64)
            return torch.matmul(p, true_emb)

        # take prob-weighted average of normalized ground truth embeddings
        x_cat_emb_0_hat = torch.zeros_like(
            x_cat_emb_t, device=self.device, dtype=torch.float64
        )
        for i in range(self.num_cat_features):
            x_cat_emb_0_hat[:, i, :] = interpolate_emb(i)

        # plug interpolated embedding into score function to interpolate score
        sigma_cat = sigma[:, : self.num_cat_features]
        interpolated_score = (x_cat_emb_t - x_cat_emb_0_hat) / sigma_cat.unsqueeze(2)

        return interpolated_score, x_cat_emb_0_hat

    @torch.inference_mode()
    def sampler(self, cat_latents, cont_latents, y_dist=None, num_steps=200):
        B = (
            cont_latents.shape[0]
            if self.num_cont_features > 0
            else cat_latents.shape[0]
        )

        # in case we condition on label y, sample y first from empirical distribution
        if y_dist is not None:
            y_gen = torch.multinomial(
                torch.tensor(y_dist).to(self.device),
                cont_latents.shape[0],
                replacement=True,
            )
        else:
            y_gen = None

        # construct time steps
        u_steps = torch.linspace(
            1, 0, num_steps + 1, device=self.device, dtype=torch.float64
        )
        t_steps = self.timewarp_cdf(u_steps, invert=True)

        assert torch.allclose(t_steps[0].to(torch.float32), self.sigma_max.float())
        assert torch.allclose(t_steps[-1].to(torch.float32), self.sigma_min.float())
        # the final step goes onto t = 0, i.e., sigma = sigma_min = 0

        # initialize latents at maximum noise level
        t_cat_next = t_steps[0, : self.num_cat_features]
        t_cont_next = t_steps[0, self.num_cat_features :]
        x_cat_next = cat_latents.to(torch.float64) * t_cat_next.unsqueeze(1)
        x_cont_next = cont_latents.to(torch.float64) * t_cont_next

        for i, (t_cur, t_next, u_cur) in enumerate(
            zip(t_steps[:-1], t_steps[1:], u_steps[:-1])
        ):
            t_cur = t_cur.repeat((B, 1))
            t_next = t_next.repeat((B, 1))
            t_cont_cur = t_cur[:, self.num_cat_features :]

            # get score model output
            cat_logits, x_cont_denoised = self.precondition(
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                y_cond=y_gen,
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
            )

            # estimate scores
            d_cat_cur, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur)
            d_cont_cur = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur

            # adjust data samples
            h = t_next - t_cur
            x_cat_next = (
                x_cat_next + h[:, : self.num_cat_features].unsqueeze(2) * d_cat_cur
            )
            x_cont_next = x_cont_next + h[:, self.num_cat_features :] * d_cont_cur

        # final prediction of classes for categorical feature
        u_final = u_steps[:-1][-1]
        t_final = t_steps[:-1][-1].repeat(B, 1)

        cat_logits, _ = self.precondition(
            x_cat_emb_t=x_cat_next.to(torch.float32),
            x_cont_t=x_cont_next.to(torch.float32),
            y_cond=y_gen,
            u=u_final.to(torch.float32).repeat((B,)),
            sigma=t_final.to(torch.float32),
        )

        # get probabilities for each category and derive generated classes
        probs = self.score_interpolation(
            x_cat_next, cat_logits, t_final, return_probs=True
        )
        x_cat_gen = torch.empty(B, self.num_cat_features, device=self.device)
        for i in range(self.num_cat_features):
            x_cat_gen[:, i] = probs[i].argmax(1)

        y_gen = y_gen.cpu() if y_gen is not None else None
        return x_cat_gen.cpu(), x_cont_next.cpu(), y_gen

    @torch.inference_mode()
    def sample_data(self, num_samples, batch_size, y_dist, num_steps, verbose=True):
        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        )

        x_cat_list = []
        x_cont_list = []
        y_list = []

        for num_samples in tqdm(sample_sizes, disable=(not verbose)):
            cat_latents = torch.randn(
                (num_samples, self.num_cat_features, self.dim), device=self.device
            )
            cont_latents = torch.randn(
                (num_samples, self.num_cont_features), device=self.device
            )
            x_cat_gen, x_cont_gen, y_gen = self.sampler(
                cat_latents, cont_latents, y_dist, num_steps
            )
            x_cat_list.append(x_cat_gen)
            x_cont_list.append(x_cont_gen)
            if y_gen is not None:
                y_list.append(y_gen)

        x_cat = torch.cat(x_cat_list).cpu()
        x_cont = torch.cat(x_cont_list).cpu()
        y = torch.cat(y_list).cpu() if len(y_list) > 0 else None

        return x_cat, x_cont, y
