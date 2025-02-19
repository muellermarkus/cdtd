import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from layers.train_utils import normalize_emb


class TimeStepEmbedding(nn.Module):
    """
    Layer that embeds diffusion timesteps.
    
     Args:
        - dim (int): the dimension of the output.
        - max_period (int): controls the minimum frequency of the embeddings.
        - n_layers (int): number of dense layers
        - fourer (bool): whether to use random fourier features as embeddings
    """
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        n_layers: int = 2,
        fourier: bool = False,
        scale=16,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.n_layers = n_layers
        self.fourier = fourier

        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")

        if fourier:
            self.register_buffer("freqs", torch.randn(dim // 2) * scale)

        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, timesteps):
        if not self.fourier:
            d, T = self.dim, self.max_period
            mid = d // 2
            fs = torch.exp(-math.log(T) / mid * torch.arange(mid, dtype=torch.float32))
            fs = fs.to(timesteps.device)
            args = timesteps[:, None].float() * fs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = timesteps.ger((2 * torch.pi * self.freqs).to(timesteps.dtype))
            emb = torch.cat([x.cos(), x.sin()], dim=1)

        return self.fc(emb)


class FinalLayer(nn.Module):
    """
    Final layer that predicts logits for each category for categorical features 
    and scalers for continuous features.
    """

    def __init__(self, dim_in, categories, num_cont_features, bias_init=None):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories)
        dim_out = sum(categories) + self.num_cont_features
        self.linear = nn.Linear(dim_in, dim_out)
        nn.init.zeros_(self.linear.weight)
        if bias_init is None:
            nn.init.zeros_(self.linear.bias)
        else:
            self.linear.bias = nn.Parameter(bias_init)
        self.split_chunks = [self.num_cont_features, *categories]
        self.cat_idx = 0
        if self.num_cont_features > 0:
            self.cat_idx = 1

    def forward(self, x):
        x = self.linear(x)
        out = torch.split(x, self.split_chunks, dim=-1)

        if self.num_cont_features > 0:
            cont_logits = out[0]
        else:
            cont_logits = None
        if self.num_cat_features > 0:
            cat_logits = out[self.cat_idx :]
        else:
            cat_logits = None

        return cat_logits, cont_logits


class PositionalEmbedder(nn.Module):
    """
    Positional embedding layer for encoding continuous features.
    Adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    """

    def __init__(self, dim, num_features, trainable=False, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(1, num_features, self.half_dim), requires_grad=trainable
        )
        self.sigma = freq_init_scale
        bound = self.sigma * 3
        nn.init.trunc_normal_(self.weights, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, x):
        x = rearrange(x, "b f -> b f 1")
        freqs = x * self.weights * 2 * torch.pi
        fourier = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fourier


class ContEmbedder(nn.Module):
    """
    Embedding layer for continuous features that utilizes Fourier features.
    """

    def __init__(self, dim, num_features, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0
        self.pos_emb = PositionalEmbedder(
            2 * dim, num_features, trainable=True, freq_init_scale=freq_init_scale
        )
        self.nlinear = NLinear(2 * dim, dim, num_features)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.pos_emb(x)
        x = self.nlinear(x)
        return self.act(x)


class NLinear(nn.Module):
    """N separate linear layers for N separate features
    adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    x has typically 3 dimensions: (batch, features, embedding dim)
    """

    def __init__(self, in_dim, out_dim, n):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(n, out_dim))
        d_in_rsqrt = 1 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x):
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x += self.bias
        return x


class FeatCond(nn.Module):
    """
    Feature-specific conditioning module for more complex feature-sensitive conditioning.
    """

    def __init__(self, num_features, d_in, d_out, init_zero=False, cond_dim=None):
        super().__init__()

        self.num_features = num_features
        self.condition = cond_dim is not None
        self.nlinear = NLinear(d_in, d_out, num_features)
        if init_zero:
            nn.init.zeros_(self.nlinear.weight)
            nn.init.zeros_(self.nlinear.bias)
        self.act = nn.SiLU()

        if self.condition:
            self.cond_proj = nn.Linear(cond_dim, d_in * num_features)

    def forward(self, x, c=None):
        if self.condition:
            cond = F.silu(self.cond_proj(c))
            cond = rearrange(cond, "b (f d) -> b f d", f=self.num_features)
            x += cond
        h = self.nlinear(x)
        return self.act(h)


class CatEmbedding(nn.Module):
    """
    Feature-specific embedding layer for categorical features.
    bias = True adds a learnable bias term to each feature, which is is same across categories.
    """

    def __init__(self, dim, categories, cat_emb_init_sigma=0.001, bias=False):
        super().__init__()

        self.categories = torch.tensor(categories)
        categories_offset = self.categories.cumsum(dim=-1)[:-1]
        categories_offset = torch.cat(
            (torch.zeros((1,), dtype=torch.long), categories_offset)
        )
        self.register_buffer("categories_offset", categories_offset)
        self.dim = torch.tensor(dim)

        self.cat_emb = nn.Embedding(sum(categories), dim)
        nn.init.normal_(self.cat_emb.weight, std=cat_emb_init_sigma)

        self.bias = bias
        if self.bias:
            self.cat_bias = nn.Parameter(torch.zeros(len(categories), dim))

    def forward(self, x):
        x = self.cat_emb(x + self.categories_offset)
        if self.bias:
            x += self.cat_bias
        # l2 normalize embedding
        x = normalize_emb(x, dim=2) * self.dim.sqrt()
        return x

    def get_all_feat_emb(self, feat_idx):
        emb_idx = (
            torch.arange(self.categories[feat_idx], device=self.cat_emb.weight.device)
            + self.categories_offset[feat_idx]
        )
        x = self.cat_emb(emb_idx)
        if self.bias:
            x += self.cat_bias[feat_idx]
        x = normalize_emb(x, dim=1) * self.dim.sqrt()
        return x


class MLP(nn.Module):
    def __init__(
        self,
        num_cont_features,
        cat_emb_dim,
        categories,
        num_y_classes,
        emb_dim,
        n_layers,
        n_units,
        proportions=None,
        use_fourier_features=False,
        act="relu",
        feat_spec_cond=False,
        time_fourier=False,
    ):
        super().__init__()

        self.num_cont_features = num_cont_features
        self.num_cat_features = len(categories)
        self.num_features = num_cont_features + self.num_cat_features
        self.time_emb = TimeStepEmbedding(emb_dim, fourier=time_fourier)

        self.y_cond = False
        if num_y_classes is not None:
            self.y_emb = nn.Embedding(num_y_classes, emb_dim)
            self.y_cond = True

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = n_layers * [n_units]

        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers)

        self.use_fourier_features = use_fourier_features
        self.feat_spec_cond = feat_spec_cond
        if self.feat_spec_cond:
            assert self.use_fourier_features
            self.cond_feat = FeatCond(
                self.num_cat_features + num_cont_features,
                cat_emb_dim,
                cat_emb_dim,
                cond_dim=emb_dim,
                init_zero=True,
            )
            proj_dim_in = (
                num_cont_features * cat_emb_dim + self.num_cat_features * cat_emb_dim
            )
        elif self.use_fourier_features:
            proj_dim_in = (
                num_cont_features * cat_emb_dim + self.num_cat_features * cat_emb_dim
            )
        else:
            proj_dim_in = num_cont_features + self.num_cat_features * cat_emb_dim

        if self.use_fourier_features:
            self.cont_emb = ContEmbedder(cat_emb_dim, num_cont_features)

        self.proj = nn.Linear(proj_dim_in, emb_dim)

        # init final layer
        if proportions is None:
            bias_init = None
        else:
            cont_bias_init = torch.zeros((num_cont_features,))
            cat_bias_init = torch.cat(proportions).log()
            bias_init = torch.cat((cont_bias_init, cat_bias_init))

        self.final_layer = FinalLayer(
            out_dims[-1], categories, num_cont_features, bias_init=bias_init
        )

    def forward(
        self,
        x_cat_emb_t,
        x_cont_t,
        time,
        c,
    ):
        # construct time embedding
        cond_emb = self.time_emb(time)

        # construct conditioning embedding if using y_cond
        if self.y_cond:
            cond_emb = cond_emb + F.silu(self.y_emb(c))

        # map inputs to dim_emb
        if self.use_fourier_features:
            x_cont_t = self.cont_emb(x_cont_t)
            x_cont_t = rearrange(x_cont_t, "B F D -> B (F D)")

        if self.feat_spec_cond:
            x = torch.concat(
                (rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1
            )
            x_cond = self.cond_feat(
                torch.concat(
                    (
                        x_cat_emb_t,
                        rearrange(
                            x_cont_t, "B (F D) -> B F D", F=self.num_cont_features
                        ),
                    ),
                    dim=1,
                ),
                cond_emb,
            )  # feat spec encoding path
            x += rearrange(x_cond, "B F D -> B (F D)")  # add back to main path
        else:
            x = torch.concat(
                (rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1
            )

        emb = self.proj(x) + cond_emb
        h = self.fc(emb)

        return self.final_layer(h)


class TabDDPM_MLP(nn.Module):
    """
    TabDDPM-like architecture for both continuous and categorical features.
    Used for TabDDPM and CDTD.
    """

    def __init__(
        self,
        num_cont_features,
        cat_emb_dim,
        categories,
        num_y_classes,
        emb_dim,
        n_layers,
        n_units,
        proportions=None,
        use_fourier_features=False,
        act="relu",
    ):
        super().__init__()

        num_cat_features = len(categories)
        self.time_emb = TimeStepEmbedding(emb_dim, fourier=False)

        self.y_cond = False
        if num_y_classes is not None:
            self.y_emb = nn.Embedding(num_y_classes, emb_dim)
            self.y_cond = True

        self.use_fourier_features = use_fourier_features
        if self.use_fourier_features:
            self.cont_emb = ContEmbedder(cat_emb_dim, num_cont_features)

        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = n_layers * [n_units]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        self.fc = nn.Sequential(*layers)

        if self.use_fourier_features:
            dim_in = (num_cont_features + num_cat_features) * cat_emb_dim
        else:
            dim_in = num_cont_features + num_cat_features * cat_emb_dim
        self.proj = nn.Linear(dim_in, emb_dim)

        # init final layer
        if proportions is None:
            bias_init = None
        else:
            cont_bias_init = torch.zeros((num_cont_features,))
            cat_bias_init = torch.cat(proportions).log()
            bias_init = torch.cat((cont_bias_init, cat_bias_init))

        self.final_layer = FinalLayer(
            out_dims[-1], categories, num_cont_features, bias_init=bias_init
        )

    def forward(self, x_cat_emb_t, x_cont_t, time, c):
        # construct time embedding
        cond_emb = self.time_emb(time)

        # construct conditioning embedding if using y_cond
        if self.y_cond:
            cond_emb = cond_emb + F.silu(self.y_emb(c))

        if self.use_fourier_features:
            x_cont_t = self.cont_emb(x_cont_t)
            x_cont_t = rearrange(x_cont_t, "B F D -> B (F D)")

        x = torch.concat((rearrange(x_cat_emb_t, "B F D -> B (F D)"), x_cont_t), dim=-1)
        x = self.proj(x) + cond_emb
        x = self.fc(x)

        return self.final_layer(x)


class TabDDPM_MLP_Cont(nn.Module):
    """
    TabDDPM-like architecture for continuous features only.
    This is used for TabSyn as a score model for learned latents.
    """

    def __init__(self, num_features, emb_dim, n_layers, n_units, act="relu"):
        super().__init__()

        self.time_emb = TimeStepEmbedding(emb_dim, fourier=False)
        in_dims = [emb_dim] + (n_layers - 1) * [n_units]
        out_dims = n_layers * [n_units]
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            layers.append(nn.Linear(in_dims[i], out_dims[i]))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
        # add final layer
        layers.append(nn.Linear(out_dims[-1], num_features))
        self.fc = nn.Sequential(*layers)
        self.proj = nn.Linear(num_features, emb_dim)

    def forward(self, x, time):
        cond_emb = self.time_emb(time)
        x = self.proj(x) + cond_emb
        return self.fc(x)
