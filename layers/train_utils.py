import math

import torch
import torch.nn.functional as F


def cycle(dl):
    while True:
        for data in dl:
            yield data


def normalize_emb(emb, dim):
    return F.normalize(emb, dim=dim, eps=1e-20)


def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (
        single_u
        + torch.arange(
            0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False
        )
    ) % 1


class LinearScheduler:
    def __init__(
        self,
        max_update,
        base_lr=0.1,
        final_lr=0.0,
        warmup_steps=0,
        warmup_begin_lr=0,
        anneal_lr=False,
    ):
        self.anneal_lr = anneal_lr
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, step):
        increase = (
            (self.base_lr_orig - self.warmup_begin_lr)
            * float(step)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
        if (step <= self.max_update) and self.anneal_lr:
            decrease = (
                (self.final_lr - self.base_lr_orig)
                / (self.max_update - self.warmup_steps)
                * (step - self.warmup_steps)
            )
        return decrease + self.base_lr_orig if self.anneal_lr else self.base_lr_orig


class InverseSquareRootScheduler:
    def __init__(
        self,
        base_lr=0.1,
        ref_step=10000,
        warmup_steps=0,
        warmup_begin_lr=1e-6,
        anneal_lr=False,
    ):
        self.anneal_lr = anneal_lr
        self.base_lr = base_lr
        self.ref_step = ref_step

        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr

    def get_warmup_lr(self, step):
        increase = (
            (self.base_lr - self.warmup_begin_lr)
            * float(step)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
        if self.anneal_lr:
            new_lr = self.base_lr / math.sqrt(max(step / self.ref_step, 1))
        return new_lr if self.anneal_lr else self.base_lr
