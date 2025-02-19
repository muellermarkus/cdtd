import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.pyplot import cm


def plot_cdf_and_loss(
    save_path, diff_model, data_dict, num_cat_features, ax=None, prefix=""
):
    with torch.no_grad():
        losses = data_dict["unweighted_avg"]
        s = data_dict["sigma"].to(diff_model.device)
        y = diff_model.timewarp_cdf(s).cpu()

        x = (
            (s - diff_model.sigma_min) / (diff_model.sigma_max - diff_model.sigma_min)
        ).cpu()
        xlabel = "$\sigma_t$ scaled to [0,1]"

    if diff_model.timewarp_type == "single":
        x = x[:, 0].unsqueeze(1)
        loss = losses.mean(1).unsqueeze(1)
    elif diff_model.timewarp_type == "bytype":
        x = torch.stack((x[:, 0], x[:, -1]), dim=1)
        mse_loss = losses[:, num_cat_features:].mean(1)
        ce_loss = losses[:, :num_cat_features].mean(1)
        loss = torch.stack((ce_loss, mse_loss), dim=1)
    else:
        loss = losses

    if x.shape[1] == 2:
        labels = ["categorical", "continuous"]
    else:
        labels = [f"{i}" for i in range(s.shape[1])]

    colors = cm.tab20.colors * 4

    if ax is None:
        fig, ax = plt.subplots()

    leg_colors = []
    for i in range(x.shape[1]):
        ax.plot(
            x[:, i],
            y[:, i],
            marker="none",
            linestyle="-",
            linewidth=1,
            color=colors[i],
            label=labels[i],
        )
        ax.plot(
            x[:, i],
            loss[:, i],
            marker="none",
            linestyle="--",
            linewidth=1,
            color=colors[i],
            label=labels[i],
        )
        leg_colors.append(mpatches.Patch(color=colors[i], label=labels[i]))

    leg_lines = [
        mlines.Line2D([], [], linestyle="-", label="fitted function", color="grey"),
        mlines.Line2D(
            [], [], linestyle="--", label="true diffusion loss", color="grey"
        ),
    ]

    plt.legend(handles=leg_colors + leg_lines)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.xscale("log")
    plt.xlim(1e-3, 1)
    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, prefix + "cdf_fit.png"),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        return ax


def plot_weighting(save_path, diff_model, ax=None, prefix=""):
    with torch.no_grad():
        u = torch.linspace(0, 1, 500, device=diff_model.device)
        s = diff_model.timewarp_cdf(u, invert=True)
        y = diff_model.timewarp_cdf(s, return_pdf=True, normalize=True).cpu()
        x = (
            (s - diff_model.sigma_min) / (diff_model.sigma_max - diff_model.sigma_min)
        ).cpu()

        # add point at (0,0)
        x = torch.cat((torch.zeros((1, diff_model.num_features)), x), dim=0)
        y = torch.cat((torch.zeros((1, y.shape[1])), y), dim=0)

        if diff_model.timewarp_type == "single":
            x = x[:, 0].unsqueeze(1)
            labels = [f"{i}" for i in range(s.shape[1])]
        elif diff_model.timewarp_type == "bytype":
            x = torch.stack((x[:, 0], x[:, -1]), dim=1)
            labels = ["categorical", "continuous"]
        else:
            labels = [f"{i}" for i in range(s.shape[1])]
        colors = cm.tab20.colors * 4

        if ax is None:
            fig, ax = plt.subplots()
        for i in range(x.shape[1]):
            ax.plot(
                x[:, i],
                y[:, i],
                marker="none",
                linestyle="-",
                linewidth=1,
                color=colors[i],
                label=labels[i],
            )
        if y.shape[1] == 2:
            ax.legend()
        ax.set_xlabel("$\sigma_t$ scaled to [0,1]")
        ax.set_ylabel("$p(\sigma_t)$")

        if save_path is not None:
            fig.savefig(
                os.path.join(save_path, prefix + "pdf_weighting.png"),
                dpi=300,
                bbox_inches="tight",
            )
        else:
            return ax


def plot_feature_distributions(
    logdir, X_cat_train, X_cont_train, y_train, X_cat_gen, X_cont_gen, y_gen, task
):
    for i in range(X_cat_gen.shape[1]):
        counts_true = pd.DataFrame(X_cat_train[:, i]).value_counts()
        props_true = counts_true / sum(counts_true)
        counts_gen = pd.DataFrame(X_cat_gen[:, i]).value_counts()
        props_gen = counts_gen / sum(counts_gen)

        x = []
        y_true_list = []
        y_gen_list = []

        for cat in props_true.keys():
            x.append(cat[0])
            y_true_list.append(props_true[cat])
            if cat in props_gen.keys():
                y_gen_list.append(props_gen[cat])
            else:
                y_gen_list.append(0.0)

        fig, ax = plt.subplots(1, 1)
        ax.bar(
            np.arange(len(x)) - 0.1,
            y_true_list,
            color="tab:orange",
            width=0.2,
            label="true",
        )
        ax.bar(
            np.arange(len(x)) + 0.1,
            y_gen_list,
            color="tab:blue",
            width=0.2,
            label="generated",
        )
        ax.set_xticks(np.arange(len(x)), x)
        ax.set_xticklabels(x)
        plt.legend()

        if i == 0 and task != "regression" and y_train is None:
            label = "dist_target.png"
        else:
            label = f"dist_cat_feature_{i}.png"

        fig.savefig(os.path.join(logdir, "data", label), dpi=300)
        plt.close()

    for i in range(X_cont_gen.shape[1]):
        fig, ax = plt.subplots(1, 1)
        sns.kdeplot(X_cont_train[:, i], ax=ax, color="tab:orange")
        sns.kdeplot(X_cont_gen[:, i], ax=ax, color="tab:blue")
        plt.legend(["true", "generated"])
        plt.axvline(x=X_cont_train[:, i].mean().item(), color="tab:orange")
        plt.axvline(x=X_cont_gen[:, i].mean().item(), color="tab:blue")

        if task == "regression":
            label = f"dist_cont_feature_{i}.png" if i > 0 else "dist_target.png"
        else:
            label = f"dist_cont_feature_{i}.png"

        fig.savefig(os.path.join(logdir, "data", label), dpi=300)
        plt.close()

    if y_train is not None:
        counts_true = pd.DataFrame(y_train).value_counts()
        props_true = counts_true / sum(counts_true)
        counts_gen = pd.DataFrame(y_gen).value_counts()
        props_gen = counts_gen / sum(counts_gen)

        x = []
        y_true_list = []
        y_gen_list = []

        for cat in props_true.keys():
            x.append(cat[0])
            y_true_list.append(props_true[cat])
            if cat in props_gen.keys():
                y_gen_list.append(props_gen[cat])
            else:
                y_gen_list.append(0.0)

        fig, ax = plt.subplots(1, 1)
        ax.bar(
            np.arange(len(x)) - 0.1,
            y_true_list,
            color="tab:orange",
            width=0.2,
            label="true",
        )
        ax.bar(
            np.arange(len(x)) + 0.1,
            y_gen_list,
            color="tab:blue",
            width=0.2,
            label="generated",
        )
        ax.set_xticks(np.arange(len(x)), x)
        ax.set_xticklabels(x)
        plt.legend()

        fig.savefig(os.path.join(logdir, "data", "dist_target.png"), dpi=300)
        plt.close()
