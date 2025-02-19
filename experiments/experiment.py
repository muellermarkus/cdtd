import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

import plotting
from data.data_prep import DataWrangler
from evaluation.eval_detection import DetectionScore
from evaluation.eval_ml_efficiency import (
    MLEfficiencyScores,
    average_efficiency_results,
    get_ml_efficiency_table,
)
from evaluation.eval_similarity import SimilarityScores


class Experiment(ABC):
    def __init__(self, config, exp_path, dataset):
        self.common_config = OmegaConf.load("configs/common_config.yaml")
        self.device = self.common_config.device
        self.seed = self.common_config.seed
        self.eval_sample_iter = self.common_config.eval_sample_iter
        self.eval_model_iter = self.common_config.eval_model_iter
        self.config = config

        # define paths
        if exp_path is None:
            exp_path = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        self.workdir = os.path.join(
            Path(__file__).parent.parent, "results", dataset, exp_path
        )
        self.logdir = os.path.join(self.workdir, "logs")
        self.config_dir = os.path.join("configs", self.config.model_name)
        self.ckpt_restore_dir = os.path.join(self.workdir, "checkpoints")
        self.create_folders()

        # initialize dataset
        logging.warning(f"=== Initializing {dataset} dataset ===")
        self.dataset = dataset
        self.data_wrangler = DataWrangler(
            self.dataset,
            self.logdir,
            self.config,
            self.common_config.val_prop,
            self.common_config.test_prop,
            self.seed,
        )

        # sample as many observations as there are in the real train set if < 50k
        num_real_train_samples = self.data_wrangler.data.get_train_obs()
        if num_real_train_samples < 50000:
            self.num_samples = num_real_train_samples
        else:
            self.num_samples = 50000

        print(
            f"{self.data_wrangler.data.get_total_obs()} total obs. with {self.data_wrangler.num_total_features} features."
        )
        print(
            f"number of categories between {min(self.data_wrangler.num_cats)} and {max(self.data_wrangler.num_cats)}."
        )
        print(f"CUDA available? {torch.cuda.is_available()}")
        logging.info("=== Done with Experiment.__init__ ===")

    @abstractmethod
    def train(self, **kwargs): ...

    @abstractmethod
    def sample_tabular_data(self, num_samples, **kwargs): ...

    @abstractmethod
    def save_model(self): ...

    @abstractmethod
    def load_model(self): ...

    def evaluate_generative_model(self, tune_catboost=True):
        logging.warning("=== Load generative model... ===")
        self.load_model()

        logging.warning("=== Start evaluation of generative model... ===")
        out_dict = {}

        # time sampling speed for 1000 samples
        sample_start_time = time.time()
        X_cat_gen, X_cont_gen, y_gen = self.sample_tabular_data(1000, seed=42)
        sample_duration = time.time() - sample_start_time
        self.save_sample_time(sample_duration)

        # sample data for plots, etc.
        X_cat_gen, X_cont_gen, y_gen = self.sample_tabular_data(
            self.num_samples, seed=42
        )
        self.data_wrangler.save_data(X_cat_gen, X_cont_gen, y_gen)

        # plot generated features and true features
        X_cat_train, X_cont_train, y_train = self.data_wrangler.data.get_train_data()

        try:
            plotting.plot_feature_distributions(
                self.logdir,
                X_cat_train,
                X_cont_train,
                y_train,
                X_cat_gen,
                X_cont_gen,
                y_gen,
                self.data_wrangler.task,
            )
        except:
            print("plotting of feature distributions failed")

        # estimate ML efficiency
        logging.warning("=== Start estimation of ML efficiency... ===")

        X_cat, X_cont, y = self.data_wrangler.data.get_data()
        sim_scores = SimilarityScores(X_cat, X_cont, y, self.data_wrangler.task)
        catboost_cfg = OmegaConf.load("configs/catboost/default.yaml")
        ml_eff_scores = MLEfficiencyScores(
            X_cat,
            X_cont,
            y,
            self.data_wrangler.task,
            n_runs=self.eval_model_iter,
            catboost_cfg=catboost_cfg,
        )
        detection = DetectionScore(
            X_cat,
            X_cont,
            y,
            self.data_wrangler.y_cond,
            self.data_wrangler.task,
            tune_model=tune_catboost,
        )

        l2_diff_corr_gen = []
        dcr = []
        stat_sim = []
        ml_eff_results = []
        ml_eff_per_model = []
        corr_min_max = []

        for i in range(self.eval_sample_iter):
            if i > 0:
                X_cat_gen, X_cont_gen, y_gen = self.sample_tabular_data(
                    self.num_samples, seed=42 + i
                )

            # compute scores
            dcr.append(self.data_wrangler.get_DCR(X_cat_gen, X_cont_gen, y_gen))
            stat_sim.append(sim_scores.compute_similarity(X_cat_gen, X_cont_gen, y_gen))
            corr_results = sim_scores.compute_diff_in_corr(X_cat_gen, X_cont_gen, y_gen)
            l2_diff_corr_gen.append(corr_results["l2_norm"])

            corr = {}
            corr["min"] = corr_results["abs_diff"][corr_results["abs_diff"] > 0].min()
            corr["max"] = corr_results["abs_diff"].max()
            corr_min_max.append(corr)

            ml_eff_score, avg_results_per_model = ml_eff_scores.compute_ml_eff_score(
                X_cat_gen, X_cont_gen, y_gen
            )
            ml_eff_results.append(ml_eff_score)
            ml_eff_per_model.append(avg_results_per_model)

        # compute average and standard deviation of ml efficience over model and sample seeds
        df_ml_eff_real = pd.concat(
            [res["real"] for res in ml_eff_results], ignore_index=True
        )
        df_ml_eff_gen = pd.concat(
            [res["gen"] for res in ml_eff_results], ignore_index=True
        )
        out_dict["ml_efficiency"] = {
            "mean_real": df_ml_eff_real.mean(0).to_dict(),
            "mean_gen": df_ml_eff_gen.mean(0).to_dict(),
            "std_real": df_ml_eff_real.std(0).to_dict(),
            "std_gen": df_ml_eff_gen.std(0).to_dict(),
        }
        out_dict["raw_ml_eff_real"] = df_ml_eff_real
        out_dict["raw_ml_eff_gen"] = df_ml_eff_gen

        # output average performance per model over sample seeds
        avg_results_real, avg_results_gen, _, _ = average_efficiency_results(
            ml_eff_per_model
        )
        ml_efficiency_tbl = get_ml_efficiency_table(avg_results_real, avg_results_gen)
        with open(os.path.join(self.logdir, "ml_efficiency.txt"), "w") as f:
            f.write(str(ml_efficiency_tbl))

        # average stat similarity results and dcr results
        out_dict["l2_diff_corr_test"] = sim_scores.corr_test_diffs["l2_norm"].item()
        out_dict["l2_diff_corr_mean"] = torch.tensor(l2_diff_corr_gen).mean().item()
        out_dict["l2_diff_corr_std"] = torch.tensor(l2_diff_corr_gen).std().item()
        corr_min = [d["min"] for d in corr_min_max]
        corr_max = [d["max"] for d in corr_min_max]
        out_dict["abs_diff_corr_min"] = torch.tensor(corr_min).mean().item()
        out_dict["abs_diff_corr_min_std"] = torch.tensor(corr_min).std().item()
        out_dict["abs_diff_corr_max"] = torch.tensor(corr_max).mean().item()
        out_dict["abs_diff_corr_max_std"] = torch.tensor(corr_max).std().item()

        dcr_gen_means = [d["mean_gen"] for d in dcr]
        dcr_real_means = [d["mean_test"] for d in dcr]
        out_dict["dcr_real_mean_mean"] = torch.tensor(dcr_real_means).mean().item()
        out_dict["dcr_real_mean_std"] = torch.tensor(dcr_real_means).std().item()
        out_dict["dcr_gen_mean_mean"] = torch.tensor(dcr_gen_means).mean().item()
        out_dict["dcr_gen_mean_std"] = torch.tensor(dcr_gen_means).std().item()

        stat_sim_cat = [d["cat"] for d in stat_sim]
        stat_sim_cont = [d["cont"] for d in stat_sim]
        out_dict["jsd_full"] = [d["cat_full"] for d in stat_sim]
        out_dict["jsd_mean"] = torch.tensor(stat_sim_cat).mean().item()
        out_dict["jsd_std"] = torch.tensor(stat_sim_cat).std().item()
        out_dict["wd_full"] = [d["cont_full"] for d in stat_sim]
        out_dict["wd_mean"] = torch.tensor(stat_sim_cont).mean().item()
        out_dict["wd_std"] = torch.tensor(stat_sim_cont).std().item()

        stat_sim_cat_min = [d["cat_min"] for d in stat_sim]
        stat_sim_cat_max = [d["cat_max"] for d in stat_sim]
        stat_sim_cont_min = [d["cont_min"] for d in stat_sim]
        stat_sim_cont_max = [d["cont_max"] for d in stat_sim]
        out_dict["jsd_min"] = torch.tensor(stat_sim_cat_min).mean().item()
        out_dict["wd_min"] = torch.tensor(stat_sim_cont_min).mean().item()
        out_dict["jsd_max"] = torch.tensor(stat_sim_cat_max).mean().item()
        out_dict["wd_max"] = torch.tensor(stat_sim_cont_max).mean().item()
        out_dict["jsd_min_std"] = torch.tensor(stat_sim_cat_min).std().item()
        out_dict["wd_min_std"] = torch.tensor(stat_sim_cont_min).std().item()
        out_dict["jsd_max_std"] = torch.tensor(stat_sim_cat_max).std().item()
        out_dict["wd_max_std"] = torch.tensor(stat_sim_cont_max).std().item()

        # estimate detection score
        logging.warning("=== Start estimation of detection score... ===")
        X_cat_gen, X_cont_gen, y_gen = self.sample_tabular_data(
            detection.num_samples, seed=42
        )
        detection.tune_detection_model(X_cat_gen, X_cont_gen, y_gen)

        detection_scores = []
        for i in range(self.eval_sample_iter):
            X_cat_gen, X_cont_gen, y_gen = self.sample_tabular_data(
                detection.num_samples, seed=43 + i
            )
            detection_scores.append(
                detection.estimate_score(X_cat_gen, X_cont_gen, y_gen)
            )

        out_dict["detection_mean"] = torch.tensor(detection_scores).mean().item()
        out_dict["detection_std"] = torch.tensor(detection_scores).std().item()

        # save evaluation results
        with open(os.path.join(self.logdir, "eval_results.pkl"), "wb") as f:
            pickle.dump(out_dict, f)

        logging.info("=== Finished evaluation of generative model. ===")

    def create_folders(self):
        # Create log folder
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("Storing logs in:", self.logdir)

        # Create checkpoint folder
        if not os.path.exists(self.ckpt_restore_dir):
            os.makedirs(self.ckpt_restore_dir)
        print("Storing checkpoints in:", self.ckpt_restore_dir)

    def save_train_time(self, duration):
        """Save training time in minutes."""
        with open(os.path.join(self.logdir, "train_time.pkl"), "wb") as f:
            pickle.dump(duration / 60, f)

    def save_sample_time(self, duration):
        """Save sampling time in seconds."""
        with open(os.path.join(self.logdir, "sample_time.pkl"), "wb") as f:
            pickle.dump(duration, f)
