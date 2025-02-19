# Continuous Diffusion for Mixed-Type Tabular Data

This is the full replication code for the ICLR 2025 paper "Continuous Diffusion for Mixed-Type Tabular Data".
Note that more user-friendly code is available at https://github.com/muellermarkus/cdtd_simple.

![Banner](https://github.com/muellermarkus/cdtd/blob/main/images/cdtd_overview.png)

## Abstract

Score-based generative models, commonly referred to as diffusion models, have proven to be successful at generating text and image data.
However, their adaptation to mixed-type tabular data remains underexplored. 
In this work, we propose CDTD, a Continuous Diffusion model for mixed-type Tabular Data. CDTD is based on a novel combination of score matching and score interpolation to enforce a unified continuous noise distribution for *both* continuous and categorical features. We explicitly acknowledge the necessity of homogenizing distinct data types by relying on model-specific loss calibration and initialization schemes.
To further address the high heterogeneity in mixed-type tabular data, we introduce adaptive feature- or type-specific noise schedules. These ensure balanced generative performance across features and optimize the allocation of model capacity across features and diffusion time.
Our experimental results show that CDTD consistently outperforms state-of-the-art benchmark models, captures feature correlations exceptionally well, and that heterogeneity in the noise schedule design boosts sample quality.

Paper: https://arxiv.org/abs/2312.10431 (published in ICLR 2025)


## Install Instructions

Initialize virtual environment in Python 3.10, e.g., `python3.10 -m venv .venv` on Linux

Activate environment via, e.g., `source .venv/bin/activate` on Linux

Install packages via `pip install -r requirements.txt`.


## Model Training

After the environment has been setup, you can run all models using one of the commands below. Note that this automatically also runs the evaluation, which can be quite costly. 
If you do not want to automatically evaluate the model as well, comment out `experiment.evaluate_generative_model(tune_catboost=True)` in `main.py`.

Note: By default the detection model is tuned. Depending on the dataset, this can take a rather long time. To avoid tuning the detection model, use `experiment.evaluate_generative_model(tune_catboost=True)` in `main.py`.

Note: By default the evaluation is repeated for 5 sample seeds and 10 model seeds (which impact the ML efficiency models). Depending on the dataset and GPU, the evaluation can take several hours. To reduce the computational load, you can reduce the sample and models seeds by lowering `eval_sample_iter` and `eval_model_iter` in `configs/common_config.yaml`.

Outputs will appear in `results/DATA/exp_path`. For any of the commands, please replace `DATA` by one of the datasets names:
- acsincome
- adult
- bank
- beijing
- churn
- covertype
- default
- diabetes
- lending
- news
- nmes

All raw datasets are located in `data/raw_data`.

By default, we save the training data for each model and one set of generated synthetic data under `results/DATA/exp_path/logs/data`.
After running the evaluation, the same folder will contain comparison plots for all features.

If you want to change any of the model parameters, either change them in the corresponding config file (see `cfg_path`) or edit the relevant experiment file in `experiments/` directly.


## Score model architecture

![Architecture](https://github.com/muellermarkus/cdtd_simple/blob/main/images/architecture.png)


### CDTD (ours)

You can select any of our three noise schedule variants by changing the placeholder `SCHEDULE` in the commands below.

- a single noise schedule: `SCHEDULE=single`
- per type noise schedules: `SCHEDULE=bytype`
- per feature noise schedules: `SCHEDULE=all`

```
python main.py DATA cdtd train --cfg_path=configs/cdtd/default_SCHEDULE.yaml --exp_path=cdtd_SCHEDULE
```

### ARF

```
python main.py DATA arf train --cfg_path=configs/arf/default.yaml --exp_path=arf
```


### TVAE

```
python main.py DATA tvae train --cfg_path=configs/tvae/default.yaml --exp_path=tvae
```

### CTGAN

```
python main.py DATA ctgan train --cfg_path=configs/ctgan/default.yaml --exp_path=ctgan
```

### TabDDPM

For classification task datasets

```
python main.py DATA tabddpm train --cfg_path=configs/tabddpm/default_class.yaml --exp_path=tabddpm
```

For regression task datasets

```
python main.py DATA tabddpm train --cfg_path=configs/tabddpm/default_reg.yaml --exp_path=tabddpm`
```

### CoDi

```
python main.py DATA codi train --cfg_path=configs/codi/default.yaml --exp_path=codi`
```

## Example: Absolute difference in correlation matrices for news dataset
![CorrMatrices](https://github.com/muellermarkus/cdtd_simple/blob/main/images/abs_corr_diff_news.png)


## Example: Learned noise schedules on the acsincome dataset

![Schedules](https://github.com/muellermarkus/cdtd_simple/blob/main/images/learned_schedules_acsincome.png)


