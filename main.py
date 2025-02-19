import argparse
from omegaconf import OmegaConf
import os
from experiments import *

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "dataset to be used", type = str)
parser.add_argument("model", help = "generative model", type = str)
parser.add_argument("mode", help = "whether to train or eval the model", type = str, default = 'train')
parser.add_argument("--cfg_path", help = "path to yaml config file", type = str)
parser.add_argument("--exp_path", help = "subfolder in experiments folder in which results are saved", type = str)


def main(args):
    
    if args.cfg_path is None:
        print("load default config")
        standard_params_path = os.path.join('configs', args.model, 'default.yaml')
        config = OmegaConf.load(standard_params_path)
    else:
        config = OmegaConf.load(args.cfg_path)


    if args.model == 'ctgan':
        experiment = Experiment_CTGAN(config, args.exp_path, args.data)
    elif args.model == 'tvae':
        experiment = Experiment_TVAE(config, args.exp_path, args.data)
    elif args.model == 'codi':
        experiment = Experiment_CoDi(config, args.exp_path, args.data)
    elif args.model == 'arf':
        experiment = Experiment_ARF(config, args.exp_path, args.data)
    elif args.model == 'tabddpm':
        experiment = Experiment_TabDDPM(config, args.exp_path, args.data)
    elif args.model == 'cdtd':
        experiment = Experiment_CDTD(config, args.exp_path, args.data)
    elif args.model == 'smote':
        experiment = Experiment_SMOTE(config, args.exp_path, args.data)
    elif args.model == 'tabsyn':
        experiment = Experiment_TabSyn(config, args.exp_path, args.data)
        
    
    if args.mode == "train":
        experiment.train(save_model=True, plot_figures=True)
        experiment.evaluate_generative_model(tune_catboost=True)
    elif args.mode == "eval":
        experiment.evaluate_generative_model(tune_catboost=True)
    elif args.mode == 'impute':
        experiment.evaluate_imputation()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    
    
