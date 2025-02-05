"""
Main file for running the AE example.
"""

import sys
import tensorflow as tf
from omegacli import OmegaConf
import argparse
import train
import copy
from typing import TypedDict
from dotenv import dotenv_values

class EnvConfig(TypedDict):
    PROJECT_DIR: str
    DATASET_DIR: str
    RESULTS_DIR: str

env_config: EnvConfig = dotenv_values(".env")

def main(argv):
    # hide memory from tensorflow or it miht conflict with jax
    tf.config.experimental.set_visible_devices([], "GPU")
    print('launching training...')

    parser = argparse.ArgumentParser("SAE model")
    parser.add_argument("--learning_rate", dest='config.learning_rate', type=float, default=0.0001)
    parser.add_argument("--latent_dim", dest='config.latent_dim', type=int, default=30)
    parser.add_argument("--batch_size", dest='config.batch_size', type=int, default=30)
    parser.add_argument("--num_epochs", dest='config.num_epochs', type=int, default=15)
    parser.add_argument("--roi_class", dest='config.roi_class', default='floc-bodies') # floc-bodies, ...
    parser.add_argument("--hem", dest='config.hem', default='lh') # lh, rh, all
    # parser.add_argument("--ds", dest='config.ds', default='fmri') # mnist, cifar10, fmri
    parser.add_argument("--sparsity", dest='config.sparsity', type=float, default=0.8)
    parser.add_argument("--l1", dest='config.l1', type=float, default=0.1)
    parser.add_argument("--subject", dest='config.subject', type=int, default=3)

    usr_args, default_args = OmegaConf.from_argparse(parser)

    # create the results folder
    # results_folder = f'{env_config.RESULTS_DIR}/subj{usr_args.config.subject}/latent{usr_args.config.latent_dim}_sparsity{usr_args.config.sparsity}_bs{usr_args.config.batch_size}_lOne{usr_args.config.l1}'
    # os.makedirs(results_folder, exist_ok=True)
    # usr_args.config['results_folder'] = results_folder

    # # write the config to the results folder
    # with open(f"{results_folder}/config", 'w') as f:
    #     for key, value in usr_args.config.items():
    #         f.write(f'{key}:{value}\n')

    train.train_and_evaluate(default_args.config if usr_args == None else usr_args.config, env_config)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
