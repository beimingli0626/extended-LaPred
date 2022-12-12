"""
Import packages and add current directory to system path for Google Colab Usage
"""
import os
import sys
import yaml
from torch.utils.tensorboard import SummaryWriter
import wandb
from scripts.trainer import Trainer
from datetime import datetime
import argparse
import torch
import numpy as np
import random

root_path = os.path.dirname(__file__)
sys.path.append(root_path) 

if __name__ == '__main__':
    # Arg Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=datetime.now().strftime("%d_%m_%Y_%H_%M"), help='where to store tensorboard log')
    parser.add_argument('--name', type=str, default='test', help='name of wandb log')
    args = parser.parse_args()

    # Manually set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Configuration
    config_path = os.path.join(root_path, 'configs/train_nuscenes.yml')
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Dataset path
    config['data_path'] = os.path.join(root_path, 'datasets/nuscenes')

    # Preprocess path
    config['preprocess_path'] = os.path.join(root_path, 'datasets/preprocess/')
    if config['dataset'] == 'v1.0-mini':
        train_path, val_path = 'mini_train_lapred_orig.p', 'mini_val_lapred_orig.p'
    else:
        train_path, val_path = 'train_lapred_orig.p', 'val_lapred_orig.p'
    config['preprocess_train'] = os.path.join(config['preprocess_path'], train_path)
    config['preprocess_val'] = os.path.join(config['preprocess_path'], val_path)

    # define logger
    if config['logger'] == 'tensorboard':
        writer = SummaryWriter(log_dir=os.path.join(root_path, 'summary/log/', args.log_dir))
    elif config['logger'] == 'wandb':
        wandb.init(project="LaPred", entity="lapred", group='Extended_LaPred', name=args.name, config=config)
        writer = None

    # evoke training process
    trainer = Trainer(config, writer)
    trainer.train()

    # close tensorboard if needed
    if config['logger'] == 'tensorboard': writer.close()