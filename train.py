"""
Import packages and add current directory to system path for Google Colab Usage
"""
import os
import sys
import yaml
from torch.utils.tensorboard import SummaryWriter
from scripts.trainer import Trainer
from datetime import datetime
import argparse

root_path = os.path.dirname(__file__)
sys.path.append(root_path) 

if __name__ == '__main__':
    # Arg Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=datetime.now().strftime("%d_%m_%Y_%H_%M"), help='where to store tensorboard log')
    args = parser.parse_args()

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

    # evoke training process
    writer = SummaryWriter(log_dir=os.path.join(root_path, 'summary/log/', args.log_dir))
    trainer = Trainer(config, writer)
    trainer.train()
    writer.close()