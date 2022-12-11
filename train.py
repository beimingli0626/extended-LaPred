"""
Import packages and add current directory to system path for Google Colab Usage
"""
import os
import sys
import yaml
from torch.utils.tensorboard import SummaryWriter
from scripts.trainer import Trainer

root_path = os.path.dirname(__file__)
sys.path.append(root_path) 

if __name__ == '__main__':
    config_path = os.path.join(root_path, 'configs/train_nuscenes.yml')
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Dataset path
    config['data_path'] = os.path.join(root_path, 'datasets/nuscenes')

    # Preprocess path
    config['preprocess_path'] = os.path.join(root_path, 'datasets/preprocess/')
    # TODO: change preprocess data path if run full dataset
    config['preprocess_train'] = os.path.join(config['preprocess_path'], 'train_lapred_orig.p')
    config['preprocess_val'] = os.path.join(config['preprocess_path'], 'val_lapred_orig.p')

    # evoke training process
    scheduler = True
    writer = SummaryWriter(os.path.join(root_path, 'summary/logs'))
    trainer = Trainer(config, writer)
    trainer.train(scheduler)
    writer.close()