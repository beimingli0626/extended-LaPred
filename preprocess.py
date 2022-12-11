import os
import sys
import pickle
from tqdm import tqdm
import yaml

import numpy as np
from torch.utils.data import DataLoader
import torch.multiprocessing

from scripts.utils import to_numpy, to_int16
from datasets.nuScenes import nuScenesDataset, collate_fn

root_path = os.path.dirname(__file__)
sys.path.append(root_path) 
sys.path

# Benefit: don't need cache the file descriptors, comparing to 'file_descriptor'
torch.multiprocessing.set_sharing_strategy('file_system')
os.umask(0)
os.environ['CUDA_VISIBLE_DEVICES']='0'  # use only one GPU

"""
Configuration for dataset initialization
"""
# Load numeric configuration from yml file in confis
config_path = os.path.join(root_path, "configs/preprocess_nuscenes.yml")
with open(config_path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Dataset path
config["data_path"] = os.path.join(root_path, 'datasets/nuscenes')

# Preprocess path
config["preprocess_path"] = os.path.join(root_path, "datasets/preprocess/")

# Make preprocess directory if doesn't exist
os.makedirs(os.path.dirname(config["preprocess_path"]), exist_ok=True)

def preprocessor(split) :
    """
    preprocess the data implicitly during __getitem__
    and dump into a file through pickle
    """
    dataset = nuScenesDataset(split=split, config=config)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=16, \
        shuffle=False, collate_fn=collate_fn, pin_memory=True, drop_last=False)

    stores = [None for _ in range(len(dataset))]

    for i, data in enumerate(tqdm(train_loader)):
      for j in range(len(data["idx"])) :
        store = dict()
        for key in ["idx","feats","ctrs","orig","theta","rot", \
            "gt_preds","has_preds","ins_sam","map_info"] :
          # take in arbitrary data type, convert to numpy arrays
          store[key] = to_numpy(data[key][j])
          if key == "map_info":
            # take in numpy arrays, convert to int16
            store[key] = to_int16(store[key])
        stores[store["idx"]] = store

    # dump into file
    file_name = '{}_lapred_orig.p'.format(split)
    f = open(os.path.join(config["preprocess_path"], file_name), 'wb')
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == "__main__":
    if config['dataset'] == 'v1.0-trainval':
        print('preprocess train dataset')
        preprocessor('train') 
        print('preprocess val dataset')
        preprocessor('val') 
    elif config['dataset'] == 'v1.0-mini':
        print('preprocess mini train dataset')
        preprocessor('mini_train') 
        print('preprocess mini val dataset')
        preprocessor('mini_val')   