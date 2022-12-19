# extended-LaPred
This repository contains restructured, modularized and extended code, based on ideas proposed in [LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents](https://arxiv.org/abs/2104.00249), and their [original implementation](https://github.com/bdokim/LaPred).

## Introduction
We implement original ideas presented in LaPred in a modularized way, along with our customized extensions. The difference with original implementation includes:
- Modularized codebase, which make it easier to test different encoder-decoder architectures, try different losses, and carry out ablation study.
- Implemented and tested a network variant which consider multiple nearby agents during feature extraction stage, while the original paper consider single nearby agent.
- Implemented a network variant which rank the likelihood of each modality naively and compute the modality selection loss with cosine similarity. Compared the performance with original implementation to justify the necessity of Modality Selection Block (check our report for details).
- All the variants could be easily switched on/off by modifying the configuration file in [./configs](https://github.com/beimingli0626/extended-LaPred/tree/main/configs).

## Dataset

- Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we need:
    - Metadata for all 850 scenes in Trainval split (v1.0)
    - Map expansion pack (v1.3)

- Placed the downloaded data under ./datasets and the directory should be organized as follows
```plain
└── datasets/
    └── nuscenes/
        ├── maps/
        |   ├── basemap/
        |   ├── expansion/
        |   ├── prediction/
        |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
            ...
        |   └── 93406b464a165eaba6d9de76ca09f5da.png
        └── v1.0-trainval
            ├── attribute.json
            ...
            └── visibility.json
        └── v1.0-mini
            ├── attribute.json
            ...
            └── visibility.json         
```

## Data Preprocess
Run the following script execute preprocessing pipeline, which is adopted from original implementation.
```shell
python3 preprocess.py
```
Note that the above script reads in ./configs/preprocess_nuscenes.yml as configuration file, where you could specify preprocessing parameters and select different data split. The preprocessed data will be saved under ./datasets/preprocess.

## Training

To train the model from scratch, run
```shell
python3 train.py --name <name of your run>
```
Similar to preprocessing, the above script reads in ./configs/train_nuscenes.yml as configuration file, where you could specify training hyperparameters and select different data split. If the logger is set to be 'wandb', the training statistics will be logged to wandb.ai, which is a wonderful tool for visualization. Otherwise, the training script will save training checkpoints and tensorboard logs in the output directory.

## Results
Most of the variations we implemented and experimented achieve minADE5 $\approx$ 1.67 and minADE10 $\approx$ 1.36. We also performs extensive ablation study and experiments to investigate the necessity and role of different submodules, you could refer to our report for experimental details and analysis.

## Citation
Kudos to LaPred authors for their amazing results and original code implementations:
```
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, ByeoungDo and Park, Seong Hyeon and Lee, Seokhwan and Khoshimjonov, Elbek and Kum, Dongsuk and Kim, Junsoo and Kim, Jeong Soo and Choi, Jun Won},
    title     = {LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14636-14645}
}

@misc{lapredCode,
    author={Kim, ByeoungDo and Park, Seong Hyeon and Lee, Seokhwan and Khoshimjonov, Elbek and Kum, Dongsuk and Kim, Junsoo and Kim, Jeong Soo and Choi, Jun Won},
    title = {LaPred Github Repository},
    howpublished = {\url{https://github.com/bdokim/LaPred}},
    note = {Accessed: 2022-12-17}
}
```
