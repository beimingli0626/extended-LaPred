from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

from datasets.nuScenes import nuScenesDataset, collate_fn
from models.Prediction import PredictionModel
from scripts.initialization import initialize_metric
from metrics.evaluation_metrics import EvaluationMetrics
from scripts.utils import CosineLR


class Trainer:
    """
    Trainer class for running train loops
    """
    def __init__(self, config, writer):
        """
        Initialize trainer object
        :param config: Configuration parameters
        :param write: Tensorboard Writer
        """
        self.config = config

        # Initialize datasets
        train_set = nuScenesDataset('train', self.config)
        val_set = nuScenesDataset('val', self.config)

        # Initialize dataloaders
        self.train_loader = DataLoader(train_set, config['batch_size'], shuffle=True, pin_memory=True, \
                                      num_workers=self.config['num_workers'], collate_fn=collate_fn)
        self.val_loader = DataLoader(val_set, config['batch_size'], shuffle=True, pin_memory=True, \
                                    num_workers=self.config['num_workers'], collate_fn=collate_fn)
        
        # Define model
        self.model = PredictionModel(config).cuda()

        # Define optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optim_args']['lr'])
        if config['optim_args']['scheduler'] == 'cos':
            self.scheduler = CosineLR(self.config, self.optimizer)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['optim_args']['scheduler_step'], \
                                                            gamma=config['optim_args']['scheduler_gamma'])

        # Define losses
        self.losses = [initialize_metric(config['losses'][i]) for i in range(len(config['losses']))]
        self.loss_weights = self.config['loss_weights']

        # Define evaluation metrics
        self.metrics = EvaluationMetrics()

        # Define training stats logger
        self.writer = writer
        self.tb_iters = 0
    

    def train(self):
        """
        Main function to train model
        """
        for epoch in range(self.config['current_epoch'], self.config['total_epoch']):
            # Train
            train_metrics, train_loss_dict = self.run_epoch('train', self.train_loader)

            # Validation
            val_metrics, val_loss_dict = self.run_epoch('val', self.val_loader)

            # Log Tensorboard
            self.log_tensorboard(train_metrics, train_loss_dict, val_metrics, val_loss_dict)

            # Print current status for sanity check
            self.print_metrics(epoch, train_metrics, train_loss_dict, val_metrics, val_loss_dict)


    def run_epoch(self, mode, dataloader):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dataloader: dataloader object
        :return metrics: performance metric for last batch in this epoch
        :return loss_dict: dictionary consists of each seperate loss
        """
        if mode == 'val':
            self.model.eval()
        elif mode == 'train':
            self.model.train()
            if self.config['optim_args']['scheduler'] == 'cos': self.scheduler.step() # CosineLR step happens before first epoch

        iter_metrics = []
        for _, data in tqdm(enumerate(dataloader)):
            # Model prediction
            prediction = self.model(data)

            # Loss and backprop if training
            total_loss, loss_dict = self.compute_loss(prediction, data)
            if mode == 'train': self.back_prop(total_loss)

            # Get metrics for each iteration
            iter_metrics.append(self.metrics.compute(prediction, data))
        
        # Aggregate metrics per batch to get metrics across the epoch
        metrics = self.aggregate_metrics(iter_metrics)

        # StepLR step
        if self.config['optim_args']['scheduler'] != 'cos' and mode == 'train': self.scheduler.step()
        return metrics, loss_dict


    def compute_loss(self, model_outputs, data):
        """
        Computes loss given model outputs and ground truth labels
        """
        loss_values = [loss.compute(model_outputs, data) for loss in self.losses] # list of list
        total_loss = torch.as_tensor(0, device='cuda').float()
        for n in range(len(loss_values)):
            total_loss += self.loss_weights[n] * loss_values[n][0]  # loss_values[n][0] is the total loss for each loss module, loss_values[n][1..] could be seperate smaller loss
      
        # reformulate loss_values
        loss_dict = {'loss':total_loss.item(),
                      'lane_select_loss':loss_values[0][0].item(), 
                      'position_loss':loss_values[1][1].item(),
                      'lane_off_loss':loss_values[1][2].item(),
                      'modality_select_loss':loss_values[2][0].item()}
        return total_loss, loss_dict

    
    def back_prop(self, loss, grad_clip_thresh=10):
        """
        Backpropagation, clip the gradient to avoid explosion
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
        self.optimizer.step()

    
    def aggregate_metrics(self, iter_metrics: list):
        """
        Aggregate metrics across one epoch
        """
        # add metrics across one epoch
        metrics = None
        for metric in iter_metrics:
            if metrics is None:
                metrics = metric
            else:
                for key, value in metric.items():
                    metrics[key] += value

        # average metrics across one epoch
        data_size = metrics.pop('data_size')
        for key, value in metrics.items():
            metrics[key] /= data_size
        return metrics


    def log_tensorboard(self, train_metrics, train_loss_dict, val_metrics, val_loss_dict):
        """
        Log losses and metrics
        """
        self.tb_iters += 1
        for key, value in train_loss_dict.items():
            self.writer.add_scalar('train/' + key, value, self.tb_iters)
        for key, value in val_loss_dict.items():
            self.writer.add_scalar('val/' + key, value, self.tb_iters)
        for key, value in train_metrics.items():
            self.writer.add_scalar('train/' + key, value, self.tb_iters)
        for key, value in val_metrics.items():
            self.writer.add_scalar('val/' + key, value, self.tb_iters)
        self.writer.flush()

    
    def print_metrics(self, epoch, train_metrics, train_loss_dict, val_metrics, val_loss_dict):
        """
        Print losses and metrics per epoch for sanity check
        """
        print('Epoch: [', epoch + 1, '/', self.config['total_epoch'], ']:')
        print('   Train Losses: ', train_loss_dict)
        print('   Train Metrics: ', train_metrics)
        print('   Validation Losses: ', val_loss_dict)
        print('   Validation Metrics: ', val_metrics)