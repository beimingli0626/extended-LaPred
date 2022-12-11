from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from datasets.nuScenes import nuScenesDataset, collate_fn
from models.Prediction import PredictionModel
from scripts.initialization import initialize_metric
from metrics.evaluation_metrics import EvaluationMetrics


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
        self.train_loader = DataLoader(train_set, self.config['batch_size'], shuffle=True, pin_memory=True, \
                                      num_workers=self.config['num_workers'], collate_fn=collate_fn)
        self.val_loader = DataLoader(train_set, config['batch_size'], shuffle=True, pin_memory=True, \
                                    num_workers=self.config['num_workers'], collate_fn=collate_fn)
        
        # Define model
        self.model = PredictionModel(config).cuda()

        # Define optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['optim_args']['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['optim_args']['scheduler_step'], \
                                                            gamma=config['optim_args']['scheduler_gamma'])

        # Define losses
        self.losses = [initialize_metric(config['losses'][i]) for i in range(len(config['losses']))]
        self.loss_weights = self.config['loss_weights']

        # Define evaluation metrics
        self.metrics = EvaluationMetrics()

        # Define training stats logger
        self.writer = writer
    

    def train(self, scheduler):
        """
        Main function to train model
        """
        # T0 = int(self.config['total_epoch'] / 5)
        # eta_max = self.config['optim_args']['eta_max']
        for epoch in range(self.config['current_epoch'], self.config['total_epoch']):
            # Optimizer
            # if scheduler:
            #   if epoch <= T0:
            #     eta = 0.0001 + (epoch / T0) * eta_max
            #   else:
            #     eta = eta_max * np.cos((np.pi / 2) * (epoch - T0) / (self.config['total_epoch'] - T0)) + 0.000001
            #   for op_params in self.optimizer.param_groups:
            #     op_params['lr'] = eta

            # Train
            train_epoch_metrics, loss_dict = self.run_epoch('train', self.train_loader)

            # Log train loss
            for key, value in loss_dict.items():
                self.writer.add_scalar(key, value)
            print('Epoch: [', epoch + 1, '/', self.config['total_epoch'], '], Train Losses: ', loss_dict)
            print('Train Epoch Metrics: ', train_epoch_metrics)

            # TODO: val
            # val_metrics, val_loss = self.run_epoch('val', self.val_loader)
            # self.writer.add_scalar("validation loss", val_loss)
            # print('Epoch: [', epoch, '/', self.config['total_epoch'], '], ValidationLoss: ', val_loss)

            self.writer.flush()


    def run_epoch(self, mode, dataloader):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dataloader: dataloader object
        :return metrics: performance metric for last batch in this epoch
        :return loss: overall loss values
        :return loss_values: list consist of each seperate loss
        """
        # if mode == 'val':
        #     self.model.eval()
        # else:
        #     self.model.train()
        for i, data in tqdm(enumerate(dataloader)):
            # Model prediction
            prediction = self.model(data)

            # Loss and backprop
            total_loss, loss_dict = self.compute_loss(mode, prediction, data)
            self.back_prop(total_loss)

            # Evaluation metrics
            metrics = self.metrics.compute(prediction, data)
        self.scheduler.step()
        return metrics, loss_dict


    def compute_loss(self, mode, model_outputs, data):
        """
        Computes loss given model outputs and ground truth labels
        """

        loss_values = [loss.compute(model_outputs, data) for loss in self.losses] # list of list
        total_loss = torch.as_tensor(0, device='cuda').float()
        for n in range(len(loss_values)):
            total_loss += self.loss_weights[n] * loss_values[n][0]  # loss_values[n][0] is the total loss for each loss module, loss_values[n][1..] could be seperate smaller loss
      
        # reformulate loss_values
        loss_dict = {mode+'_loss':total_loss.item(),
                      mode+'_lane_select_loss':loss_values[0][0].item(), 
                      mode+'_position_loss':loss_values[1][1].item(),
                      mode+'_lane_off_loss':loss_values[1][2].item(),
                      mode+'_modality_select_loss':loss_values[2][0].item()}
        return total_loss, loss_dict

    
    def back_prop(self, loss, grad_clip_thresh=10):
        """
        Backpropagate loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
        self.optimizer.step()