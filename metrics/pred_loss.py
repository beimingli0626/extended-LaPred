from metrics.metric import Metric
import torch
import torch.nn as nn
from scripts.utils import gpu
from metrics.position_loss import PositionLoss
from metrics.lane_off_loss import LaneOffLoss

class PredictionLoss(Metric):
    """
    Prediction Loss, which composed of position loss and lane-off loss
    """
    def __init__(self):
        self.positionloss = PositionLoss()
        self.laneoffloss = LaneOffLoss()

    def compute(self, model_outputs, data):
        """
        Compute prediction loss
        :param model_outputs: a dict includes the full model prediction
        :param data: raw data contains ground truth infos
        :return loss: computed loss, a list [loss, position_loss, laneoff_loss]
        """
        has_gt_preds = torch.cat([x[0:1] for x in gpu(data['has_preds'])], 0)  # (batch_size, 2*pred_size)
        gt_preds = torch.cat([x[0:1] for x in gpu(data['gt_preds'])], 0)    # (batch_size, 2*pred_size, 2), ground truth in global coordinate
        preds = model_outputs['pred_trajs']                                 # (batch_size, k_mod, 2*pred_size, 2)

        # find the last future timestamp that have ground truth future trajectories
        last = has_gt_preds.float() + (torch.arange(1, has_gt_preds.shape[-1]+1).float().to(has_gt_preds.device)) / 100 # divided by 100 to make sure number in range is smaller than 1
        max_last, last_idx = last.max(1)    # max would be the last timestamp if has_gt_preds are all 1, (batch_size), (batch_size)
        mask = max_last > 1.0               # if there exists GT for that batch

        # only keep the batch data when the target vehicle do have future trajectory in that scene
        preds = preds[mask]                 
        gt_preds = gt_preds[mask]           # ground truth that has future traj, (_batch_size_, 2*pred_size, 2)
        has_gt_preds = has_gt_preds[mask]   # (_batch_size_, 2*pred_size)
        last_idx = last_idx[mask]

        # calculate the distance between the destination of trajectory predicted by each modality and last ground truth point
        row_idx = torch.arange(len(last_idx)).long().to(last_idx.device) # (_batch_size_)
        dist = []
        for i in range(preds.shape[1]):  # iterate through k_mod
            # preds[row_idcs, i, last_idcs]: (_batch_size_, 2)
            # gt_preds[row_idcs, last_idcs]: (_batch_size_, 2)
            dist.append(torch.sqrt(((preds[row_idx, i, last_idx] - gt_preds[row_idx, last_idx])**2).sum(1))) # each iteration, get (_batch_size_)
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1) # (_batch_size_, k_mod)

        # for each batch, keep the modality, whose predicted destination is closest to the GT destination
        _, min_idx = dist.min(1)                            # (_batch_size_)
        preds = preds[row_idx, min_idx]                     # (_barch_size_, 2*pred_size, 2)

        # calculate two losses
        position_loss = self.positionloss.compute(has_gt_preds, gt_preds, preds)
        laneoff_loss = self.laneoffloss.compute(data, mask, has_gt_preds, gt_preds, preds)
        return [position_loss + laneoff_loss, position_loss, laneoff_loss]
