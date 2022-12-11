import torch.nn as nn
from scripts.utils import gpu

class PositionLoss():
    """
    Position Loss, which is the L1 loss between the best modality and ground truth trajectory
    """
    def __init__(self):
        # use default beta = 1.0
        self.criterion = nn.SmoothL1Loss(reduction="sum")

    def compute(self, has_gt_preds, gt_preds, preds):
        """
        Compute position loss
        :param has_gt_preds: indicator of whether has future trajectory at certain timestamp, (_batch_size_, 2*pred_size)
        :param gt_preds: ground truth future trajectory, (_batch_size_, 2*pred_size, 2)
        :param preds: selected predicted trajectory, (_batch_size_, 2*pred_size, 2)
        :return loss: computed position loss
        """    
        loss = self.criterion(preds[has_gt_preds], gt_preds[has_gt_preds])
        return loss / has_gt_preds.sum().item()    # loss averaged by number of points that have ground truth prediction