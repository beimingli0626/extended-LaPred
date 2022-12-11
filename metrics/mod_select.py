from metrics.metric import Metric
import torch
import torch.nn as nn
from scripts.utils import gpu


class ModSelectionLoss(Metric):
    """
    Modality selection loss, which helps the model to rank the modality
    by likelihood correctly, thus help ADE and FDE metrics
    """
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def compute(self, model_outputs, data):
        """
        Compute self-supervised loss with CrossEntropy criterion
        :param model_outputs: a dict includes the full model prediction
        :param data: raw data contains ground truth infos
        :return: computed loss, reformed to a list [mod_selection_loss] for uni-format across losses
        """
        gt_preds = torch.cat([x[0:1] for x in gpu(data['gt_preds'])], 0)    # (batch_size, 2*pred_size, 2), ground truth in global coordinate
        preds = model_outputs['pred_trajs']                                 # (batch_size, k_mod, 2*pred_size, 2)
        cls = model_outputs['cls']                                          # (batch_size, k_mod)

        # (batch_size, k_mod, 2*pred_size, 2) -> (batch_size, k_mod) -> (batch_size), which modality is closest to the ground truth
        _, cls_tar = (torch.square(preds - gt_preds.unsqueeze(1)).sum(dim=(-2, -1))).min(-1)

        loss = self.criterion(cls, cls_tar)
        return [loss]


