from metrics.metric import Metric
import torch
import torch.nn as nn
from scripts.utils import gpu
from scipy import stats


class ModSelectionLoss(Metric):
    """
    Modality selection loss, which helps the model to rank the modality
    by likelihood correctly, thus help ADE and FDE metrics
    """
    def __init__(self, config):
        self.config = config

        if self.config['nn_mod_select']:
            self.criterion = nn.CrossEntropyLoss()
        elif not self.config['nn_mod_select']:
            self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)


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

        if self.config['nn_mod_select']:
            # (batch_size, k_mod, 2*pred_size, 2) -> (batch_size, k_mod) -> (batch_size), which modality is closest to the ground truth
            _, cls_tar = (torch.square(preds - gt_preds.unsqueeze(1)).sum(dim=(-2, -1))).min(-1)    # (batch_size)
            loss = self.criterion(cls, cls_tar)
        elif not self.config['nn_mod_select']:
            # compare the difference between two rankings with cosine similarity
            _, cls_tar = (torch.square(preds - gt_preds.unsqueeze(1)).sum(dim=(-2, -1))).sort(-1)   # (batch_size, k_mod)
            loss = self.criterion(cls.float(), cls_tar.float()).mean()
        return [loss]


