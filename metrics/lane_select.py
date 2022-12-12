from metrics.metric import Metric
import torch
import torch.nn as nn


class LaneSelectionLoss(Metric):
    """
    Lane selection self-supervised loss, which helps the model to put more
    weight on ground truth reference path during the Lane-Attention block
    """
    def __init__(self, config):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def compute(self, model_outputs, data):
        """
        Compute self-supervised loss with CrossEntropy criterion
        :param model_outputs: a dict includes the full model prediction
        :param data: raw data contains ground truth infos
        :return: computed loss
        """
        loss = 0

        map_info = data['map_info']
        att_weights = model_outputs['att_weights']  # (batch_size, num_lane)
        for i in range(len(map_info)):
            gt_idx = map_info[i]['label']
            if gt_idx == 90: continue   # refer to nuScenes.py, lane_label=90 indicates invalid reference lane
            if self.config['weighted_lane_select']: # give higher weight to GT lanes that are far away
                loss += self.criterion(att_weights[i], torch.tensor(gt_idx).to('cuda')) * (gt_idx + 1)
            else:
                loss += self.criterion(att_weights[i], torch.tensor(gt_idx).to('cuda'))
        return [loss / len(att_weights)]  # need to average here because batch-wise loss are added during the for loss
