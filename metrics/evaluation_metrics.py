from metrics.metric import Metric
import torch
import numpy as np


class EvaluationMetrics(Metric):
    """
    Evaluation metrics, including:
        - MinADE_1
        - MinADE_5
        - MinADE_10
        - MinFDE_1
        - MinFDE_5
        - MinFDE_10
    MinADE: The average of pointwise L2 distances between the predicted trajectory and ground truth over the k most likely predictions.
    MinFDE: L2 distance between the final points of the prediction and ground truth. We take the minimum FDE over the k most likely predictions and average over all agents.
    """
    def __init__(self):
        pass

    def compute(self, model_outputs, data):
        """
        Compute self-supervised loss with CrossEntropy criterion
        :param model_outputs: a dict includes the full model prediction
        :param data: raw data contains ground truth infos
        :return: computed ADE and FDE loss
        """
        gt_preds = torch.cat([x[0:1] for x in data['gt_preds']], 0)         # (batch_size, 2*pred_size, 2), ground truth in global coordinate
        preds = model_outputs['pred_trajs'].detach().cpu().numpy()          # (batch_size, k_mod, 2*pred_size, 2)

        gt_preds = np.asarray(gt_preds, np.float32)
        preds = np.asarray(preds, np.float32)

        err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(-1)) # L2 distances between predicted trajectory and ground truth, (batch_size, k_mod, 2*pred_size)
        ade_1 = err[:, 0].mean()                                            # Average pointwise L2 distances accross batch and points. 
                                                                            # Note that we sort the modality by likelihood in MTP, err[:, 0] corresponds to most likely modality
        fde_1 = err[:, 0, -1].mean()                                        # Average distances between final points of prediction and ground truth

        row_idcs = np.arange(preds.shape[0]).astype(np.int64)               # (batch_size)
        ade_min_idcs_5 = err[:, :5].mean(2).argmin(1)                       # (batch_size), among 5 modality, which one has smallest L2 pointwise distances
        fde_min_idcs_5 = err[:, :5, -1].argmin(1)                           # (batch_size), among 5 modality, which one has closest distances at final point
        ade_5 = err[:, :5][row_idcs, ade_min_idcs_5].mean()                   
        fde_5 = err[:, :5][row_idcs, fde_min_idcs_5][:, -1].mean()          # (batch_size, 2*pred_size) -> (brach_size, 1) -> (1)

        ade_min_idcs_10 = err[:, :].mean(2).argmin(1)                       # (batch_size), among 10 modality, which one has smallest L2 pointwise distances, note that we set k_mod to be 10
        fde_min_idcs_10 = err[:, :, -1].argmin(1)                           # (batch_size), among 10 modality, which one has closest distances at final point
        ade_10 = err[row_idcs, ade_min_idcs_10].mean()
        fde_10 = err[row_idcs, fde_min_idcs_10][:, -1].mean()

        return {'ade_1':ade_1, 'fde_1':fde_1, 'ade_5':ade_5, 'fde_5':fde_5, 'ade_10':ade_10, 'fde_10':fde_10}
