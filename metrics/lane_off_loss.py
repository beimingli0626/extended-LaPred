import torch
from scripts.utils import gpu


class LaneOffLoss():
    """
    Lane Off Loss, which reflects the tendency of the target agent stick close to the reference lane in the future,
    this encourages the model to reduce the distance from the lane whenever the prediction deviates from a lane 
    farther than the ground truth.
    Calculated in pred_loss
    
    Refer to original LAPred Paper for details: https://arxiv.org/abs/2104.00249
    """
    def __init__(self, config):
        self.config = config

    def compute(self, data, mask, has_gt_preds, gt_preds, preds):
        """
        Compute lane-off loss
        :param data: batches from dataset
        :param mask: if there exists GT for that batch, (batch_size)
        :param has_gt_preds: indicator of whether has future trajectory at certain timestamp, (_batch_size_, 2*pred_size)
        :param gt_preds: ground truth future trajectory, (_batch_size_, 2*pred_size, 2)
        :param preds: selected predicted trajectory, (_batch_size_, 2*pred_size, 2)
        :return loss: computed lane-off loss, reformed to a list [laneoff_loss] for uni-format across losses
        """
        rot, orig = gpu(data["rot"]), gpu(data['orig'])
        map_info = gpu(data['map_info'])

        loss = 0
        count = 0
        for i in range(len(map_info)):
            gt_idx = map_info[i]['label']
            if gt_idx != 90 and mask[i]:  # refer to nuScenes, lane_label=90 indicates invalid reference lane
                # get discretized points of reference lane
                lane_feats = map_info[i]['lane_feats'].to(torch.float32)        # (num_lane, num_points, 2)
                ref_lane = lane_feats[gt_idx]                                   # (num_points, 2), discretized points of reference lane
                ref_lane = torch.matmul(ref_lane, rot[i]) + orig[i].view(1, -1) # local coord to global coord

                # calculate minimum distance between predicted/GT trajectories w.r.t reference lane
                filtered_idx = mask[:i+1].sum() - 1 # get the index in the filtered index, filter means batch_size -> _batch_size_ 
                pred = preds[filtered_idx]          # (2*pred_size, 2)
                gt = gt_preds[filtered_idx]         # (2*pred_size, 2)
                delta_pred = torch.norm(ref_lane.unsqueeze(0) - pred.unsqueeze(1),dim=-1).min(-1)[0] # (2*pred_size, num_points, 2) -norm-> (2*pred_size, num_points) -min-> (2*pred_size), minimum distance between predicted trajectory and reference lane
                delta_gt = torch.norm(ref_lane.unsqueeze(0) - gt.unsqueeze(1), dim=-1).min(-1)[0]    # (2*pred_size, num_points, 2) -norm-> (2*pred_size, num_points) -min-> (2*pred_size), minimum distance between ground truth point and reference lane

                # get lane-off loss
                indicator = torch.ge(delta_pred, delta_gt).type(torch.float)  # if distance from predicted point to ref lane is larger than GT point, (12)
                diff = delta_pred - delta_gt                                  # larger by how much, (12)
                loss += (diff*indicator)[has_gt_preds[i]].mean()              # take the average over points that have future GT
                count += 1
        return loss / count                                                   # take average over batches