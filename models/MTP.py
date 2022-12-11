import torch
import torch.nn as nn
from torch.nn.modules import BatchNorm1d
from models.general import Linear, LinearRes
from scripts.utils import gpu, to_long

class MTP(nn.Module):
    """
    Multi-modal Trajectory Prediction Block

    Paper: https://arxiv.org/abs/1809.10732
    Code Refer to MTP implementation in https://github.com/bdokim/LaPred/blob/master/Lapred_original.py
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config

        # dimensions from yml
        agent_enc_dim = config['agent_enc_dim']
        mtp_dim = config['mtp_dim']
        la_dim = config['tde_merger_dim']

        # multi-modal trajectory generators, one FC block for each modality
        layers = []
        for _ in range(config['k_mod']):
            layers.append(
                nn.Sequential(
                  nn.Linear(agent_enc_dim + la_dim, mtp_dim),
                  nn.ReLU(inplace = True),
                  LinearRes(mtp_dim, mtp_dim, 'BN'),
                  nn.Linear(mtp_dim, 2 * 2 * config['pred_size'])
                )
            )
        self.layers = nn.ModuleList(layers)

        # layers predict which modality matches the ground truth best
        self.attDest = AttDest(config)
        self.clsNet = nn.Sequential(
            LinearRes(mtp_dim, mtp_dim, 'BN'),
            nn.Linear(mtp_dim, 1)
        )


    def forward(self, inputs, target_feats, target_past, att_feats):
        """
        :param inputs:
          :subparam agent_ctrs: current location of all the agents in each scene
                              shape: (batch_size, n_agent, 2)
        :param target_feats: encoded features of the target agent, from TFE blocks
                              shape: (batch_size, agent_enc_dim)
        :param target_past: past trajectory of target agents
                              shape: (batch_size, past_t*2+1, 6), past_t*2+1 = 5 in our setting
        :param att_feats: weighted environmental feature from LA
                              shape: (batch_size, tde_merger_dim) 
        :return out[pred_trajs]: predicted global trajectories
                              shape: (batch_size, k_mod, 2*config['pred_size'], 2)
        :return out[cls]: likelihood of each modality (relevance between modality and GT)
                              shape: (batch_size, k_mod)
        """
        agent_ctrs, rot, orig = gpu(inputs['ctrs']), gpu(inputs['rot']), gpu(inputs['orig'])
        integrated_feats = torch.cat([target_feats, att_feats], 1)  # (batch_size, agent_enc_dim + la_dim)

        pred_trajs = []
        predictions = []
        # predict one modality from each generator
        for head in self.layers:
            pred_trajs.append(head(integrated_feats))   # (batch_size, 4*config['pred_size'])
            predictions.append(head(integrated_feats))  # (batch_size, 4*config['pred_size'])

        # predicted trajectory, k modality in total, for each modality, predict 2*pred_size steps, each step is (x, y)
        pred_trajs = torch.cat([traj.unsqueeze(1) for traj in pred_trajs], 1)       # (batch_size, k_mod, 4*config['pred_size'])
        pred_trajs = pred_trajs.view(pred_trajs.size(0), pred_trajs.size(1), -1, 2) # (batch_size, k_mod, 2*config['pred_size'], 2)
        
        # concatenate past trajectory and predicted trajectory
        cls = torch.cat([pred.unsqueeze(1) for pred in predictions], 1) # (batch_size, 4*pred_size, k_mod)
        cls = cls.view(-1, 2 * 2 * self.config['pred_size'])            # (batch_size*k_mod, 4*pred_size)
        target_past = target_past[:, :, :2].repeat((self.config['k_mod'], 1, 1))  # (batch_size*k_mod, 5, 2)
        target_past = target_past.view(cls.size(0), -1)                           # (batch_size*k_mod, 10)
        cls_feats = torch.cat((target_past, cls), 1)                              # (batch_size*k_mod, 34)

        # get current location of targets and destination locations
        target_ctrs = torch.cat([ctr[0:1] for ctr in agent_ctrs], 0)              # (batch_size, 2)                
        dest_ctrs = pred_trajs[:, :, -1].detach()                                 # (batch_size, k_mod, 2), destination of predicted trajectory
        
        # neural network, which predicts the relevance of each modality to the ground truth
        cls_inputs = self.attDest(integrated_feats, cls_feats, target_ctrs, dest_ctrs)  # (batch_size*k_mod, mtp_dim)
        cls_outputs = self.clsNet(cls_inputs).view(-1, self.config['k_mod'])            # (batch_size, k_mod)
        
        # sort predict trajectories, by the predicted relevant factor cls_outputs
        cls_outputs, sort_idx = cls_outputs.sort(1, descending = True)  # (batch_size, k_mod), (batch_size, k_mod)
        row_idx = torch.arange(len(sort_idx)).long()
        row_idx = row_idx.to(sort_idx.device).view(-1, 1).repeat(1, sort_idx.size(1)).view(-1)
        sort_idx = sort_idx.view(-1)
        pred_trajs = pred_trajs[row_idx, sort_idx].view(cls_outputs.size(0), cls_outputs.size(1), -1, 2)  # (batch_size, k_mod, 2*config['pred_size'], 2)

        # turn local trajectory to global coordinate
        outputs = {}
        outputs['cls'] = cls_outputs
        outputs['pred_trajs'] = []
        for i in range(len(pred_trajs)):
            pred_traj = torch.matmul(pred_trajs[i], rot[i]) + orig[i].view(1, 1, -1)    # rotate from local coord to global coord
            outputs['pred_trajs'].append(pred_traj)
        outputs['pred_trajs'] = torch.cat([traj.unsqueeze(0) for traj in outputs['pred_trajs']], 0)
        return outputs


class AttDest(nn.Module):
    """
    Implementation largely based on https://github.com/bdokim/LaPred/blob/master/Lapred_original.py
    """
    def __init__(self, config):
        super().__init__()

        agent_enc_dim = config['agent_enc_dim']
        mtp_dim = config['mtp_dim']
        la_dim = config['tde_merger_dim']

        # encode full trajectories
        self.clsNet = nn.Sequential(
            LinearRes(4 * config['pred_size'] + 4 * config['train_size'] + 2, mtp_dim, 'BN'),
            Linear(mtp_dim, mtp_dim, 'BN')
        )

        # encode distance between current location and destination
        self.dist = nn.Sequential(
            nn.Linear(2, mtp_dim),
            nn.ReLU(inplace = True),
            Linear(mtp_dim, mtp_dim, 'BN')
        )

        # aggregate full traj encoding, distance encoding, environmental encoding
        input_dim = 2 * mtp_dim + agent_enc_dim + la_dim
        self.agt = Linear(input_dim, mtp_dim, 'BN')
      

    def forward(self, integrated_feats, cls_feats, target_ctrs, dest_ctrs):
        """
        :param integrated_feats: encoded target agent feature + environmental feature from LA
                              shape: (batch_size, agent_enc_dim + la_dim)
        :param cls_feats: past_trajectory of target agent + predicted future trajectory
                              shape: (batch_size*k_mod, 34)
        :param target_ctrs: current location of target agent
                              shape: (batch_size, 2)
        :param dest_ctrs: destination location of predicted trajectory
                              shape: (batch_size, k_mod, 2)
        :return agt_output: aggregator output, TODO??????
                              shape: (batch_size*k_mod, mtp_dim)
        """
        integrated_feats = integrated_feats.unsqueeze(1).repeat(1, dest_ctrs.size(1), 1)  
        integrated_feats = integrated_feats.view(-1, integrated_feats.size(-1))  # (batch_size*k_mod, agent_enc_dim + la_dim)
        
        cls_feats = self.clsNet(cls_feats)  # (batch_size*k_mod, mtp_dim)

        dist_input = (target_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2) # (batch_size*kmod, 2)
        dist_output = self.dist(dist_input)                             # (batch_size*kmod, mtp_dim)

        agt_input = torch.cat((dist_output, integrated_feats, cls_feats), 1)
        agt_output = self.agt(agt_input)
        return agt_output
