import torch
import torch.nn as nn

from models.general import Linear

class LaneAttention(nn.Module):
    """
    Lane Attention Block
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config
        
        self.n_in = config['tde_merger_dim']
        n_lane_att = config['lane_att_dim']
        self.n_lane = config["lane"]

        # TODO: why this structure
        self.attention = nn.Sequential(nn.Linear(self.n_in * self.n_lane, n_lane_att), \
                                      nn.ReLU(inplace=True), \
                                      Linear(n_lane_att, n_lane_att, norm='BN', act=False), \
                                      nn.Linear(n_lane_att, self.n_lane))

        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        """
        Forward pass for Lane Attention block
        :param inputs: lane-trajectory feature extracted from previous block, (batch_size*num_lane, tde_merger_dim)
        :return att_weights: attention weights, (batch_size, num_lane)
        :return att_feats: weighted environmental feature, (batch_size, tde_merger_dim)
        """
        feats = inputs.view(-1, self.n_lane, self.n_in).flatten(1, 2)  # (batch_size, num_lane*tde_merger_dim)
        
        # get attention coefficients
        att_weights = self.attention(feats)     # (batch_size, num_lane), however this is a logit
        att_weights = self.softmax(att_weights) # (batch_size, num_lane), this is a valid probability

        # weight the original features by attention coeff
        att_feats = att_weights.view(-1, 1) * inputs            # (batch_size*num_lane, tde_merger_dim)
        att_feats = att_feats.view(-1, self.n_lane, self.n_in)  # (batch_size, num_lane, tde_merger_dim)
        att_feats = att_feats.sum(1)                            # (batch_size, tde_merger_dim)

        return att_weights, att_feats

