import torch
import torch.nn as nn
from models.TFE import TFE
from models.LaneAttention import LaneAttention
from models.MTP import MTP


class PredictionModel(nn.Module):
    """
    Single-agent prediction model (LaPred)
    """
    def __init__(self, config):
        """
        Initializes model for single-agent trajectory prediction
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config
        
        self.TFE = TFE(config)
        self.LA = LaneAttention(config)
        self.MTP = MTP(config)


    def forward(self, inputs):
        """
        Forward pass for prediction model
        :param inputs: Dictionary with preprocessed data
        :return out: K Predicted trajectories and/or their probabilities
        """
        TFEfeatures, targets_encoding, targets = self.TFE(inputs)
        att_weights, att_feats = self.LA(TFEfeatures)
        out = self.MTP(inputs, targets_encoding, targets, att_feats)

        out['att_weights'] = att_weights
        return out