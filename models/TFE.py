import torch
import torch.nn as nn

from models.general import Linear, Conv1d
from scripts.utils import gpu, to_long


class TFE(nn.Module):
    """
    Trajectory-Lane Feature Extraction Block
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config
        self.target_encoder = AgentEncoder(config)
        self.agents_encoder = AgentEncoder(config)
        self.lanes_encoder = LaneEncoder(config)
        self.merger = FeatureMerger(config)


    def forward(self, inputs):
        """
        Forward pass for TFE block
        :param inputs: full data
        """
        # get features of target agent, lane and surrounding agent feature
        targets = self.get_targets(gpu(inputs['feats']))      # (batch_size, past_t*2+1, 6)
        lane_feats, nearby_agents = self.get_lanes(to_long(gpu(inputs["map_info"])))

        # encode target agent / nearby agents / lane feature
        targets_encoding = self.target_encoder(targets)       # (batch_size, encoder_dim)
        agents_encoding = self.agents_encoder(nearby_agents)  # (batch_size*num_lane, encoder_dim)
        lanes_encoding = self.lanes_encoder(lane_feats)       # (batch_size*num_lane, encoder_dim)

        # merge encodings and get so called trajectory-lane features
        features = self.merger(targets_encoding, agents_encoding, lanes_encoding) # (batch_size*num_lane, merger_dim)

        return features, targets_encoding, targets

    
    def get_targets(self, inputs):
        """
        Concatenate target agent features
        :param inputs: data['feats'], which is a list of tensors. 
                       shape: [batch_size * (n, past_t*2+1, 6)]
        :return targets: features of each target agent, (batch_size, past_t*2+1, 6)
        """
        # note that the first agent of the 'feats' are always the target agent
        targets = [feats[[0]] for feats in inputs]
        targets = torch.cat(targets, 0)
        return targets 

    
    def get_lanes(self, map_info):
        """
        :param inputs: data['map_info'], which is a list of dictionary
                       shape: [batch_size * dict()]
        :return lane_feats: poses of nearby lanes, (batch_size*num_lane, num_points, 2)
        :return nearby_agents: features of closest agent for each lane (batch_size*num_lane, past_t*2+1, 6) / (batch_size*num_lane, 3*(past_t*2+1), 6)
        """
        lane_feats = [x['lane_feats'].to(torch.float32) for x in map_info]
        lane_feats = torch.cat(lane_feats, 0)

        batch_size = len(map_info)
        num_lane = self.config['lane']
        nearby_agents = torch.zeros((batch_size, self.config['lane'], self.config['train_size']*2+1, 6)).cuda()
        for i in range(batch_size): 
            trajs = map_info[i]['nearby_trajs']       # trajs: (num_lane, num_nearby_vehicle, (past_t * 2 + 1, 6))
            for j in range(len(trajs)):               # iterate through each nearby lane
                if len(trajs[j]) > 0:                 # if has at least one nearby vehicle
                    nearby_agents[i, j] = trajs[j][0] # select the closest agent
                    # TODO: do we need to consider 'if len(x[j][0].shape) < 2'
        nearby_agents = nearby_agents.flatten(0, 1)   # (batch_size*num_lane, past_t*2+1, 6)


        # NOTE: try concatenate each lane with nearby two lanes information, the same for nearby agent
        # lanes and agents at left and right of current lane
        # lane_feats_left = torch.zeros_like(lane_feats)
        nearby_agents_left = torch.zeros_like(nearby_agents)
        # lane_feats_right = torch.zeros_like(lane_feats)
        nearby_agents_right = torch.zeros_like(nearby_agents)
        for i in range(batch_size):
            for j in range(num_lane):
                lane_idx = i*num_lane + j
                if j >= 1:
                    # lane_feats_left[lane_idx] = lane_feats[lane_idx-1]
                    nearby_agents_left[lane_idx] = nearby_agents[lane_idx-1]
                if j < num_lane - 1:
                    # lane_feats_right[lane_idx] = lane_feats[lane_idx+1]
                    nearby_agents_right[lane_idx] = nearby_agents[lane_idx+1]
                    
        # concatenate the information of current lane with nearby two lanes
        # NOTE: exceeds GPU memory if concatenate lane features
        # lane_feats = torch.cat([lane_feats_left, lane_feats, lane_feats_right], dim=1)              # (batch_size*num_lane, 3*num_points, 2)
        nearby_agents = torch.cat([nearby_agents_left, nearby_agents, nearby_agents_right], dim=1)  # (batch_size*num_lane, 3*(past_t*2+1), 6)
                
        return lane_feats, nearby_agents

  
class AgentEncoder(nn.Module):
    """
    CNN+LSTM Encoder for encoding agent features
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config 

        n_in = 6  # each agent has 6 raw features at each timestep
        n_out = config['agent_enc_dim']

        # TODO: why 2, 0, 1? why not 3, 1, 1
        self.conv = nn.Sequential(Conv1d(n_in, n_out, 2, 0, 1, 'BN'), \
                                  Conv1d(n_out, n_out, 2, 0, 1, 'BN'))
        self.lstm = nn.LSTM(n_out, n_out, num_layers=1, batch_first=True)


    def forward(self, inputs):
        """
        Forward pass for Encoder
        :param inputs: features of the target agent, (batch_size, past_t*2+1, 6) / (batch_size*num_lane, past_t*2+1, 6) / (batch_size*num_lane, 3*(past_t*2+1), 6)
        :return: agent features encoded by CNN+LSTM, (batch_size, H_out) / (batch_size*num_lane, past_t*2+1, 6) / (batch_size*num_lane, 3*(past_t*2+1), 6)

        Note: 'transpose' func returns a new tensor with same data
              batch_size*num_lane for surrounding agents, batch_size for target agent
        """
        # 1D CNN
        out = self.conv(inputs.transpose(1, 2)) #(batch_size, 6, 5)->(batch_size, 128, 3) / (batch_size, 6, 15)->(batch_size, 128, 13)

        # LSTM
        out = out.transpose(1, 2)   # (batch_size, 3, 128), where 3 acts as sequence length / (batch_size, 13, 128)
        out, _ = self.lstm(out)     # (batch_size, L, H_in) -> (batch_size, L, H_out)
        out = out[:, -1]            # (batch_size, H_out), only keep projection from final timestep 
        return out


class LaneEncoder(nn.Module):
    """
    CNN+LSTM Encoder for encoding lane features
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config

        n_in = 2  # each lane pose has 2 features
        n_out = config['lane_enc_dim']

        # TODO: why these CNN params
        self.conv = nn.Sequential(Conv1d(n_in, n_out, 3, 1, 1, 'BN'), \
                                  Conv1d(n_out, n_out, 3, 1, 1, 'BN'), \
                                  Conv1d(n_out, n_out, 3, 1, 1, 'BN'), \
                                  Conv1d(n_out, n_out, 3, 1, 1, 'BN'))
        self.lstm = nn.LSTM(n_out, n_out, num_layers=1, batch_first=True)


    def forward(self, inputs):
        """
        Forward pass for Encoder
        :param inputs: features of nearby lanes, (batch_size*num_lane, num_points, 2)
        :return: lane features encoded by CNN+LSTM, (batch_size*num_lane, H_out)
        """
        # 1D CNN
        out = self.conv(inputs.transpose(1, 2)) #(batch_size*num_lane, 2, num_points)->(batch_size*num_lane, 128, num_points)

        # LSTM
        out = out.transpose(1, 2)   # (batch_size*num_lane, num_points, 128), where num_points acts as sequence length
        out, _ = self.lstm(out)     # (batch_size*num_lane, L, H_in) -> (batch_size_num_lane, L, H_out)
        out = out[:, -1]            # (batch_size*num_lane, H_out), only keep projection from final timestep 
        return out


class FeatureMerger(nn.Module):
    """
    Feature Merger for merging target/nearby agents/lane encodings
    """
    def __init__(self, config):
        """
        Initialization
        :param config: configuration from yml file 
        """
        super().__init__()
        self.config = config

        n_in = config['agent_enc_dim'] * 2 + config['lane_enc_dim']
        n_out = config['tde_merger_dim']

        # TODO: why these dimensions
        self.merge = nn.Sequential(
            nn.Linear(n_in, n_out), \
            nn.ReLU(inplace=True), \
            Linear(n_out, n_out, norm='BN', act=False), \
            nn.Linear(n_out, n_out), \
            nn.ReLU(inplace=True), \
            Linear(n_out, n_out, norm='BN', act=False))


    def forward(self, targets, agents, lanes):
        """
        Forward pass for Feature Merger
        :param targets: targets encoding, (batch_size, encoder_dim)
        :param agents: agents encoding, (batch_size*num_lane, encoder_dim)
        :param lanes: lanes encoding, (batch_size*num_lane, encoder_dim)
        :return: merged feature, (batch_size*num_lane, merger_dim)
        """
        targets_rep = targets.repeat([self.config["lane"], 1])  # (batch_size*num_lane, encoder_dim)
        out = torch.cat([targets_rep, lanes, agents], 1)        # (batch_size*num_lane, encoder_dim*3)
        out = self.merge(out)
        return out


