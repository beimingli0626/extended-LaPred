######## Derived Dataset class for nuScenes Dataset #########
# Author: Beiming Li
# Date: 10/12/22
# Description: 
# Define a customized Dataset class which acts as dataloader 
# as well as data preprocessor. We preprocess the data first
# to save time during run time.
#
# This code is largely based on the following nuScenes preprocess 
# pipeline with annotation and modification:
# https://github.com/bdokim/LaPred/blob/master/data_process.py

import numpy as np
import torch
from torch.utils.data import Dataset

import os
import copy
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion import arcline_path_utils

from scripts.utils import ref_copy, from_numpy

class nuScenesDataset(Dataset):
    def __init__(self, split, config):
        self.config = config
        
        # use preprocessed data for train/val/test
        if 'preprocess' in config and config['preprocess']:
          if split == 'train':
            self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
          elif split == 'val' :
            self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
          elif split == 'test' :
            self.split = np.load(self.config['preprocess_test'], allow_pickle=True)
        # perform data preprocessing
        else:
          self.ns = NuScenes(self.config['dataset'], dataroot=self.config['data_path'])
          self.helper = PredictHelper(self.ns)
          self.token_list = get_prediction_challenge_split(split, dataroot=self.config['data_path'])
          self.maps = load_all_maps(self.helper)


    # def debug(self,idx) :
    #   import matplotlib.pyplot as plt
    #   instance_token, sample_token = self.token_list[idx].split("_")
    #   map_name = self.helper.get_map_name_from_sample_token(sample_token)
    #   self.map_api = self.maps[map_name]

    #   data = self.get_agent_feats(instance_token, sample_token)
    #   data['map_info'] = self.get_lane_feats(data)

    #   data['idx'] = idx
    #   return data


    def __getitem__(self, idx):
      # use preprocessed data for train/val/test
      if 'preprocess' in self.config and self.config['preprocess']:
        data = self.split[idx]

        new_data = dict()
        for key in ['orig', 'ctrs','gt_preds', 'has_preds', 'theta', 'rot', 'feats', "grid_ctrs", "grid_feats", 'grid_xy', 'ins_sam', 'map_info']:
            if key in data:
                new_data[key] = ref_copy(data[key])
        data = new_data

        return data

      # perform data preprocessing
      instance_token, sample_token = self.token_list[idx].split("_")
      map_name = self.helper.get_map_name_from_sample_token(sample_token)
      self.map_api = self.maps[map_name]

      data = self.get_agent_feats(instance_token, sample_token)
      data['map_info'] = self.get_lane_feats(data)
      data['idx'] = idx
      return data
    

    def __len__(self):
      if 'preprocess' in self.config and self.config['preprocess']:
        return len(self.split)
      else:
        return len(self.token_list)


    def get_agent_feats(self, instance_token, sample_token):
      # return global x, y position for past 2 seconds
      past_traj = self.helper.get_past_for_agent(instance_token, sample_token, \
          seconds=self.config['train_size'], in_agent_frame=False)

      past_traj = np.asarray(past_traj, dtype=np.float32)

      # orig: current global x, y position, therefore call it orig(in)
      cur_traj = self.helper.get_sample_annotation(instance_token, \
          sample_token)["translation"][:2]
      orig = np.asarray(cur_traj, dtype=np.float32)

      # last position - current position
      prev = past_traj[0] - orig

      # Note: Here, we use the last position - current position to estimate the yaw of the vehicle at current timestep. But we could use quaternion_yaw(Quaternion(sample_annotation['rotation'])) instead
      # TODO: shouldn't we use np.pi / 2 ????? But maybe the model is not affected as long as all the features and lanes are rotated by the same angle
      # NOTE: rotate (np.pi / 2 - yaw) so that the new coordinate y-axis is align with the yaw direction in original direction; rotate(np.pi - yaw) would results in new coordinate x-axis negatively align with the yaw direction, which doesn't make much sense
      theta = np.pi / 2 - np.arctan2(prev[1], prev[0])
      rot = np.asarray([[np.cos(theta), -np.sin(theta)], \
          [np.sin(theta), np.cos(theta)]], np.float32)

      ori_trajs, trajs, gt_preds, has_preds = [], [], [], []

      # given this instance (e.g. car A), and this sample (e.g. sample at time T), current car position
      # return:
      #     1) past global x, y
      #     2) past local x, y, v, acc, ang_velocity, constant 1.0
      #     3) gt future global x, y
      #     4) bool mask, whether has future gt
      ori_traj , agt_traj, agt_gt_pred, agt_has_pred = \
          self.get_trajs(instance_token, sample_token, orig, rot)

      trajs.append(agt_traj)
      ori_trajs.append(ori_traj)
      gt_preds.append(agt_gt_pred)
      has_preds.append(agt_has_pred)

      # there could be multiple agents in one sample, for example, pedestrain, etc
      present_history = self.helper.get_annotations_for_sample(sample_token)
      for pre_h in range(len(present_history)) :
          # only care about all the other vehicles (although there could be different type, e.g. car, bus, truck)
          if present_history[pre_h]['category_name'][:7] == 'vehicle':
              nei_ins, nei_sam = present_history[pre_h]["instance_token"], present_history[pre_h]["sample_token"]
              # for one agent exists in this sample, get all the past trajectory information for that particular agent
              ori_traj, nei_traj, nei_gt_pred, nei_has_pred = self.get_trajs(nei_ins, nei_sam, orig, rot)

              if len(nei_traj) == 1: # NOTE: this means there is no past trajectory for this agent
                continue
              if np.sum(trajs[0]-nei_traj)==0. : # this is the instance which we've already put into the trajs
                continue

              x_min, x_max, y_min, y_max = self.config['pred_range'] # default to be [-100, 100]
              if nei_traj[-1, 0] < x_min or nei_traj[-1, 0] > x_max or nei_traj[-1, 1] < y_min or nei_traj[
                  -1, 1] > y_max: # we only care about the agent that could be seen by the target agent
                  continue

              trajs.append(nei_traj)
              ori_trajs.append(ori_traj)
              gt_preds.append(nei_gt_pred)
              has_preds.append(nei_has_pred)

      ori_trajs = np.asarray(ori_trajs, np.float32)
      trajs = np.asarray(trajs, np.float32)
      ctrs = np.asarray(trajs[:, -1, :2], np.float32)

      gt_preds = np.asarray(gt_preds, np.float32)
      has_preds = np.asarray(has_preds, np.bool)

      data = dict()

      data['ctrs'] = ctrs # current local x, y of all agents in this scene
      data['feats'] = trajs # local x, y coordinate + v + acc + angular velocity + constant 1.0
      data['orig'] = orig # current position of the target agent
      data['theta'] = theta # estimated current yaw
      data['rot'] = rot # rotation matrix which rotate global translation to local translation
      data['ori_trajs'] = ori_trajs # past global x, y of all the instances in the current sample, THIS IS NOT ACTUALLY USED
      data['gt_preds'] = gt_preds # gt future global x, y of all the instances in the current sample
      data['has_preds'] = has_preds
      data['ins_sam'] = [instance_token, sample_token]
      return data


    def get_trajs(self, instance_token, sample_token, orig, rot):
        """
        given this instance (e.g. car A), and this sample (e.g. sample at time T), current car position
        return:
          1) past global x, y, 
          2) , 
          3) gt future global x, y
          4) bool mask, whether has future gt
        """

        # past x, y in global coordinate, 4*2 in our case
        past_traj = self.helper.get_past_for_agent(instance_token, sample_token, \
            seconds=self.config['train_size'], in_agent_frame=False)

        past_traj = np.asarray(past_traj, dtype=np.float32)

        if past_traj.shape[0] == 0 :
          return [0], [0], [0],[0]

        # get full annotation information
        traj_records = self.helper.get_past_for_agent(instance_token, sample_token, \
            seconds=self.config['train_size'], in_agent_frame=False, just_xy=False)

        # NOTE: this shouldn't happen if seconds are set correctly
        if past_traj.shape[0] > self.config['train_size'] * 2:
          past_traj = past_traj[0:int(self.config['train_size']) * 2]
          traj_records = traj_records[0:int(self.config['train_size']) * 2]

        # current x, y in global coordinate
        cur_traj = self.helper.get_sample_annotation(instance_token, \
            sample_token)["translation"][:2]
        cur_traj = np.asarray(cur_traj, dtype=np.float32)

        # all_trajs: store current + past global x, y
        all_trajs = np.zeros((past_traj.shape[0] + 1, 2), np.float32)
        all_trajs[0,:] = cur_traj
        all_trajs[1:, :] = past_traj

        # ori_traj: store current + past global x, y
        ori_traj = copy.deepcopy(all_trajs)

        trajs = np.zeros((all_trajs.shape[0], 6), dtype=np.float32)
        # NOTE: all_trajs - orig = translation vector in global coordinate, rotate to local coordinate
        trajs[:, 0:2] = np.matmul(rot, (all_trajs - orig.reshape(-1, 2)).T).T

        # in time decreasing order
        sam_tokens = [traj_records[i]['sample_token'] for i in range(len(traj_records))]
        sam_tokens.insert(0, sample_token)

        i_t = instance_token
        for k in range(len(sam_tokens)) :
          s_t = sam_tokens[k]
          v = self.helper.get_velocity_for_agent(i_t, s_t)
          a = self.helper.get_acceleration_for_agent(i_t, s_t)
          theta = self.helper.get_heading_change_rate_for_agent(i_t, s_t)

          if np.isnan(v):
              v = 0
          if np.isnan(a):
              a = 0
          if np.isnan(theta):
              theta = 0
          trajs[k, 2] = v
          trajs[k, 3] = a
          trajs[k, 4] = theta
          # TODO: why we need this last column
          trajs[k, 5] = 1.0

        # after padding, the matrix should be aranged from past to current, e.g. -2s, -1.5s, -1s, -0.5s, cur
        # if -2s data doesn't exist, fill with 0
        traj_zeropadded = np.zeros((int(self.config['train_size']) * 2 + 1, 6), dtype=np.float32)
        ori_zeropadded = np.zeros((int(self.config['train_size']) * 2 + 1, 2), dtype=np.float32)

        trajs = np.flip(trajs, 0)
        traj_zeropadded[-trajs.shape[0]:] = trajs

        ori_traj = np.flip(ori_traj, 0)
        ori_zeropadded[-ori_traj.shape[0]:] = ori_traj

        agt_pred = np.zeros((self.config["pred_size"] * 2, 2), np.float32)
        agt_has_pred = np.zeros(self.config["pred_size"] * 2, np.bool)

        # get ground truth future trajectory in global coordinate
        agt_gt_trajs =self.helper.get_future_for_agent(instance_token, sample_token, \
            seconds=self.config["pred_size"], in_agent_frame=False)

        agt_gt_trajs = np.asarray(agt_gt_trajs, dtype=np.float32)

        if agt_gt_trajs.shape[0] == 0 :
            return [0], [0], [0],[0]

        agt_pred[:agt_gt_trajs.shape[0],:] = agt_gt_trajs
        agt_has_pred[:agt_gt_trajs.shape[0]] = 1

        return np.asarray(ori_zeropadded, np.float32), \
            np.asarray(traj_zeropadded, np.float32), agt_pred, agt_has_pred


    def get_lane_feats(self, data):
        instance_token , sample_token = data['ins_sam']
        x, y = data['orig']
        nearby_trajs = data['ori_trajs'] # global trajs of nearby agents

        # Get all the record tokens that intersect a square patch of side length 2*radius centered on (x,y).
        lanes = self.map_api.get_records_in_radius(x, y, self.config["lane_radius"], ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        if len(lanes) ==0:
          lane_feats = np.zeros([self.config['lane'], (self.config["num_points"]), 2], dtype=float)
          label = [90]
          mask = np.zeros([self.config['lane']], dtype=float)
          mask = np.asarray(mask, np.float32)
          nearby_trajs_mask = \
              np.zeros([1,], dtype=float)
          nearby_trajs = \
              np.zeros([1,5,6], dtype=float)

          map_info = dict()

          map_info['lane_feats'] = lane_feats
          map_info['lane_ids'] = self.lane_id_seqs
          map_info['label'] = label
          map_info['mask'] = mask
          map_info['nearby_trajs'] = nearby_trajs

          return map_info

        # TODO: what does discretize_lane do to lane and lane_connector
        lanes = self.map_api.discretize_lanes(lanes, 1)

        self.lane_feats = []
        self.used_lane_ids = []
        self.snippet = []
        self.lane_snippets = []
        self.lane_id_seqs = []
        self.non_redun = []
        self.pass_lane = list(lanes.keys())
        pos = np.expand_dims(np.array([x,y]),0)
        # init_pos = data['ori_trajs'][0,0:1,:2] # the global position of target instance at time -2s, we don't need that no where

        for lane in lanes.keys(): # iterate through lane tokens
          if lane in self.used_lane_ids :
            continue
          try:
              lane_record = self.map_api.get_arcline_path(lane)
          except:
              continue
          cur_len = arcline_path_utils.length_of_lane(lane_record)
          lane_feat = [] + arcline_path_utils.discretize_lane(lane_record, resolution_meters=1) # x, y, yaw of all the discretized pose
          self.used_lane_ids.append(lane)
          extend_id_seqs, extend_points_seqs = self.extend_lane(lane,lane_feat,pos,cur_len)
          self.lane_id_seqs.extend(extend_id_seqs)
          self.lane_feats.extend(extend_points_seqs)

        map_info = dict()
        weight = np.arange(1, 13)
        if len(self.lane_feats) == 0:
          lane_feats = np.zeros([self.config['lane'], (self.config["num_points"]), 2],dtype=float)
          label = [90]

          map_info['lane_feats'] = lane_feats
          map_info['label'] = label
          return map_info

        lane_candidates = [np.array(lc,dtype=np.float32) for lc in self.lane_feats]
        dist = [(np.square(lc - np.reshape(data['orig'],[1,2])).sum(-1)).min(0) \
            for lc in lane_candidates] # find the minimum distance from a lane to current global origin
        last_ind = np.argsort(np.array(dist))[:self.config["lane"]]
        lane = [lane_candidates[i] for i in last_ind]
        lane_ids = [self.lane_id_seqs[i] for i in last_ind]

        # select the ground truth reference lane
        dist = [((np.square(np.expand_dims(lc,1)- \
            np.reshape(data['gt_preds'][0],[1,12,2])).sum(-1) \
            ).min(0)*weight).sum(-1) for lc in lane]
        lane_ind = np.argsort(np.array(dist))
        label = lane_ind[0]

        lane = np.array([self.unify_line(l[:,:2], self.config['num_points']) for l in lane], dtype=np.float32)

        def similar_check(lanes) :
          """
          delete all the lanes that are too close to each other
          """
          def dist(l1,l2) :
            return np.greater_equal(np.square((l1-l2)).sum(-1).max(-1),1.)
          cnt = 0
          while(cnt!=len(lanes)) :
            flag = dist(np.expand_dims(lanes[cnt],0),lanes)
            flag[cnt] = True
            lanes = lanes[flag]
            cnt += 1
          return lanes

        lane = similar_check(lane)

        lane_feats = np.zeros((len(lane), self.config["num_points"], 2))

        orig = np.expand_dims(np.expand_dims(data['orig'],0),0) # (1, 1, 2)
        lane_trajs = data['ori_trajs'][1:]
        rot_trajs = data['feats'][1:,:]

        nearby_trajs = []

        for i in range(lane.shape[0]) :
          _lane = lane[i]
          diff = np.expand_dims(np.expand_dims(_lane,1), 0) - np.expand_dims(lane_trajs, 1)
          diff = np.min(np.min(np.sqrt(np.sum(np.square(diff),-1)),-1),-1)
          lane_on_check = diff<1.5
          l_idx = np.arange(lane_on_check.shape[0])
          l_trajs = lane_trajs[lane_on_check]
          l_idx = l_idx[lane_on_check]
          diff = np.sqrt(np.sum(np.square(l_trajs-orig),-1))
          real_diff = []
          for j in range(diff.shape[0]) :
            tmp=diff[j]
            tmp=tmp[tmp!=0.]
            real_diff.append(tmp.min(-1))
          real_diff = np.array(real_diff)

          order = np.argsort(real_diff)

          nb_trjs = []
          for j in range(order.shape[0]) :
            # for each lane, there could be multiple agents have past trajectory that are close to that lane
            nb_trjs.append(rot_trajs[l_idx[order[j]]]) 
          nearby_trajs.append(nb_trjs)


        for i in range(len(lane)) :
          lane_feats[i, :, :] = np.matmul(data['rot'],(lane[i]-data['orig'].reshape(-1,2)).T).T

        mask = np.zeros([self.config['lane']], dtype=float)
        mask = np.asarray(mask, np.float32)
        mask[:len(lane_feats)] = 1.
        if lane_feats.shape[0] != self.config['lane'] :
          padding = np.asarray(np.zeros([self.config['lane'] - lane_feats.shape[0], \
              self.config["num_points"], 2],dtype=float), np.float32)
          lane_feats = np.vstack([lane_feats, padding])
          lane_feats = np.asarray(lane_feats, np.float32)

        map_info['lane_feats'] = lane_feats # rotated nearby lane feature
        map_info['lane_ids'] = lane_ids # nearby lane ids
        map_info['label'] = label # gt reference lane label
        map_info['mask'] = mask # whether have meaningful lane feature
        map_info['nearby_trajs'] = nearby_trajs # (num_lane, num_nearby_vehicle, rot_trajs_of_each_agent)
        return map_info


    def extend_lane(self,lane_id,lane_points,cur_pos,cur_len) :
      base_points = np.array(lane_points)[:,:2]

      ##### check meaningful inst
      # backward_ids = lane_id
      # backward_points = base_points
      backward_ids, backward_points = self.find_backward_lane(lane_id,base_points,cur_pos,cur_len)
      self.snippet = backward_ids
      forward_id_seqs, forward_points_list = self.find_forward_lanes(lane_id,base_points,cur_pos,cur_len)
      lane_id_seqs = [backward_ids+forward_ids for forward_ids in forward_id_seqs]
      points_seqs = [np.concatenate([backward_points,forward_points],axis=0) for forward_points in forward_points_list]

      return lane_id_seqs, points_seqs


    def find_forward_lanes(self,lane_id,points,cur_pos, \
        cum_dist=0.,flag=True) :
      """
      Input:
        lane_id: lane token of the lane to be extended
        points: discretized pose (x, y, yaw) of the lane to be extended
        cur_pos: current position of the target vehicle in global coord
      Output:
        lane_id_seqs: [[lane ids of one set of possible forward lanes], [lane ids of second set of possible forward lanes], ...]
        points_list: points on multiple possible forward lanes
      """
      self.snippet.extend([lane_id])
      if flag :
        cur_pos_idx = np.linalg.norm(points-cur_pos,axis=1).argmin()
        cur_pos_idx = points.shape[0]-2 if cur_pos_idx > points.shape[0]-2 else cur_pos_idx # TODO: ???
        points = points[cur_pos_idx:]
        dist_pts = np.linalg.norm(points[1:]-points[:-1],axis=1)
        cum_dist_pts = np.cumsum(dist_pts)
        flag = False
      else :
        lane_id, forward_points, check_pts = self.get_lane_points(lane_id)
        if not check_pts :
          # self.snippet.extend([lane_id])
          self.lane_snippets.append(self.snippet)
          self.snippet = []
          return [[lane_id]], [points]
        points = np.concatenate([points,forward_points],axis=0)
        dist_pts = np.linalg.norm(points[1:]-points[:-1],axis=1)
        cum_dist_pts = np.cumsum(dist_pts)

      if cum_dist_pts[-1] > self.config['lane_forward_length'] : # current lane already long enough in forward direction
        idx = 0
        while cum_dist_pts[idx] < self.config['lane_forward_length'] :
          idx += 1
        points = points[:idx]
        # self.snippet.extend([lane_id])
        self.lane_snippets.append(self.snippet)
        self.snippet = []
        return [[lane_id]], [points]
      else : # need to append more outgoing lines
        lane_id_seqs = []
        points_list = []
        outgoing_lane_ids = self.map_api.get_outgoing_lane_ids(lane_id)
        if len(outgoing_lane_ids) == 0 :
          # self.snippet.extend([lane_id])
          self.lane_snippets.append(self.snippet)
          self.snippet = []
          return [[lane_id]], [points]
        else :
          if len(outgoing_lane_ids)>1 :
            self.lane_snippets.append(self.snippet)
            self.snippet = []
          for og_lane_id in outgoing_lane_ids :
            self.used_lane_ids.append(og_lane_id)
            # Note: W T F? I guess it's trying to search forward lanes until get enough forward distance
            forward_lane_ids, forward_points = self.find_forward_lanes(og_lane_id,points,cur_pos,flag=flag)
            lane_id_seqs.extend([[lane_id]+fw_ids for fw_ids in forward_lane_ids])
            points_list.extend(forward_points)
          return lane_id_seqs, points_list


    def find_backward_lane(self,lane_id,points,cur_pos, \
        cum_dist=0.,flag=True) :
      """
      Input:
        lane_id: lane token of the lane to be extended
        points: discretized pose (x, y, yaw) of the lane to be extended
        cur_pos: current position of the target vehicle in global coord
      Output:
        lane_ids: lane behind the target vehicle
        points: points on backward lane, truncated to be about 20 in total length
      """
      lane_ids = []
      cum_dist_pts = [np.inf]
      while flag or cum_dist_pts[-1] < self.config['lane_backward_length'] :
        if flag :
          # find the road pose closest to the current position
          # NOTE: points are 3 dimentional, cur_pos is 2 dimentional?
          cur_pos_idx = np.linalg.norm(points-cur_pos,axis=1).argmin()
          cur_pos_idx = 2 if cur_pos_idx < 2 else cur_pos_idx # TODO: ???
          points = points[:cur_pos_idx]
          dist_pts = np.linalg.norm(points[1:]-points[:-1],axis=1)[::-1]
          cum_dist_pts = np.cumsum(dist_pts)
          flag = False
        else :
          incoming_lane_ids = self.map_api.get_incoming_lane_ids(lane_id)
          if len(incoming_lane_ids) == 0 :
            break
          # lane_id = get_closest_lane(nusc_map,incoming_lane_ids,start_pos)
          for li in incoming_lane_ids :
            self.used_lane_ids.append(li) # TODO: why add all incoming lane to used_lane_ids
          lane_id = incoming_lane_ids[0] # the neareast incoming lane
          lane_id, backward_points, check_pts = self.get_lane_points(lane_id)
          if not check_pts :
            break
          points = np.concatenate([backward_points,points],axis=0)
          dist_pts = np.linalg.norm(points[1:]-points[:-1],axis=1)[::-1]
          cum_dist_pts = np.cumsum(dist_pts)
          lane_ids = [lane_id]+lane_ids
        if cum_dist_pts.shape[0] == 0 :
          cum_dist_pts = [-1]

      # idx = 0
      # while cum_dist_pts[idx] < config.lane_backward_length :
      #   idx += 1
      #   if idx == cum_dist_pts.shape[0] :
      #     break
      idx_test = np.abs(cum_dist_pts- self.config['lane_backward_length']).argmin()
      points = points[-idx_test:]

      return lane_ids, points


    def get_lane_points(self,lane_ids) :
      if isinstance(lane_ids,str) : # called by find_backward_line
        lane_id = lane_ids
        try :
          lane_record = self.map_api.get_arcline_path(lane_id)
          lane_feat = [] + arcline_path_utils.discretize_lane( \
              lane_record, resolution_meters=1)
        except :
          lane_feat = []
        # print(np.array(points).max(0)-np.array(points).min(0))
        if len(lane_feat) == 0 :
          check_pts = False
        else :
          check_pts = True
          lane_feat = np.array(lane_feat)[:,:2]
        return lane_id, lane_feat, check_pts
      else :
        check_pts = []
        lane_feats = []
        for idx, lane_id in enumerate(lane_ids) :
          # print('\n',lane_ids[idx],'\n',lane_id)
          # assert lane_ids[idx] == lane_id, 'check get lane points'
          try :
            lane_record = self.map_api.get_arcline_path(lane_id)
            lane_feat = [] + arcline_path_utils.discretize_lane( \
                lane_record, resolution_meters=1)
          except :
            lane_feat = []
          if len(lane_feat) == 0 :
            check_ = False
          else :
            check_ = True
            lane_feat = np.array(lane_feat)[:,:2]
          check_pts.append(check_)
          lane_feats.append(lane_feat)
        return lane_ids, lane_feats, check_pts


    def unify_line(self, points, num_points):
        """
        Make line unified
        Input:
          points: lane features
          num_points
        Return:
          list of points of length 'num_points', basically turn the arbitraty number of  lane points to be of 
          fixed number of points (which is set to 20) in our case
        """

        # Return array
        ret = np.zeros(shape=(num_points, 2), dtype=np.float32)

        # Line distance
        dist_pts = np.linalg.norm(points[1:] - points[:-1], axis=1)
        line_dist = np.cumsum(dist_pts)[-1]

        # Distance per segment
        segment_dist = line_dist / float(num_points - 1)

        # Current distance
        curr_dist = segment_dist

        # Copy first point as it is
        ret[0] = points[0]

        # Current point
        curr_point = points[0]

        # Current index for ret array
        ret_index = 1

        # Current index in line
        i = 1

        # Loop num_points-1 times
        while ret_index < num_points - 1:
            # Distance
            dist = np.linalg.norm(curr_point - points[i])

            if dist != 0:
                if dist >= curr_dist:
                    # Displacement vector
                    disp_vector = points[i] - curr_point
                    # Ratio of distances
                    ratio = curr_dist / dist
                    # Calculate curr_point
                    curr_point = curr_point + disp_vector * ratio
                    # Add curr_point to ret
                    ret[ret_index] = curr_point
                    ret_index = ret_index + 1
                    # Update curr_distance
                    curr_dist = segment_dist
                else:
                    # Update distance
                    curr_dist = curr_dist - dist
                    # Update curr_point and i
                    curr_point = points[i]
                    i = min(i + 1, len(points) - 1)
            else:
                i = min(i + 1, len(points) - 1)

        # Copy last point as it is
        ret[num_points - 1] = points[-1]

        return ret


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch
