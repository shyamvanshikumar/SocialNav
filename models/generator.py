import torch
import torch.nn as nn
import numpy as np
import math
from models.model import CollLoss
from scipy.spatial.distance import directed_hausdorff, cdist

class Generator(nn.Module):
    ''' Load a model and generate possible robot paths in beam search fashion '''

    def __init__(self,
                 model,
                 beam_size=10,
                 max_seq_len=6
                 ):
        
        super().__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        
        self.model = model
        self.model.eval()

    def _get_init_state(self, rgb_img, lidar_img):
        beam_size = self.beam_size

        rgb_img = rgb_img.unsqueeze(dim=0)
        lidar_img = lidar_img.unsqueeze(dim=0)
        rgb_enc_out = self.model.rgb_encoder(rgb_img)
        lidar_enc_out = self.model.lidar_encoder(lidar_img)
        enc_output = torch.cat([rgb_enc_out, lidar_enc_out], dim=2)
        enc_output = enc_output.repeat(beam_size,1,1)

        init_seq = torch.zeros((beam_size,self.max_seq_len+1,2), dtype=torch.float32)

        return enc_output, init_seq
    
    def _get_next_state(self, enc_output, gen_seq, ts):
        dec_output = self.model.rob_decoder(enc_output[:,1:], gen_seq[:,:ts])
        mean_pos = dec_output[:,-1]
        gen_seq[:,ts] = torch.normal(mean_pos, std=0.3)
        return gen_seq

    def generate_path(self, rgb_img, lidar_img):
        enc_output, gen_seq = self._get_init_state(rgb_img, lidar_img)

        for ts in range(1,self.max_seq_len+1):
            gen_seq = self._get_next_state(enc_output, gen_seq, ts)
        
        return gen_seq

    def coll_metric(self, pred_rob_traj, mot_traj, num_obj):
        mot_traj = mot_traj[:,5::5]
        print("here")
        loss = 0
        for i in range(num_obj):
            for j in range(len(pred_rob_traj)):
                if math.dist(pred_rob_traj[j], mot_traj[i][j]) <= 1:
                    loss += 1
        return loss
    
    def select_one(self, gen_seq):
        idx = np.random.randint(length(gen_seq))
        return gen_seq[idx]
    
    def select_best(self, gen_seq, mot_traj, num_obj, criteria="count"):
        if criteria == "count":
            coll_criteria = self.coll_metric
        else:
            coll_criteria = CollLoss()
        coll_dist = []
        for i in range(len(gen_seq)):
            if gen_seq[i,-1,0].item() - gen_seq[i,0,0] <= 1:
                a = 1000 if criteria == "count" else 0.0
                coll_dist.append(a)
            else:
                coll_dist.append(coll_criteria(gen_seq[i,1:], mot_traj, num_obj))
        
        print(coll_dist)
        if criteria == "count":
            max_dist = min(coll_dist)
        else:
            max_dist = max(coll_dist)
        idx = coll_dist.index(max_dist)
        return gen_seq[idx], max_dist

        


