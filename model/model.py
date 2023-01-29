import math
import warnings

import torch
from torch import nn, Tensor

from encoder import VisionTransformer
from decoder import TransformerDecoder

class AttnNav(nn.Module):
    def __init__(self,
                 rgb_encoder,
                 lidar_encoder,
                 decoder):

        super().__init__()
        
        self.rgb_encoder = rgb_encoder
        
        self.lidar_encoder = lidar_encoder
        
        self.decoder = decoder

    def forward(self, rgb_img, lidar_img, trg_pose_seq):
        rgb_enc_out = self.rgb_encoder(rgb_img)
        lidar_enc_out = self.lidar_encoder(lidar_img)

        enc_output = torch.cat([rgb_enc_out, lidar_enc_out], dim=1)
        dec_output = self.decoder(enc_output, trg_pose_seq)
        return dec_output