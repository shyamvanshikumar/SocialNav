import math
import warnings

import torch
from torch import nn, Tensor
import pytorch_lightning as pl

from .encoder import VisionTransformer
from .decoder import TransformerDecoder

class AttnNav(pl.LightningModule):
    def __init__(self,
                 rgb_encoder,
                 lidar_encoder,
                 decoder):

        super().__init__()
        
        self.rgb_encoder = rgb_encoder
        self.lidar_encoder = lidar_encoder
        self.decoder = decoder
        self.criterion = nn.MSELoss(reduction='sum')
        self.lr = 0.0003

    def forward(self, rgb_img, lidar_img, trg_pose_seq):
        B, N, C = trg_pose_seq.shape
        rgb_enc_out = self.rgb_encoder(rgb_img)
        lidar_enc_out = self.lidar_encoder(lidar_img)
        enc_output = torch.cat([rgb_enc_out, lidar_enc_out], dim=2)
        trg_pose_seq = torch.cat([torch.zeros((B,2), device=trg_pose_seq.device).unsqueeze(dim=1), trg_pose_seq], dim=1)
        dec_output = self.decoder(enc_output, trg_pose_seq[:,:-1])
        return dec_output
    
    def training_step(self, batch, batch_idx):
        image, lidar, pose = batch
        dec_output = self.forward(image, lidar, pose)
        loss = self.criterion(dec_output, pose)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, lidar, pose = batch
        dec_output = self.forward(image, lidar, pose)
        loss = self.criterion(dec_output, pose)
        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
