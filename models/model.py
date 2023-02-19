import math
import warnings

import torch
from torch import nn, Tensor

import pytorch_lightning as pl

from train import config as CFG
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder

class AttnNav(pl.LightningModule):
    def __init__(self,
                 rgb_encoder,
                 lidar_encoder,
                 rob_traj_decoder,
                 mot_decoder=None,
                 only_rob=True,
                 only_mot=False,
                 optimizer="AdamW",
                 lr=0.001,
                 weight_decay=0.01,
                 momentum=0.9):

        super().__init__()
        
        self.rgb_encoder = rgb_encoder
        self.lidar_encoder = lidar_encoder
        self.decoder = rob_traj_decoder
        self.mot_decoder = mot_decoder

        self.only_mot = only_mot
        self.only_rob = only_rob

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.save_hyperparameters()

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
        #print("\n ++ ",loss, "++\n")
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, lidar, pose = batch
        dec_output = self.forward(image, lidar, pose)
        loss = self.criterion(dec_output, pose)
        #print("val",batch_idx, loss)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            raise ValueError("Invalid name of optimizer")
        lambda1 = lambda epoch: (1-epoch/CFG.epochs)**0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        return [optimizer], [lr_scheduler]
    
