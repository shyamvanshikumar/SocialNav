import math
import warnings

import torch
from torch import nn, Tensor

import pytorch_lightning as pl

from train import config as CFG
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder
from models.utils import MLP

class AttnNav(pl.LightningModule):
    def __init__(self,
                 rgb_encoder,
                 lidar_encoder,
                 rob_traj_decoder,
                 mot_decoder,
                 embed_dim,
                 enable_rob_dec=True,
                 enable_mot_dec=False,
                 freeze_enc=False,
                 optimizer="AdamW",
                 lr=0.001,
                 weight_decay=0.01,
                 momentum=0.9):

        super().__init__()
        
        self.rgb_encoder = rgb_encoder
        self.lidar_encoder = lidar_encoder
        self.rob_decoder = rob_traj_decoder
        self.mot_decoder = mot_decoder

        self.mlp_head = nn.Linear(in_features=embed_dim, out_features=1)

        self.enable_mot_dec = enable_mot_dec
        self.enable_rob_dec = enable_rob_dec

        if freeze_enc:
            self.rgb_encoder.requires_grad_(False)
            self.lidar_decoder.requires_grad_(False)

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.save_hyperparameters(ignore=["rgb_encoder", "lidar_encoder", "rob_traj_decoder", "mot_decoder"])

    def forward(self, rgb_img, lidar_img, trg_pose_seq, trg_mot_seq):
        B, N, C = trg_pose_seq.shape
        rgb_enc_out = self.rgb_encoder(rgb_img)
        lidar_enc_out = self.lidar_encoder(lidar_img)
        enc_output = torch.cat([rgb_enc_out, lidar_enc_out], dim=2)
        trg_pose_seq = torch.cat([torch.zeros((B,2), device=trg_pose_seq.device).unsqueeze(dim=1), trg_pose_seq], dim=1)
        B, O, N, C = trg_mot_seq.shape
        #trg_mot_seq = torch.cat([torch.zeros((B,O,2), device=trg_mot_seq.device).unsqueeze(dim=2), trg_mot_seq], dim=2)

        num_objects = None
        rob_dec_output = None
        mot_dec_output = None

        if self.enable_rob_dec:
            rob_dec_output = self.rob_decoder(enc_output[:,1:], trg_pose_seq[:,:-1])
            
        if self.enable_mot_dec:
            num_objects = self.mlp_head(enc_output[:,1])
            #input_seq = trg_mot_seq[:,:,:-1].reshape(B*O, N, C)
            mot_dec_output = self.mot_decoder(enc_output[:,1:], trg_mot_seq[:,:,:-1]) #.reshape(B,O,N,C)
        return rob_dec_output, mot_dec_output, num_objects
    
    def training_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        rob_dec_output, mot_dec_output, num_objects = self.forward(image, lidar, pose, mot_traj)
        if self.enable_rob_dec:
            loss = self.criterion(rob_dec_output, pose)
        if self.enable_mot_dec:
            B, O, N, C = mot_traj.shape
            mot_traj = mot_traj[:,:,1:]
            loss = self.criterion(num_objects.squeeze(), num_obj.to(dtype=torch.float32))
            for i in range(B):
                loss += self.criterion(mot_dec_output[i,:num_obj[i]], mot_traj[i,:num_obj[i]])
        #print("\n ++ ",loss, "++\n")

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        rob_dec_output, mot_dec_output, num_objects = self.forward(image, lidar, pose, mot_traj)
        if self.enable_rob_dec:
            loss = self.criterion(rob_dec_output, pose)
        elif self.enable_mot_dec:
            B, O, N, C = mot_traj.shape
            mot_traj = mot_traj[:,:,1:]
            #print(num_objects.shape, num_obj.shape)
            loss = self.criterion(num_objects.squeeze(), num_obj.to(dtype=torch.float32))
            for i in range(B):
                loss += self.criterion(mot_dec_output[i,:num_obj[i]], mot_traj[i,:num_obj[i]])
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
    
