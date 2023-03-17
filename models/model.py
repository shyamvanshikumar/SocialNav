import math
import warnings

import torch
from torch import nn, Tensor
from scipy.spatial.distance import directed_hausdorff

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
                 auto_reg=True,
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
        self.auto_reg = auto_reg

        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.fc2 = nn.Linear(in_features=embed_dim, out_features=12)

        if freeze_enc:
            print("encoder_frozen")
            self.rgb_encoder.requires_grad_(False)
            self.lidar_decoder.requires_grad_(False)

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.test_output = {"collision_dist":0.0, "num_examples":0.0, "pose_error":0.0, "hausdorff":0.0}

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
            if self.auto_reg:
                rob_dec_output = self.rob_decoder(enc_output[:,1:], trg_pose_seq[:,:-1])
            else:
                initial_pose = torch.zeros((B,2), device=trg_pose_seq.device).unsqueeze(dim=1)
                output = self.rob_decoder(enc_output[:,1:], initial_pose)
                rob_dec_output = self.fc2(self.fc1(output)).reshape(B,6,2)
            
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
    
    def test_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        #insure that rob_dec is active and mot_dec is inactive
        B, N, C = pose.shape
        rob_dec_output, _, _ = self.forward(image, lidar, pose, mot_traj)
        pose_error = self.criterion(rob_dec_output, pose)/len(pose)

        coll_dist = 0.0
        hausdorff_dist = 0.0

        for i in range(B):
            coll_dist += self.collision_dist(rob_dec_output[i], mot_traj[i], num_obj[i])
            hausdorff_dist += max(directed_hausdorff(rob_dec_output[i].to('cpu'), pose[i].to('cpu'))[0], 
                                                    directed_hausdorff(pose[i].to('cpu'), rob_dec_output[i].to('cpu'))[0])

        self.test_output["num_examples"] += len(batch)
        self.test_output["collision_dist"] += coll_dist
        self.test_output["pose_error"] += pose_error.item()
        self.test_output["hausdorff"] += hausdorff_dist
    
    def on_test_epoch_end(self):
        avg_coll_dist = self.test_output["collision_dist"]/self.test_output["num_examples"]
        avg_pose_error = self.test_output["pose_error"]/self.test_output["num_examples"]
        avg_hausdorff_dist = self.test_output["hausdorff"]/self.test_output["num_examples"]

        print("avg_coll_dist", avg_coll_dist)
        print("avg_pose_error", avg_pose_error)
        print("avg_hausdorff_dist", avg_hausdorff_dist)

        self.log("avg_coll_dist", avg_coll_dist)
        self.log("avg_pose_error", avg_pose_error)
        self.log("avg_hausdorff_dist", avg_hausdorff_dist)
    
    def generate_seq(self, rgb_img, lidar_img, pose, seq_len):
        B = rgb_img.shape[0]
        rgb_enc_out = self.rgb_encoder(rgb_img)
        lidar_enc_out = self.lidar_encoder(lidar_img)
        enc_output = torch.cat([rgb_enc_out, lidar_enc_out], dim=2)
        gen_seq = torch.zeros((B,seq_len,2), dtype=torch.float32)
        pose = torch.cat([torch.zeros((B,2), device=pose.device).unsqueeze(dim=1), pose], dim=1)
        
        pose_out = self.rob_decoder(enc_output[:,1:], pose)
        #print(rgb_enc_out)
        #print(lidar_enc_out)
        print(pose)
        for ts in range(0,seq_len-2):
            dec_output = self.rob_decoder(enc_output[:,1:], gen_seq[:,:ts+1])
            print(dec_output)
            gen_seq[:,ts+1] = dec_output[:,-1]
        
        return gen_seq, pose_out, enc_output
    
    @staticmethod
    def collision_dist(rob_traj, mot_traj, num_obj):
        coll_hausdorff_dist = 0.0
        for i in range(num_obj):
            coll_hausdorff_dist += max(directed_hausdorff(rob_traj.to('cpu'), mot_traj[i].to('cpu'))[0],
                                        directed_hausdorff(mot_traj[i].to('cpu'), rob_traj.to('cpu'))[0])
        return coll_hausdorff_dist/num_obj

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
        return [optimizer] #, [lr_scheduler]
    
