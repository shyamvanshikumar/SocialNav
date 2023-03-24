import math
import warnings

import torch
from torch import nn, Tensor
from scipy.spatial.distance import directed_hausdorff, cdist
#from scipy.spactial import ConvexHull
import numpy as np

import pytorch_lightning as pl

from train import config as CFG
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder
from models.utils import MLP

def dist_line_point(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

class CollLoss(nn.Module):
    def __init__(self, sparse_mot=False):
        super().__init__()
        self.sparse_mot = sparse_mot
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, pred_rob_traj, mot_traj, num_obj):
        if not self.sparse_mot:
            mot_traj = mot_traj[:,5::5]
        # loss = 0
        # for i in range(num_obj):
        #     for j in range(len(pred_rob_traj)):
        #         if math.dist(pred_rob_traj[j], mot_traj[i][j]) <= 1:
        #             loss += 1

        loss = 0
        for i in range(num_obj):
            dist = cdist(pred_rob_traj.to('cpu').detach(), mot_traj[i].to('cpu').detach())
            loss += np.trace(dist)

        # for i in range(num_obj):
        #     mot_traj_np = mot_traj[i].to('cpu').detach()
        #     hull = ConvexHull(mot_traj_np)
        #     points = hull.points
        #     dists = []
        #     p = pred_rob_traj
        #     for i in range(len(points)-1):
        #         dists.append(dist(points[i][0],points[i][1],points[i+1][0],points[i+1][1],p[0],p[1]))
        #     dist = min(dists)
        return loss

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
                 use_coll_loss=False,
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
        self.use_coll_loss = use_coll_loss

        #self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        #self.fc2 = nn.Linear(in_features=embed_dim, out_features=12)

        if freeze_enc:
            print("encoder_frozen")
            self.rgb_encoder.requires_grad_(False)
            self.lidar_decoder.requires_grad_(False)

        self.criterion = nn.MSELoss(reduction='sum')
        self.coll_loss = CollLoss()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.test_output = {"coll_hausdorff_dist":0.0, "coll_minimin_dist":0.0, "num_examples":0.0, "pose_error":0.0, "hausdorff":0.0}

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
                #rob_dec_output = self.fc2(self.fc1(output)).reshape(B,6,2)
            
        if self.enable_mot_dec:
            num_objects = self.mlp_head(enc_output[:,1])
            #input_seq = trg_mot_seq[:,:,:-1].reshape(B*O, N, C)
            mot_dec_output = self.mot_decoder(enc_output[:,1:], trg_mot_seq[:,:,:-1]) #.reshape(B,O,N,C)
        return rob_dec_output, mot_dec_output, num_objects
    
    def training_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        rob_dec_output, mot_dec_output, num_objects = self.forward(image, lidar, pose, mot_traj)
        if self.enable_rob_dec and (not self.enable_mot_dec):
            loss = self.criterion(rob_dec_output, pose)

        B, O, N, C = mot_traj.shape

        if self.enable_mot_dec:
            mot_traj = mot_traj[:,:,1:]
            loss = self.criterion(num_objects.squeeze(), num_obj.to(dtype=torch.float32))
            for i in range(B):
                loss += self.criterion(mot_dec_output[i,:num_obj[i]], mot_traj[i,:num_obj[i]])
            if self.enable_rob_dec:
                loss += self.criterion(rob_dec_output, pose)
        #print("\n ++ ",loss, "++\n")

        if self.use_coll_loss:
            for i in range(B):
                loss += -1e-3 * self.coll_loss(rob_dec_output[i], mot_traj[i], num_obj[i])

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        rob_dec_output, mot_dec_output, num_objects = self.forward(image, lidar, pose, mot_traj)
        if self.enable_rob_dec and (not self.enable_mot_dec):
            loss = self.criterion(rob_dec_output, pose)

        B, O, N, C = mot_traj.shape
            
        if self.enable_mot_dec:
            mot_traj = mot_traj[:,:,1:]
            loss = self.criterion(num_objects.squeeze(), num_obj.to(dtype=torch.float32))
            for i in range(B):
                loss += self.criterion(mot_dec_output[i,:num_obj[i]], mot_traj[i,:num_obj[i]])
            if self.enable_rob_dec:
                loss += self.criterion(rob_dec_output, pose)
        #print("val",batch_idx, loss)
        if self.use_coll_loss:
            for i in range(B):
                loss += -1e-3 * self.coll_loss(rob_dec_output[i], mot_traj[i], num_obj[i])

        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        image, lidar, pose, mot_traj, num_obj = batch
        #insure that rob_dec is active and mot_dec is inactive
        B, N, C = pose.shape
        rob_dec_output, _, _ = self.forward(image, lidar, pose, mot_traj)
        pose_error = self.criterion(rob_dec_output, pose)/len(pose)

        coll_hausdorff_dist = 0.0
        coll_minimin_dist = 0.0
        hausdorff_dist = 0.0

        for i in range(B):
            haus_dist, minimin_dist = self.collision_dist(rob_dec_output[i], mot_traj[i], num_obj[i])
            coll_hausdorff_dist += haus_dist
            coll_minimin_dist += minimin_dist
            hausdorff_dist += max(directed_hausdorff(rob_dec_output[i].to('cpu'), pose[i].to('cpu'))[0], 
                                    directed_hausdorff(pose[i].to('cpu'), rob_dec_output[i].to('cpu'))[0])

        self.test_output["num_examples"] += len(batch)
        self.test_output["coll_hausdorff_dist"] += coll_hausdorff_dist
        self.test_output["coll_minimin_dist"] += coll_minimin_dist
        self.test_output["pose_error"] += pose_error.item()
        self.test_output["hausdorff"] += hausdorff_dist
    
    def on_test_epoch_end(self):
        avg_coll_hausdorff_dist = self.test_output["coll_hausdorff_dist"]/self.test_output["num_examples"]
        avg_coll_minimin_dist = self.test_output["coll_minimin_dist"]/self.test_output["num_examples"]
        avg_pose_error = self.test_output["pose_error"]/self.test_output["num_examples"]
        avg_hausdorff_dist = self.test_output["hausdorff"]/self.test_output["num_examples"]

        print("avg_coll_hausdorff_dist", avg_coll_hausdorff_dist)
        print("avg_coll_minimin_dist", avg_coll_minimin_dist)
        print("avg_pose_error", avg_pose_error)
        print("avg_hausdorff_dist", avg_hausdorff_dist)

        self.log("avg_coll_hausdorff_dist", avg_coll_hausdorff_dist)
        self.log("avg_coll_minimin_dist", avg_coll_minimin_dist)
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
        coll_minimin_dist = 0.0
        for i in range(num_obj):
            coll_hausdorff_dist += max(directed_hausdorff(rob_traj.to('cpu'), mot_traj[i].to('cpu'))[0],
                                        directed_hausdorff(mot_traj[i].to('cpu'), rob_traj.to('cpu'))[0])
            coll_minimin_dist += np.amin(cdist(rob_traj.to('cpu'), mot_traj[i].to('cpu')))
        return coll_hausdorff_dist/num_obj, coll_minimin_dist/num_obj

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
    
