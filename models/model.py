import math
import warnings

import torch
from torch import nn, Tensor
import pytorch_lightning as pl

from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder

class AttnNav(pl.LightningModule):
    def __init__(self,
                 img_size=240,
                 patch_size=8,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 optimizer="AdamW",
                 lr=0.001,
                 weight_decay=0.01,
                 momentum=0.9):

        super().__init__()
        
        self.rgb_encoder = VisionTransformer(img_size=img_size,
                                            patch_size=patch_size,
                                            input_channels=3,
                                            embed_dim=embed_dim,
                                            depth=depth,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop_rate=drop_rate,
                                            attn_drop_rate=attn_drop_rate,
                                            drop_path_rate=drop_path_rate,
                                            norm_layer=norm_layer)
        self.lidar_encoder = VisionTransformer(img_size=img_size,
                                            patch_size=patch_size,
                                            input_channels=1,
                                            embed_dim=embed_dim,
                                            depth=depth,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop_rate=drop_rate,
                                            attn_drop_rate=attn_drop_rate,
                                            drop_path_rate=drop_path_rate,
                                            norm_layer=norm_layer)
        self.decoder = TransformerDecoder(embed_dim=embed_dim,
                                          depth=depth,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          drop_rate=drop_rate,
                                          attn_drop_rate=attn_drop_rate,
                                          drop_path_rate=drop_path_rate,
                                          norm_layer=norm_layer)

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, lidar, pose = batch
        dec_output = self.forward(image, lidar, pose)
        loss = self.criterion(dec_output, pose)
        #print("val",batch_idx, loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            raise ValueError("Invalid name of optimizer")
        return optimizer
    
