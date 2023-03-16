import torch
import numpy as np

from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint


from models.model import AttnNav
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder
from models.dataloader import NavSetDataModule
import train.config as CFG

rgb_encoder = VisionTransformer(
                    img_size=CFG.img_size,
                    patch_size=CFG.patch_size,
                    input_channels=3,
                    embed_dim=CFG.embed_dim,
                    depth=CFG.depth,
                    num_heads=CFG.num_heads,
                    drop_rate=CFG.drop_rate,
                    attn_drop_rate=CFG.attn_drop_rate,
                    drop_path_rate=CFG.drop_path_rate
                    )

lidar_encoder = VisionTransformer(
                    img_size=CFG.img_size,
                    patch_size=CFG.patch_size,
                    input_channels=1,
                    embed_dim=CFG.embed_dim,
                    depth=CFG.depth,
                    num_heads=CFG.num_heads,
                    drop_rate=CFG.drop_rate,
                    attn_drop_rate=CFG.attn_drop_rate,
                    drop_path_rate=CFG.drop_path_rate
                    )

rob_traj_decoder = TransformerDecoder(
                    embed_dim=CFG.embed_dim,
                    depth=CFG.depth,
                    num_heads=CFG.num_heads,
                    drop_rate=CFG.drop_rate,
                    attn_drop_rate=CFG.attn_drop_rate,
                    drop_path_rate=CFG.drop_path_rate
                    )

mot_decoder = TransformerDecoder(
                    embed_dim=CFG.embed_dim,
                    depth=CFG.depth,
                    num_heads=CFG.num_heads,
                    drop_rate=CFG.drop_rate,
                    attn_drop_rate=CFG.attn_drop_rate,
                    drop_path_rate=CFG.drop_path_rate,
                    multi=True
                    )

model_ckpt = "/workspace/project/trained_models/rob_train_3sec_spread_pose_end2end15-03-2023-16-34-19.ckpt"

print("Model loaded from checkpoint"+model_ckpt)
model = AttnNav.load_from_checkpoint(model_ckpt,
                                    rgb_encoder=rgb_encoder,
                                    lidar_encoder=lidar_encoder,
                                    rob_traj_decoder=rob_traj_decoder,
                                    mot_decoder=mot_decoder,
                                    enable_rob_dec=True,
                                    enable_mot_dec=False)

datamodel = NavSetDataModule(save_data_path=CFG.save_data_path,
                             train_rosbag_path=CFG.train_rosbag_path,
                             val_rosbag_path=CFG.val_rosbag_path,
                             test_rosbag_path=CFG.val_rosbag_path,
                             batch_size=CFG.batch_size,
                             num_workers=CFG.num_workers,
                             pin_memory=CFG.pin_memory)

trainer = Trainer(
    accelerator='gpu',
    devices=1,
    logger=pl_loggers.TensorBoardLogger("logs/"),
    )


print("Starting testing")
trainer.test(model, datamodel)
