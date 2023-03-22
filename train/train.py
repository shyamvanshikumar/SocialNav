from datetime import datetime
import torch
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint

from models.model import AttnNav
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder
from models.dataloader import NavSetDataModule
import train.config as CFG

# training callbacks
early_stopping_cb = EarlyStopping(monitor='val_loss',
                                  mode='min',
                                  min_delta=0.00,
                                  patience=10)
swa_cb = StochasticWeightAveraging(swa_lrs=1e-2)
model_checkpoint_cb = ModelCheckpoint(
    dirpath='trained_models/',
    filename='rob_train_coll_loss_trial'+datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
    monitor='val_loss',
    mode='min')

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
                    drop_path_rate=CFG.drop_path_rate,
                    auto_reg=CFG.auto_reg
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

if CFG.ckp_path != None:
    print("Model loaded from checkpoint"+CFG.ckp_path)
    model = AttnNav.load_from_checkpoint(CFG.ckp_path,
                                     rgb_encoder=rgb_encoder,
                                     lidar_encoder=lidar_encoder,
                                     rob_traj_decoder=rob_traj_decoder,
                                     mot_decoder=mot_decoder,
                                     auto_reg=CFG.auto_reg,
                                     use_coll_loss=CFG.use_coll_loss)

else: 
    model = AttnNav(rgb_encoder=rgb_encoder,
                    lidar_encoder=lidar_encoder,
                    rob_traj_decoder=rob_traj_decoder,
                    mot_decoder=mot_decoder,
                    enable_rob_dec=True,
                    enable_mot_dec=True,
                    embed_dim=CFG.embed_dim,
                    auto_reg=CFG.auto_reg,
                    use_coll_loss=CFG.use_coll_loss,
                    lr=CFG.learning_rate,
                    optimizer=CFG.optimizer,
                    weight_decay=CFG.weight_decay)

if CFG.freeze_enc:
    print("Encoder parameters frozen")
    model.rgb_encoder.requires_grad_(False)
    model.lidar_encoder.requires_grad_(False)
    model.mot_decoder.requires_grad_(False)
    model.enable_rob_dec=True
    model.enable_mot_dec=True

datamodel = NavSetDataModule(save_data_path=CFG.save_data_path,
                             train_rosbag_path=CFG.train_rosbag_path,
                             val_rosbag_path=CFG.val_rosbag_path,
                             test_rosbag_path=CFG.val_rosbag_path,
                             batch_size=CFG.batch_size,
                             num_workers=CFG.num_workers,
                             pin_memory=CFG.pin_memory)

torch.set_float32_matmul_precision("high")

num_gpus = torch.cuda.device_count()
trainer = Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy='ddp_find_unused_parameters_true',
    logger=pl_loggers.TensorBoardLogger("logs/"),
    callbacks=[model_checkpoint_cb], #early_stopping_cb],
    gradient_clip_val=1.0,
    max_epochs=CFG.epochs,
    log_every_n_steps=20)

print("Starting training!!!")

trainer.fit(model, datamodel)

print('Model has been trained and saved!')