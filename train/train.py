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
    filename='mot_train_2_no_earlystop'+datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
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
                    drop_path_rate=CFG.drop_path_rate
                    )

mot_traj_decoder = TransformerDecoder(
                    embed_dim=CFG.embed_dim,
                    depth=CFG.depth,
                    num_heads=CFG.num_heads,
                    drop_rate=CFG.drop_rate,
                    attn_drop_rate=CFG.attn_drop_rate,
                    drop_path_rate=CFG.drop_path_rate,
                    multi=True
                    )

model = AttnNav(rgb_encoder=rgb_encoder,
                lidar_encoder=lidar_encoder,
                rob_traj_decoder=rob_traj_decoder,
                mot_decoder=mot_traj_decoder,
                enable_rob_dec=False,
                enable_mot_dec=True,
                embed_dim=CFG.embed_dim,
                lr=CFG.learning_rate,
                optimizer=CFG.optimizer,
                weight_decay=CFG.weight_decay)

datamodel = NavSetDataModule(save_data_path=CFG.save_data_path,
                             train_rosbag_path=CFG.train_rosbag_path,
                             val_rosbag_path=CFG.val_rosbag_path,
                             batch_size=CFG.batch_size,
                             num_workers=CFG.num_workers,
                             pin_memory=CFG.pin_memory)

num_gpus = torch.cuda.device_count()
trainer = Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy='ddp',
    logger=pl_loggers.TensorBoardLogger("logs/"),
    callbacks=[model_checkpoint_cb],
    gradient_clip_val=1.0,
    max_epochs=CFG.epochs,
    log_every_n_steps=20)

print("Starting training!!!")

trainer.fit(model, datamodel)


torch.save(
    model.state_dict(),
    'trained_models/' + 'mot_train_2_no_earlystop' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')

print('Model has been trained and saved!')