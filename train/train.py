from datetime import datetime
import torch
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint

from models.model import AttnNav
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
    filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
    monitor='val_loss',
    mode='min')

model = AttnNav(img_size=CFG.img_size,
                patch_size=CFG.patch_size,
                embed_dim=CFG.embed_dim,
                depth=CFG.depth,
                num_heads=CFG.num_heads,
                drop_rate=CFG.drop_rate,
                attn_drop_rate=CFG.attn_drop_rate,
                drop_path_rate=CFG.drop_path_rate,
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
    callbacks=[model_checkpoint_cb, swa_cb, early_stopping_cb],
    gradient_clip_val=1.0,
    max_epochs=CFG.epochs,
    log_every_n_steps=20)

print("Starting training!!!")

trainer.fit(model, datamodel)

print('Model has been trained and saved!')