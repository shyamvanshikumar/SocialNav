from models.dataloader import NavSet, NavSetDataModule
from models.encoder import VisionTransformer
from models.decoder import TransformerDecoder
from models.model import AttnNav
from torch.utils.data import DataLoader, ConcatDataset
from train import config as CFG
import numpy as np

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

model = AttnNav.load_from_checkpoint("./trained_models/rob_train_scaled_l2_coll_loss23-03-2023-23-04-46.ckpt",
                                     #"./trained_models/rob_train_3sec_spread_pose04-03-2023-19-23-47.ckpt",
                                     rgb_encoder=rgb_encoder,
                                     lidar_encoder=lidar_encoder,
                                     rob_traj_decoder=rob_traj_decoder,
                                     mot_decoder=mot_decoder,
                                     )
model.enable_rob_dec = True
model.eval()
model.to_device('cpu')

from models.generator import Generator
generator = Generator(model,beam_size=10)

def get_path(rgb, lidar):
    generated_paths_tensor = generator.generate_path(rgb, lidar).detach()
    generated_paths = generated_paths_tensor.numpy()
    path = generator.select_one(generated_paths_tensor)
    return path.detach().numpy()