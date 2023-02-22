import torch

save_data_path = '/workspace/project/data/'
train_rosbag_path = '/workspace/project/data/train_bags/'
val_rosbag_path = '/workspace/project/data/val_bags/'

# dataloader config
batch_size = 6
num_workers = 8
pin_memory = True
use_weighted_sampling = False

# image encoder config
img_size = 240
patch_size = 8
embed_dim = 256
depth = 8
num_heads = 8
drop_rate = 0.1
attn_drop_rate = 0.1
drop_path_rate = 0.1

# both encoders
output_dim = 128

# optimizer config
optimizer = 'AdamW'
learning_rate = 5e-4
weight_decay = 0.02
temperature = 0.07
patience = 2
factor = 0.5

# training parameters
epochs = 75

# gpu vs cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')