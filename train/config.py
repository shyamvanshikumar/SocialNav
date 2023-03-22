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

#decoder auto-reg determines whether to output all co-ordinates at once or output one at a time
auto_reg = True

#use collision loss
use_coll_loss = True

# optimizer config
optimizer = 'AdamW'
learning_rate = 5e-4
weight_decay = 0.02
temperature = 0.07
patience = 2
factor = 0.5

# training parameters
epochs = 35
freeze_enc = True
ckp_path = "/workspace/project/trained_models/mot_train_2_no_earlystop22-02-2023-21-59-00.ckpt"

# gpu vs cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')