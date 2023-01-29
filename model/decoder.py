import math
import warnings

import torch
from torch import nn, Tensor
from utils import DecAttnBlock, trunc_normal_

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class TransformerDecoder(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim

        # embeding coordinates to embed_dim Tensors
        self.linear_in_emb = nn.Linear(in_features=2, out_features=embed_dim)
        #sinusoidal positional encoding
        self.pos_embed = PositionalEncoding(embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Attention Blocks
        self.blocks = nn.ModuleList([
            DecAttnBlock(dim=embed_dim,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=dpr[i],
                      norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        self.linear_out_emb = nn.Linear(in_features=embed_dim, out_features=2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ initialize weight matrix
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, enc_output, dec_input):
        x = self.pos_embed(self.linear_in_emb(dec_input))
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, enc_output)
        x = self.linear_out_emb(self.norm(x))
        return x

