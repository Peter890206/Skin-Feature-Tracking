import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pos_embed import get_2d_sincos_pos_embed

from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.swin_transformer import PatchMerging
from timm.models.swin_transformer import window_partition, window_reverse
from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath
from timm.models.registry import register_model
from torch.utils.checkpoint import checkpoint


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     mask = mask.reshape(B_, 1, 1, N)
        #     mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(2, 3)
        #     mask_new = 1 - mask_new
        #     if mask_new.dtype == torch.float16:
        #         attn = attn - 65500 * mask_new
        #     else:
        #         attn = attn - 1e30 * mask_new
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixMIMBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # if attn_mask is not None:
        #     attn_mask = attn_mask.repeat(B, 1, 1)   # B, N, 1
        #     attn_mask = attn_mask.view(B, H, W, 1)
        #     attn_mask = window_partition(attn_mask, self.window_size)
        #     attn_mask = attn_mask.view(-1, self.window_size * self.window_size, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MixMIMLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        # print("-------------------------")
        # print("dim = ",self.dim)
        # print("-------------------------")
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMIMBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads, 
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                    norm_layer=norm_layer)
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class MixMIM(nn.Module):
    def __init__(self, mlp_ratio=4, norm_pix_loss=True, 
                 img_size=32, patch_size=4, in_chans=3, num_classes=0,
                 embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7], qkv_bias=True, qk_scale=None, patch_norm=True,
                 drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0, 
                 norm_layer=nn.LayerNorm, use_checkpoint=False, range_mask_ratio=0.0, **kwargs):
        super().__init__()
       
        # encoder args
        self.encoder_stride = 32
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # reconstruction args
        self.norm_pix_loss = norm_pix_loss
        self.range_mask_ratio = range_mask_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(MixMIMLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(self.patch_grid[0] // (2 ** i_layer), self.patch_grid[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint)
            )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)

        num_patches = self.patch_embed.num_patches
        out_num_patches = (self.img_size // self.encoder_stride) ** 2
        self.out_num_patches = out_num_patches
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim), requires_grad=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


@register_model
def swin_transformer_base(**kwargs):
    default_args = dict(
        img_size=32, patch_size=4, in_chans=3, num_classes=0,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=[14, 14, 14, 7], mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, drop_path_rate=0.0, 
        patch_norm=True, use_checkpoint=False,
    )
    default_args.update(**kwargs)
    model = MixMIM(**default_args)

    return model


@register_model
def swin_transformer_large(**kwargs):
    default_args = dict(
        img_size=64, patch_size=4, in_chans=3, num_classes=0,
        embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
        window_size=[14, 14, 14, 7], mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, drop_path_rate=0.0, 
        patch_norm=True, use_checkpoint=False,
        decoder_dim=512, decoder_depth=8, decoder_num_heads=16,
    )
    default_args.update(**kwargs)
    model = MixMIM(**default_args)

    return model


@register_model
def swin_transformer_huge(**kwargs):
    default_args = dict(
        img_size=32, patch_size=4, in_chans=3, num_classes=0,
        embed_dim=352, depths=[2, 2, 18, 2], num_heads=[11, 22, 44, 88],
        window_size=[14, 14, 14, 7], mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, drop_path_rate=0.0, 
        patch_norm=True, use_checkpoint=False,
        decoder_dim=512, decoder_depth=8, decoder_num_heads=16,
    )
    default_args.update(**kwargs)
    model = MixMIM(**default_args)

    return model