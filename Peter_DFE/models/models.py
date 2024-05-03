import torch
import torch.nn as nn    #torch version 2.0.1
from .EfficientNet_model import efficientnet_b0
from .EfficientNetV2_model import efficientnetv2_s
from .Resnet34_model import resnet34, resnet50
from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath
from .swin_transformer_v2 import SwinTransformerV2
from .Swin_transformer_model import swin_transformer_base
from .ConvNeXt_model import convnext_base


class DFE_Model(nn.Module):
    def __init__(self):
        super(DFE_Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=7)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=7)

        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=7)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=7)
        self.conv_trans3 = nn.ConvTranspose2d(32, 32, kernel_size=7)
        self.conv_trans4 = nn.ConvTranspose2d(32, 16, kernel_size=7)
        self.conv_trans5 = nn.ConvTranspose2d(16, 3, kernel_size=7)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x
    
class EfficientNet_Decoder(nn.Module):
    def __init__(self):
        super(EfficientNet_Decoder, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(1280, 512, kernel_size=6)
        self.conv_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.conv_trans3 = nn.ConvTranspose2d(256, 64, kernel_size=6)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=6)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=6)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=7)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv_trans1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans6(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x
    
class EfficientNetV2_Decoder(nn.Module):
    def __init__(self):
        super(EfficientNetV2_Decoder, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(1280, 512, kernel_size=6)
        self.conv_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.conv_trans3 = nn.ConvTranspose2d(256, 64, kernel_size=6)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=6)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=6)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=7)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self.conv_trans1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans6(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x
    
class Resnet34_Decoder(nn.Module):
    def __init__(self):
        super(Resnet34_Decoder, self).__init__()

        self.conv_trans0 = nn.ConvTranspose2d(1024, 512, kernel_size=1)
        self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=6)    #self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=6)    #self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=6)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=6)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=6)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=6)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=7)

        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x = self.conv_trans0(x)
        # x = self.bn0(x)
        # x = self.relu(x)
        x = self.conv_trans1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans6(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x

class Resnet50_Decoder(nn.Module):
    def __init__(self):
        super(Resnet50_Decoder, self).__init__()

        self.conv_trans0 = nn.ConvTranspose2d(2048, 1024, kernel_size=5)
        self.conv_trans1 = nn.ConvTranspose2d(1024, 256, kernel_size=5)    #self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=5)    #self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=6)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=5)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=6)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=6)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=6)

        self.bn0 = nn.BatchNorm2d(1024)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv_trans0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv_trans1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans6(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x




class Swin_transformer_Decoder(nn.Module):
    def __init__(self):
        super(Swin_transformer_Decoder, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=6)
        self.conv_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.conv_trans3 = nn.ConvTranspose2d(256, 64, kernel_size=6)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=6)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=6)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=7)

        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze()
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        # print(x.shape)
        
        x = self.conv_trans1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv_trans4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv_trans5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv_trans6(x)
        x = self.bn6(x)
        # x = self.sigmoid(x)
        
        return x
    
class ViT_based_decoder(nn.Module):
    def __init__(self, num_features, decoder_dim=512, decoder_depth=8, decoder_num_heads=16, 
                 mlp_ratio=4, img_size=32, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        # decoder args
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.encoder_stride = 32
        self.num_features = num_features
        out_num_patches = (img_size // self.encoder_stride) ** 2
        self.out_num_patches = out_num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, out_num_patches, decoder_dim), requires_grad=False)

        self.decoder_embed = nn.Linear(self.num_features, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim,
            self.encoder_stride ** 2 * 3
        )

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder_stride
        # h = w = int(x.shape[1]**.5)
        h = w = 1
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        x = x.squeeze()
        # print(x.shape)
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x.squeeze()
        # print(x.shape)
        
        x = self.unpatchify(x)
        
        return x


class DFE_EfficientNet(nn.Module):
    def __init__(self):
        super(DFE_EfficientNet, self).__init__()
        self.encoder = efficientnet_b0(num_classes=0)    #change the model encoder here
        self.decoder = EfficientNet_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        return x
    
class DFE_EfficientNetV2(nn.Module):
    def __init__(self):
        super(DFE_EfficientNetV2, self).__init__()
        self.encoder = efficientnetv2_s(num_classes=0)    #change the model encoder here
        self.decoder = EfficientNetV2_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        return x
    
class DFE_Resnet34(nn.Module):
    def __init__(self):
        super(DFE_Resnet34, self).__init__()
        self.encoder = resnet34(num_classes=0)    #change the model encoder here
        self.decoder = Resnet34_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        # print(x.size)
        return x
    
class DFE_Resnet50(nn.Module):
    def __init__(self):
        super(DFE_Resnet50, self).__init__()
        self.encoder = resnet50(num_classes=0)    #change the model encoder here
        self.decoder = Resnet50_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        # print(x.size)
        return x
    



class DFE_EfficientNet_vit_decoder(nn.Module):
    def __init__(self):
        super(DFE_EfficientNet_vit_decoder, self).__init__()
        self.encoder = efficientnet_b0(num_classes=0)    #change the model encoder here
        self.decoder = ViT_based_decoder(num_features = 1280, decoder_dim=512, decoder_depth=16, decoder_num_heads=16)    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x
    
class DFE_Swin_transformer(nn.Module):
    def __init__(self):
        super(DFE_Swin_transformer, self).__init__()
        self.encoder = swin_transformer_base()    #change the model encoder here
        self.decoder = Swin_transformer_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DFE_Swin_transformerv2(nn.Module):
    def __init__(self):
        super(DFE_Swin_transformerv2, self).__init__()
        self.encoder = SwinTransformerV2()    #change the model encoder here
        self.decoder = Swin_transformer_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        return x
    
class DFE_ConvNeXt(nn.Module):
    def __init__(self):
        super(DFE_ConvNeXt, self).__init__()
        self.encoder = convnext_base(num_classes=0)    #change the model encoder here
        self.decoder = Swin_transformer_Decoder()    #make sure the decoder dimensions match the encoder
    def forward(self, x):
        x = self.encoder(x)    
        x = self.decoder(x)
        return x