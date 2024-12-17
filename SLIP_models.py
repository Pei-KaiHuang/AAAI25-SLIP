import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import clip
from PIL import Image
import numpy as np
from einops import rearrange
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # from (512, 7, 7) to (512, 32, 32)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=6, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU()
        )
        # from (512, 32, 32) to (1, 32, 32)
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.reduce_channels(x)
        return x
    
class EntangleModel(nn.Module):
    def __init__(self):
        super(EntangleModel, self).__init__()
        # from (1024, 7, 7) to (512, 7, 7)
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_live, z_spoof):
        z_ent = torch.cat((z_live, z_spoof), dim = 1)

        z_ent = self.module(z_ent)

        return z_ent

class SLIP(nn.Module):
    def __init__(self):
        super(SLIP, self).__init__()

        self.clipModel, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
        clip.model.convert_weights(self.clipModel)

        self.decoder = Decoder()
        self.entangle = EntangleModel()

    def expandSize(self, feature):
        output_feature = feature.unsqueeze(2).unsqueeze(3)
        output_feature = output_feature.expand(-1, -1, 7, 7)

        return output_feature

    def forward(self, I, T1, T2, T3, T4, update = "I"):
        if update == "I":
            self.clipModel.visual.requires_grad = True
            self.clipModel.transformer.requires_grad = False
            self.decoder.requires_grad = True
            self.entangle.requires_grad = False
        elif update == "T":
            self.clipModel.visual.requires_grad = False
            self.clipModel.transformer.requires_grad = True
            self.decoder.requires_grad = False
            self.entangle.requires_grad = True

        s = I.shape[0]
        image_features, z_hat = self.clipModel.encode_image(I)
        z_hat = rearrange(z_hat, 'b h w c -> b c h w') # (8, 512, 7, 7)
        
        image_features = 0.5 * (self.expandSize(image_features) + z_hat)
        text_features1 = 0.5 * (self.expandSize(self.clipModel.encode_text(T1)) + z_hat)
        text_features2 = 0.5 * (self.expandSize(self.clipModel.encode_text(T2)) + z_hat)
        text_features3 = 0.5 * (self.expandSize(self.clipModel.encode_text(T3)) + z_hat)
        text_features4 = 0.5 * (self.expandSize(self.clipModel.encode_text(T4)) + z_hat) # (8, 512, 7, 7)

        image_features_normalized = F.normalize(image_features, p=2, dim=1).float()
        text_features1_normalized = F.normalize(text_features1, p=2, dim=1).float()
        text_features2_normalized = F.normalize(text_features2, p=2, dim=1).float()
        text_features3_normalized = F.normalize(text_features3, p=2, dim=1).float()
        text_features4_normalized = F.normalize(text_features4, p=2, dim=1).float()

        text_features_ent = self.entangle(text_features1_normalized, text_features2_normalized)
        image_features_ent = self.entangle(image_features_normalized, text_features2_normalized)

        image_map = self.decoder(image_features_normalized)
        text_map1 = self.decoder(text_features1_normalized)
        text_map2 = self.decoder(text_features2_normalized)
        image_map_ent = self.decoder(image_features_ent)

        return image_map, text_map1, text_map2, image_features_normalized, text_features1_normalized, text_features2_normalized, text_features3_normalized, text_features4_normalized, text_features_ent, image_map_ent
