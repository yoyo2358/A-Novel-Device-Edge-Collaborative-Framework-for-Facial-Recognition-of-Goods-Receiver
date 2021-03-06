#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn


class Discriminator(nn.Module):
    """
    multi-task CNN for identity and pose classification

    ### init
    Nd : Number of identitiy to classify
    Np : Number of pose to classify

    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        D_enc_convLayers = [
            nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(), 
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, args.Nf, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(args.Nf),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.D_enc_convLayers = nn.Sequential(*D_enc_convLayers)
        
        self.fc = nn.Linear(args.Nf, args.Nd+args.Np+1)

    def forward(self, input):
        # ???????????? -> ?????????????????????????????? B x 320 x 1 x 1??????????????????
        x = self.D_enc_convLayers(input)

        # ????????????????????????????????????????????????????????????????????????
        x = x.squeeze(2)
        x = x.squeeze(2)

        # ?????????
        x = self.fc(x) # Bx320 -> B x (Nd+1+Np)

        return x

class Crop(nn.Module):
    """
    Generator ?????????????????????????????????????????? ???????????????????????????ZeroPad2d ????????????????????????????????????
    ??????????????? Tensorflow ??? padding='SAME' ???????????????????????????????????????????????????????????????
    ??????????????????????????????ZeroPad2d?????????????????????????????????????????? Crop ?????????

    ### init
    crop_list : specify the list(top, bottom, left, right) to cut data
    """

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


class Generator(nn.Module):
    """
    Encoder/Decoder conditional GAN conditioned with pose vector and noise vector

    ### init
    Np : Dimension of pose vector (Corresponds to number of dicrete pose classes of the data)
    Nz : Dimension of noise vector
    Nf : Dimension of features
    """

    def __init__(self, args):
        super(Generator, self).__init__()
        G_enc_convLayers = [
            nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, args.Nf, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6, Nf=320
            nn.BatchNorm2d(args.Nf),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1

        ]
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(args.Nf, 160, 3,1,1, bias=False), # Bx320x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x6x6 -> Bx256x13x13
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]), # Bx256x13x13 -> Bx256x12x12  
            nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x12x12 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x12x12 -> Bx192x25x25
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x24x24 -> Bx128x49x49
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x48x48 -> Bx64x97x97
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, args.channel_num,  3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_dec_fc = nn.Linear(args.Nf+args.Np+args.Nz, args.Nf*6*6)

    def forward(self, input, pose=None, noise=None, extract=False):
        x = self.G_enc_convLayers(input) # Bxchx96x96 -> Bx320x1x1
        x = x.squeeze(2)
        x = x.squeeze(2)
        if extract:
            return x
        # print([x.size(), pose.size(), noise.size()])
        x = torch.cat([x, pose, noise], 1)  # Bx320 -> B x (320+Np+Nz)
        x = self.G_dec_fc(x) # B x (320+Np+Nz) -> B x (320x6x6)
        x = x.view(-1, 320, 6, 6) # B x (320x6x6) -> B x 320 x 6 x 6
        x = self.G_dec_convLayers(x) #  B x 320 x 6 x 6 -> Bxchx96x96
        return x

