# UNet code from https://github.com/milesial/Pytorch-UNet/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 1))#n_classes))

        n_feat = 64
        self.n_feat = n_feat
        self.timeembed1 = EmbedFC(1, 16*n_feat)
        self.timeembed2 = EmbedFC(1, 8*n_feat)
        self.timeembed3 = EmbedFC(1, 4*n_feat)
        self.timeembed4 = EmbedFC(1, 2*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 16*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 8*n_feat)
        self.contextembed3 = EmbedFC(n_classes, 4*n_feat)
        self.contextembed4 = EmbedFC(n_classes, 2*n_feat)

    def forward(self, x, t, ctx, context_mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ctx = nn.functional.one_hot(ctx, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        ctx = ctx * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(ctx).view(-1, self.n_feat * 16, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 16, 1, 1)
        cemb2 = self.contextembed2(ctx).view(-1, self.n_feat * 8, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 8, 1, 1)
        cemb3 = self.contextembed3(ctx).view(-1, self.n_feat * 4, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat * 4, 1, 1)
        cemb4 = self.contextembed4(ctx).view(-1, self.n_feat * 2, 1, 1)
        temb4 = self.timeembed4(t).view(-1, self.n_feat * 2, 1, 1)

        x = self.up1(x5*cemb1 + temb1, x4)
        x = self.up2(x*cemb2 + temb2, x3)
        x = self.up3(x*cemb3 + temb3, x2)
        x = self.up4(x*cemb4 + temb4, x1)
        # import ipdb; ipdb.set_trace()
        logits = self.outc(x)
        return logits