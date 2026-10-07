#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch
from torch.nn import functional as F

def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False, in_channels=3):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False, in_channels=3):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        # Estimate a spatial attention map from channel-wise mean and max responses.
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class PDNMediumCoordinateFusion(nn.Module):

    def __init__(self, out_channels=384, padding=False, coord_channels=4):
        super().__init__()
        pad_mult = 1 if padding else 0

        # RGB feature stem.
        self.rgb_block = nn.Sequential(
            nn.Conv2d(3, out_channels=256, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        )

        # Coordinate-field embedding branch.

        self.coord_embed = nn.Sequential(
            nn.Conv2d(coord_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpatialAttention(kernel_size=7)
        )

        # Fuse RGB and coordinate features before the remaining PDN blocks.
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels=288, out_channels=512, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1 * pad_mult),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x, coords, coord_scale=1.0):
        feat_rgb = self.rgb_block(x)
        h, w = feat_rgb.shape[2], feat_rgb.shape[3]

        coords_resized = F.interpolate(coords, size=(h, w), mode='area')
        feat_coord = self.coord_embed(coords_resized)


        feat_fused = torch.cat([feat_rgb, feat_coord], dim=1)

        out = self.fusion_block(feat_fused)
        return out

def get_pdn_medium_coord(out_channels=384, padding=False, coord_channels=4):
    return PDNMediumCoordinateFusion(out_channels, padding, coord_channels)


class CoordinateConditionedAutoencoder(nn.Module):
    def __init__(self, out_channels=384, coord_channels=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=8)
        )

        self.up_3 = nn.Upsample(size=3, mode='bilinear')
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, 4, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout2d(0.2))

        self.up_8 = nn.Upsample(size=8, mode='bilinear')
        self.conv_8 = nn.Sequential(nn.Conv2d(64, 64, 4, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout2d(0.2))

        self.up_15 = nn.Upsample(size=15, mode='bilinear')
        self.conv_15 = nn.Sequential(nn.Conv2d(64, 64, 4, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout2d(0.2))

        self.embed_32 = nn.Sequential(
            nn.Conv2d(coord_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.embed_63 = nn.Sequential(
            nn.Conv2d(coord_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.embed_127 = nn.Sequential(
            nn.Conv2d(coord_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


        self.up_32 = nn.Upsample(size=32, mode='bilinear')
        # Coordinate-conditioned decoder block at 32x32 resolution.
        self.fuse_32 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=1, padding=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up_63 = nn.Upsample(size=63, mode='bilinear')
        # Coordinate-conditioned decoder block.
        self.fuse_63 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=1, padding=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up_127 = nn.Upsample(size=127, mode='bilinear')
        # Coordinate-conditioned decoder block.
        self.fuse_127 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=1, padding=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.up_56 = nn.Upsample(size=56, mode='bilinear')
        self.conv_56 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, x, coords, coord_scale=1.0):
        feat = self.encoder(x)

        feat = self.conv_3(self.up_3(feat))
        feat = self.conv_8(self.up_8(feat))
        feat = self.conv_15(self.up_15(feat))

        # Coordinate injection at 32x32.
        feat = self.up_32(feat)
        c_32 = self.embed_32(F.interpolate(coords, size=(32, 32), mode='area'))

        feat = self.fuse_32(feat + c_32 * coord_scale)

        # Coordinate injection at 63x63.
        feat = self.up_63(feat)
        c_63 = self.embed_63(F.interpolate(coords, size=(63, 63), mode='area'))

        feat = self.fuse_63(feat + c_63 * coord_scale)

        # Coordinate injection at 127x127.
        feat = self.up_127(feat)
        c_127 = self.embed_127(F.interpolate(coords, size=(127, 127), mode='area'))

        feat = self.fuse_127(feat + c_127 * coord_scale)

        feat = self.up_56(feat)
        out = self.conv_56(feat)
        return out

def get_coord_autoencoder(out_channels=384, coord_channels=4):
    return CoordinateConditionedAutoencoder(out_channels, coord_channels)