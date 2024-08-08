import torch
import torch.nn as nn
from einops import rearrange

from models.vit import ViT


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x) #torch.Size([1, 256, 32, 32])

        # x: 1,128,64,64
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x) # torch.Size([1, 256, 64, 64])

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x) #torch.Size([1, 256, 32, 32])

        x = self.conv3(x)
        x = self.norm3(x) #torch.Size([1, 256, 32, 32])
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def  __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape) #torch.Size([1, 512, 16, 16])
        # exit()

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
            # print(x.shape) #torch.Size([1, 1024, 16, 16])

        x = self.layer(x)
        # print(x.shape)
        # exit()
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2) #128,256
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2) #256,512
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2) #512,1024

        self.vit_img_dim = img_dim // patch_dim #8
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False) #8,1024,1024,4,512,8,1

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape) #torch.Size([1, 128, 64, 64])

        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        # print(x2.shape) #torch.Size([1, 256, 32, 32])
        x3 = self.encoder2(x2)
        # print(x3.shape) #torch.Size([1, 512, 16, 16])

        x = self.encoder3(x3)
        # print(x.shape) #torch.Size([1, 1024, 32, 32])
        # exit()
        # print(x.shape) #torch.Size([1, 1024, 8, 8])


        x = self.vit(x)
        # print(x.shape)
        # exit()
        # print(x.shape) #torch.Size([1, 64, 1024])

        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # print(x.shape) #torch.Size([1, 1024, 8, 8])


        x = self.conv2(x)
        # print(x.shape) #torch.Size([1, 512, 8, 8])
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2) #1024,256
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels) #512,128
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2)) #256,64
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8)) #64,16

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # exit()
        x = self.decoder1(x, x3)
        # print(x.shape)
        # exit()
        x = self.decoder2(x, x2)
        # print(x.shape)
        # exit()
        x = self.decoder3(x, x1)
        # print(x.shape)
        x = self.decoder4(x)
        # print(x.shape)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x


if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=512,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=3)

    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 3, 512, 512)).shape)