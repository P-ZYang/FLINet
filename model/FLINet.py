import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from thop import profile
from torch.nn import MaxPool2d

from model.L_LF import L_LF

from model.G_LF import G_LF

from model.DAF import DAF

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class GLF(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv3_3 = nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1, groups=in_channels//2)
        self.conv5_5 = nn.Conv2d(in_channels//2, in_channels, kernel_size=5, padding=2, groups=in_channels//2)
        self.fuzzy = nn.Sequential(
            L_LF(in_channels, fuzzynum=in_channels//4, fuzzychannel=in_channels//4),
            nn.ReLU(inplace=True),
        )

        self.ma = nn.Sequential(
            G_LF(in_channels, fuzzynum=in_channels//4, fuzzychannel=in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.eca = eca_layer_2d(out_channels)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        f = self.fuzzy(self.conv3_3(x1))
        g = self.ma(self.conv5_5(x2))
        f_g = torch.concat([f, g], dim=1)
        out = self.conv1(f_g)
        return self.eca(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.fusion = DAF(in_channels // 2)
        self.nConvs = self._make_nConv(in_channels, out_channels, nb_Conv)

    def forward(self, x, skip_x):
        up = self.up(x)

        skip_x_att = self.fusion(up, skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)

    def _make_nConv(self, in_channels, out_channels, nb_Conv):
        layers = []
        layers.append(CBN(in_channels, out_channels))

        for _ in range(nb_Conv - 1):
            layers.append(CBN(out_channels, out_channels))
        return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Ours(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, deepsuper=True, mode='train'):
        super(Ours, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_channels = 32
        self.n_classes = 1
        self.deepsuper = deepsuper
        self.mode = mode

        self.pool = MaxPool2d(2)
        self.inc = self._make_layer(Res_block, self.in_ch, self.in_channels, 1)
        self.down_encoder1 = self._make_layer(Res_block, self.in_channels, self.in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(Res_block, self.in_channels * 2, self.in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(Res_block, self.in_channels * 4, self.in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(Res_block, self.in_channels * 8, self.in_channels * 8, 1)

        self.GLF1 = GLF(in_channels=self.in_channels, out_channels=self.in_channels)
        self.GLF2 = GLF(in_channels=self.in_channels * 2, out_channels=self.in_channels * 2)
        self.GLF3 = GLF(in_channels=self.in_channels * 4, out_channels=self.in_channels * 4)
        self.GLF4 = GLF(in_channels=self.in_channels * 8, out_channels=self.in_channels * 8)

        self.up_encoder4 = UpBlock(in_channels=self.in_channels * 16, out_channels=self.in_channels * 4, nb_Conv=2)
        self.up_encoder3 = UpBlock(in_channels=self.in_channels * 8, out_channels=self.in_channels * 2, nb_Conv=2)
        self.up_encoder2 = UpBlock(in_channels=self.in_channels * 4, out_channels=self.in_channels, nb_Conv=2)
        self.up_encoder1 = UpBlock(in_channels=self.in_channels * 2, out_channels=self.in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(self.in_channels, self.n_classes, kernel_size=1, stride=1)

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(self.in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(self.in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(self.in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(self.in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        # print("x1:", x1.shape)

        x2 = self.down_encoder1(self.pool(x1))
        # print("x2:", x2.shape)

        x3 = self.down_encoder2(self.pool(x2))
        # print("x3:", x3.shape)

        x4 = self.down_encoder3(self.pool(x3))
        # print("x4:", x4.shape)

        # print("x5:", x5.shape)
        d5 = self.down_encoder4(self.pool(x4))

        # print("d5:", d5.shape)
        # print("d5:", d5.shape)

        f1 = self.GLF1(x1) + x1
        # print("f1:", f1.shape)

        f2 = self.GLF2(x2) + x2
        # print("f2:", f2.shape)

        f3 = self.GLF3(x3) + x3
        # print("f3:", f3.shape)

        f4 = self.GLF4(x4) + x4
        # print("f4:", f4.shape)

        d4 = self.up_encoder4(d5, f4)
        # print("d4:", d4.shape)
        d3 = self.up_encoder3(d4, f3)
        # print("d3:", d3.shape)
        d2 = self.up_encoder2(d3, f2)
        # print("d2:", d2.shape)
        d1 = self.up_encoder1(d2, f1)
        # print("d1:", d1.shape)
        out = self.outc(d1)
        # print("out:", out.shape)

        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2),
                        torch.sigmoid(out), torch.sigmoid(d0))

            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = Ours(mode='train', deepsuper=False)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
