from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            Basic(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            Basic(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            Basic(in_planes, inter_planes, kernel_size=1, stride=1),
            Basic(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            Basic((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            Basic(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            Basic(in_planes, inter_planes, kernel_size=1, stride=1),
            Basic(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            Basic((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            Basic(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = Basic(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = Basic(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class L_LF(nn.Module):
    def __init__(self, in_channel, fuzzynum, fuzzychannel, T=20, window_size=3) -> None:

        super().__init__()
        self.n = fuzzynum
        self.T = T
        self.window_size = window_size
        self.padding = window_size // 2

        self.conv1 = FEM(in_channel, fuzzychannel)
        # self.conv1 = nn.Conv2d(in_channel, fuzzychannel, 3, padding=1)
        self.conv2 = nn.Conv2d(3 * fuzzychannel, in_channel, 3, padding=1)

        self.mu_high = nn.Parameter(torch.zeros((fuzzychannel, self.n)))
        self.mu_low = nn.Parameter(torch.zeros((fuzzychannel, self.n)))
        self.sigma_high = nn.Parameter(torch.ones((fuzzychannel, self.n)))
        self.sigma_low = nn.Parameter(torch.ones((fuzzychannel, self.n)))

        self.avg_pool = nn.AvgPool2d(window_size, stride=1, padding=self.padding)
        self.max_pool = nn.MaxPool2d(window_size, stride=1, padding=self.padding)

    def local_statistics(self, x):
        local_avg = self.avg_pool(x)
        local_max = self.max_pool(x)
        local_min = -self.max_pool(-x)

        local_var = self.avg_pool(x**2) - local_avg**2
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

        return local_avg, local_max, local_min, local_std

    def membership(self, feat, local_avg, local_max, local_std):

        mask_high = (feat > local_avg.permute(0, 2, 3, 1)).float()
        mask_low = 1 - mask_high

        feat_ = feat.unsqueeze(-1)
        local_max_ = local_max.permute(0, 2, 3, 1).unsqueeze(-1)
        local_avg_ = local_avg.permute(0, 2, 3, 1).unsqueeze(-1)
        local_std_ = local_std.permute(0, 2, 3, 1).unsqueeze(-1)

        # print("local_std_:", local_std_.shape)

        member_high = torch.exp(-((local_max_ - feat_) / self.sigma_high.view(1, 1, 1, -1, self.n)) ** 2)
        member_low = torch.exp(-((feat_ - local_avg_) / self.sigma_low.view(1, 1, 1, -1, self.n)) ** 2)

        local_std_norm = local_std_ / (local_std_.max() + 1e-8)
        # print("local_std_norm:", local_std_norm.shape)

        member_high = member_high * (1 + local_std_norm)
        member_low = member_low

        member = mask_high.unsqueeze(-1) * member_high + mask_low.unsqueeze(-1) * member_low
        return member

    def forward(self, x):
        x_conv = self.conv1(x)

        local_avg, local_max, local_min, local_std = self.local_statistics(x_conv)

        feat = x_conv.permute((0, 2, 3, 1))

        member = self.membership(feat, local_avg, local_max, local_std)

        sample_high = torch.ones_like(member) * self.sigma_high + self.mu_high
        sample_low = torch.ones_like(member) * self.sigma_low + self.mu_low

        mask_high = (feat > local_avg.permute(0, 2, 3, 1)).unsqueeze(-1).float()
        mask_low = 1 - mask_high

        sample = mask_high * sample_high + mask_low * sample_low

        member_and = torch.sum(
            sample * F.softmax((1 - member) * self.T, dim=4),
            dim=4
        ).permute((0, 3, 1, 2))

        member_or = torch.sum(
            sample * F.softmax(member * self.T, dim=4),
            dim=4
        ).permute((0, 3, 1, 2))

        feat_fused = torch.cat([x_conv, member_and, member_or], dim=1)
        feat_fused = self.conv2(feat_fused)

        return feat_fused

if __name__ == '__main__':


    net = L_LF(in_channel=64, fuzzynum=16, fuzzychannel=16)
    inputs = torch.rand(1, 64, 128, 128)
    output = net(inputs)
    print(f"主输出形状: {output.shape}")

    flops, params = profile(net, inputs=(inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
