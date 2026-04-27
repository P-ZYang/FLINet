import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class G_LF(nn.Module):
    def __init__(self, in_channel, fuzzynum, fuzzychannel, T=20) -> None:
        super().__init__()
        self.n = fuzzynum
        self.T = T

        self.conv1 = nn.Conv2d(in_channel, fuzzychannel, 3, padding=1)
        self.conv2 = nn.Conv2d(3 * fuzzychannel, in_channel, 3, padding=1)

        self.mu = nn.Parameter(torch.randn((fuzzychannel, self.n)))
        self.sigma = nn.Parameter(torch.randn((fuzzychannel, self.n)))

        self.global_alpha = nn.Parameter(torch.ones(1))
        self.global_beta = nn.Parameter(torch.ones(1))

    def global_statistics(self, x):
        global_mean = torch.mean(x, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        global_std = torch.std(x, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        global_max, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        global_min, _ = torch.min(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)


        return global_mean, global_std

    def global_membership(self, feat, global_mean, global_std):
        B, H, W, C = feat.shape

        # 将全局统计量扩展到与feat相同的空间维度
        global_mean_exp = global_mean.permute(0, 2, 3, 1).expand(B, H, W, C)
        global_std_exp = global_std.permute(0, 2, 3, 1).expand(B, H, W, C)
        # global_max_exp = global_max.permute(0, 2, 3, 1).expand(B, H, W, C)
        # global_min_exp = global_min.permute(0, 2, 3, 1).expand(B, H, W, C)

        feat_exp = feat.unsqueeze(-1).expand(-1, -1, -1, -1, self.n)

        normalized_feat = (feat_exp - global_mean_exp.unsqueeze(-1)) / (global_std_exp.unsqueeze(-1) + 1e-8)

        member = torch.exp(-((normalized_feat - self.mu) / self.sigma) ** 2)

        return member

    def forward(self, x):
        x_conv = self.conv1(x)

        global_mean, global_std = self.global_statistics(x_conv)

        feat = x_conv.permute((0, 2, 3, 1))

        member = self.global_membership(feat, global_mean, global_std)

        sample = torch.ones_like(member) * self.sigma + self.mu

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
    net = G_LF(in_channel=64, fuzzynum=16, fuzzychannel=16)
    inputs = torch.rand(1, 64, 128, 128)
    output = net(inputs)
    print(f"主输出形状: {output.shape}")

    flops, params = profile(net, inputs=(inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')