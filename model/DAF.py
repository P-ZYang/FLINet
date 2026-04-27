import torch
from thop import profile
from torch import nn
import torch.nn.functional as F


class DAF(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.fq = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        self.fk = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        self.fv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)

        self.sigmoid_gate = nn.Sigmoid()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            # nn.BatchNorm2d(in_ch)
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)

    def forward(self, x, skip):
        gap = torch.abs(x - skip)
        x1 = x + gap
        skip1 = skip + gap
        # print("x1:", x1.shape)
        # print("skip1:", skip1.shape)
        x_c = torch.concat([x1, skip1], dim = 1)
        # print("x_c:", x_c.shape)
        x_c = self.conv1(x_c)

        fq = self.fq(x_c)
        fk = self.fk(x_c)
        fv = self.fv(x_c)

        f_sim_tensor = torch.matmul(fq, fk.transpose(2, 3)) / (fq.size(-1) ** 0.5)
        # print("fq.size(-1) :", fq.size(-1))
        f_sum_tensor = torch.sum(f_sim_tensor, dim=(2, 3))
        scores = torch.softmax(f_sum_tensor, dim=1).unsqueeze(2).unsqueeze(3)

        out = scores * fv + x_c

        out = self.conv2(self.conv(out) + skip)

        return out


if __name__ == '__main__':
    x1 = torch.randn(1,32,256,256)
    x2 = torch.randn(1,32,256,256)
    # x = (x1,x2)
    model = DAF(32)
    # print(model(x1, x2).shape)
    flops, params = profile(model, (x1,x2))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')