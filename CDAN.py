import torch
from torch import nn
from torch.nn import functional as F
n_feat = 256
kernel_size = (3, 3)


class _Res_Block(nn.Module):

    def __init__(self, c1=256, r=16, k=7):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.t = 2
        c_ = int(c1 // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c_, c1, kernel_size=1)
        )
        assert k & 1, 'The size of the convolution kernel must be odd.'

        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2)

    def forward(self, x):

        for i in range(self.t):
            if i == 0:
                x1 = self.relu(self.res_conv(x))
            x1 = self.relu(self.res_conv(x+x1))

        y = self.res_conv(x1)
        y *= 0.13

        y = torch.add(y, x)

        # Channel Attention

        ca = torch.cat([
            F.adaptive_avg_pool2d(y, 1),
            F.adaptive_max_pool2d(y, 1)
        ], dim=3)

        ca = ca.sum(dim=3, keepdims=True)

        ca1 = self.mlp(ca)
        ca = torch.add(ca, ca1)
        ca = torch.sigmoid(ca)

        y = ca * y

        # Spatial Attention

        y = self.relu(self.res_conv(y))

        sa = torch.sigmoid(self.conv(torch.cat([
            y.mean(dim=1, keepdims=True),
            y.max(dim=1, keepdims=True)[0]
        ], dim=1)))
        y = sa * y

        y = torch.add(y, x)

        return y



class CDAN(nn.Module):

    def __init__(self, scale):
        super(CDAN, self).__init__()

        in_ch = 1
        num_blocks = 32

        self.conv1 = nn.Conv2d(in_ch, n_feat, kernel_size, padding=1)
        self.conv_u = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.conv_up = nn.Conv2d(n_feat, n_feat * 4, kernel_size, padding=1)

        self.body = self.make_layer(_Res_Block, num_blocks)

        if scale == 2:
            self.conv_out = nn.Conv2d(256, in_ch, kernel_size, padding=1)
            self.upsample = nn.Sequential(self.conv_up, nn.PixelShuffle(2))

        if scale == 4:
            self.conv_out = nn.Conv2d(64, in_ch, kernel_size, padding=1)
            self.upsample = nn.Sequential(self.conv_up, nn.PixelShuffle(4))

    # 32 recurrent convolutional attention modules (RCAM)

    def make_layer(self, block, layers):

        res_block = []
        for _ in range(layers):
            res_block.append(block())

        return nn.Sequential(*res_block)

    def forward(self, x):

        out1 = self.conv1(x)

        out = self.body(out1)

        out = torch.add(out1, out)

        out = self.upsample(out)

        out = self.conv_out(out)

        return out


if __name__ == '__main__':

    x = torch.randn(2, 1, 32, 32)
    net = CDAN(scale=4)
    print(net(x).shape)
