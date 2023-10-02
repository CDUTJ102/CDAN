import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    def __init__(self, scale, inputChannel=1, outputChannel=1):
        super(FSRCNN, self).__init__()
        self.firstPart = nn.Sequential(
            nn.Conv2d(inputChannel, 56, kernel_size=5, padding=5 // 2),
            nn.PReLU(56)
        )
        self.midPart = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2),
            nn.PReLU(12),
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU(56),
        )
        self.lastPart = nn.Sequential(
            nn.ConvTranspose2d(56, outputChannel, kernel_size=9, stride=scale, padding=9//2, output_padding=scale-1),
        )

    def forward(self, x):
        x = self.firstPart(x)
        x = self.midPart(x)
        out = self.lastPart(x)
        return out


if __name__ == '__main__':
    x = torch.randn(10, 1, 20, 20)
    net = FSRCNN(scale=4)
    print(net(x).shape)