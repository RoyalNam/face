import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, n_res=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        )

        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)
                )
            )
            curr_dim *= 2

        for i in range(n_res):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        for i in range(2):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)
                )
            )
            curr_dim //= 2

        layers.append(
            nn.Sequential(
                nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
                nn.Tanh()
            )
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], 1)
        return self.main(x)


if __name__ == '__main__':
    x = torch.randn((1, 3, 128, 128))
    c = torch.randn(1, 5, )
    gen = Generator()
    y = gen(x, c)
    assert y.size() == x.size()
    print(f'Output size: {y.size()}')
    # summary(gen, [(3, 128, 256), (5, 1, 1)])


