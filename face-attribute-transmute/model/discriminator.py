import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=128, conv_dim=64, c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(Block(3, conv_dim))

        curr_dim = conv_dim
        for i in range(1, n_strided):
            layers.append(Block(curr_dim, curr_dim*2))
            curr_dim *= 2

        kernel_size = img_size // 2**n_strided
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), -1)


if __name__ == '__main__':
    x = torch.randn((3, 128, 128))
    disc = Discriminator(img_size=128)
    out_src, out_cls = disc(x)
    print(f'out_src size: {out_src.size()}')
    print(f'out_cls size: {out_cls.size()}')

    # summary(disc, (3, 256, 256))


