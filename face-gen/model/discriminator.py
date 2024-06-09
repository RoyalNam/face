import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: N x channels_img x 256 x 256
            nn.Conv2d(img_channels, features_d, 4, 2, 1),  # 128 x 128
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2),  # 64 x 64
            self._block(features_d * 2, features_d * 4),  # 32 x 32
            self._block(features_d * 4, features_d * 8),  # 16 x 16
            self._block(features_d * 8, features_d * 4),  # 8 x 8
            self._block(features_d * 4, features_d * 2),  # 4 x 4
            # After all _block img output is 4x4 (Conv below makes into 1x1)
            nn.Conv2d(
                features_d*2, 1, 4, 2, 0
            )
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    disc = Discriminator(img_channels=3)
    output = disc(x)
    print(output.shape)
    print(output)


if __name__ == '__main__':
    test()