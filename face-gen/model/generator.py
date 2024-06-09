import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels=3, features_g=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: N x channels_noise x 1 x 1
            self._block(noise_channels, features_g * 8, 4, 2, 1),  # 2x2
            self._block(features_g * 8, features_g * 4),  # 4x4
            self._block(features_g * 4, features_g * 2),  # 8x8
            self._block(features_g * 2, features_g),  # 16x16
            self._block(features_g, features_g),  # 32x32
            self._block(features_g, features_g),  # 64x64
            self._block(features_g, features_g),  # 128x128
            nn.ConvTranspose2d(
                features_g, img_channels, 4, 2, 1
            ),
            nn.Tanh()
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def test():
    noise_channels = 100
    x = torch.randn((1, noise_channels, 1, 1))
    gen = Generator(noise_channels=noise_channels, img_channels=3)
    output = gen(x)
    print(output.shape)
    # summary(gen, (100, 1, 1))


if __name__ == '__main__':
    test()
