import torch
import torch.nn as nn
import math

class UpScale(nn.Module):
    def __init__(self, n_in, n_out):
        super(UpScale, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
        )

    def forward(self, input):
        return self.upscale(input)

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            UpScale(512, 256),
            UpScale(256, 128),
            UpScale(128, 64),
            UpScale(64, 1),
        )
        self.decoder2 = nn.Sequential(
            UpScale(512, 256),
            UpScale(256, 128),
            UpScale(128, 64),
            UpScale(64, 1),
        )
        self.decoder3 = nn.Sequential(
            UpScale(512, 256),
            UpScale(256, 128),
            UpScale(128, 64),
            UpScale(64, 1),
        )
        self._initialize_weights()

    def forward(self, x):
        tmp_M1 = self.encoder1(x)
        M1 = self.decoder1(tmp_M1)
        M2 = self.decoder2(tmp_M1)
        M3 = self.decoder3(tmp_M1)
        return M1, M2, M3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d or nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
