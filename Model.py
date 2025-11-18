import torch , torch.nn as nn, torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.enc1 = self._block_3d(self.in_channels, 32, use_dropout=False)
        self.enc2 = self._block_3d(32, 64, use_dropout=False)
        self.enc3 = self._block_3d(64, 128, use_dropout=False)

        self.bottleneck = self._block_3d(128, 256, use_dropout=True)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block_3d(256, 128, use_dropout=True)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block_3d(128, 64, use_dropout=True)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block_3d(64, 32, use_dropout=True)

        self.final = nn.Conv3d(32, self.out_channels, kernel_size=1)

    def _block_3d(self, in_channels, features, use_dropout=False):
        """Blocco conv 3D con BatchNorm e ReLU. Dropout3d opzionale."""
        layers = [
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(p=self.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool3d(enc3, 2))

        up3 = self.up3(bottleneck)
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))

        up2 = self.up2(dec3)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        return self.final(dec1)