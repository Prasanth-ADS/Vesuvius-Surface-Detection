import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Basic 3D Conv Block
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
            
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------
# UNet3D (Light Version)
# -----------------------
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck (at bottom)
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2) # in: 2 (from up) + 2 (from skip) = 4

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch) # in: 1 (from up) + 1 (from skip) = 2

        # Output
        self.final_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # Base
        p1 = self.pool1(e1)     # Base/2

        e2 = self.enc2(p1)      # Base*2
        p2 = self.pool2(e2)     # Base/4

        # Bottleneck
        b = self.bottleneck(p2) # Base*4

        # Decoder
        u2 = self.up2(b)        # Base*2 (Upsampled)
        # Cat with e2 (Base*2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)       # Base (Upsampled)
        # Cat with e1 (Base)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final_conv(d1)
        return out


# -----------------------
# Test the model
# -----------------------
if __name__ == "__main__":
    model = UNet3D(in_ch=1, out_ch=1, base_ch=16)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
    assert x.shape == y.shape, "Shape mismatch!"
    print("Model OK!")
