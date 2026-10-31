import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """Independent per-mode channel mixing in two disjoint rfft2 row bands."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights_positive = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights_negative = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        if H < 2:
            raise ValueError('Need at least two rows for disjoint Fourier bands')
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.weights_positive.shape[1], H, W // 2 + 1, dtype=x_ft.dtype, device=x.device)
        m1 = min(self.modes, H // 2)
        m2 = min(self.modes, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum('bchw,cohw->bohw', x_ft[:, :, :m1, :m2], self.weights_positive[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum('bchw,cohw->bohw', x_ft[:, :, -m1:, :m2], self.weights_negative[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))

class FNOBlock(nn.Module):

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))

class UNetEncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.block(x)

class UNetDecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = nn.Sequential(nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class FNO2DLightning(nn.Module):

    def __init__(self, in_channels, width=128, modes=24, unet_base=32):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.fno1 = FNOBlock(width, modes)
        self.fno2 = FNOBlock(width, modes)
        self.fno3 = FNOBlock(width, modes)
        self.fno4 = FNOBlock(width, modes)
        b = unet_base
        self.enc1 = UNetEncoderBlock(width, b)
        self.enc2 = UNetEncoderBlock(b, b * 2)
        self.enc3 = UNetEncoderBlock(b * 2, b * 4)
        self.enc4 = UNetEncoderBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = UNetDecoderBlock(b * 8, b * 4, b * 4)
        self.dec2 = UNetDecoderBlock(b * 4, b * 2, b * 2)
        self.dec1 = UNetDecoderBlock(b * 2, b, b)
        fusion_ch = width + b
        self.fusion = nn.Sequential(nn.Conv2d(fusion_ch, fusion_ch // 2, 1), nn.GELU(), nn.Conv2d(fusion_ch // 2, 1, 1))

    def make_time_encoding(self, B, C, T, H, W, device):
        t = torch.linspace(-1, 1, T, device=device)
        t = t.view(1, T, 1, 1)
        t = t.repeat(B, 1, H, W)
        t = t.unsqueeze(1)
        t = t.repeat(1, C, 1, 1, 1)
        t = t.reshape(B, C * T, H, W)
        return t

    def make_pos_encoding(self, B, H, W, device):
        gx = torch.linspace(-1, 1, H, device=device)
        gy = torch.linspace(-1, 1, W, device=device)
        gx = gx.view(1, 1, H, 1).repeat(B, 1, 1, W)
        gy = gy.view(1, 1, 1, W).repeat(B, 1, H, 1)
        return (gx, gy)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
        t = self.make_time_encoding(B, C, T, H, W, x.device)
        x = torch.cat([x, t], dim=1)
        gx, gy = self.make_pos_encoding(B, H, W, x.device)
        x = torch.cat([x, gx, gy], dim=1)
        x = self.fc0(x)
        fno = self.fno1(x)
        fno = self.fno2(fno)
        fno = self.fno3(fno)
        fno = self.fno4(fno)
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.enc4(self.pool(s3))
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        fused = torch.cat([fno, d1], dim=1)
        return self.fusion(fused)
