"""U-Net architecture implemented from scratch in PyTorch for binary medical image segmentation."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive [Conv2D -> BatchNorm -> ReLU] blocks.

    Padding=1 ensures that spatial dimensions (H, W) are preserved across convolutions.
    Bias is disabled because BatchNorm provides an affine bias parameter.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling block: MaxPool2d (halves spatial resolution) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_and_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_and_conv(x)


class Up(nn.Module):
    """Upscaling block: Transposed Convolution (or bilinear upsample), feature concatenation, and DoubleConv.

    Recovers fine spatial details by concatenating the high-resolution skip connection
    from the contracting encoder with the upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Args:

        x1: Deep feature map from current decoder level.
        x2: Shallow skip connection from corresponding encoder level.
        """
        x1 = self.up(x1)

        # Handle potential small size differences due to non-power-of-2 dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for 2D semantic image segmentation.

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB).
        out_channels: Number of output channels (1 for binary nuclei segmentation).
        init_features: Number of base feature channels (default: 32).
        bilinear: If True, uses bilinear interpolation instead of transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        bilinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.bilinear = bilinear

        f = init_features
        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)          # e.g. 3 -> 32
        self.down1 = Down(f, f * 2)                    # 32 -> 64
        self.down2 = Down(f * 2, f * 4)                # 64 -> 128
        self.down3 = Down(f * 4, f * 8)                # 128 -> 256

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, (f * 16) // factor)   # 256 -> 512 (or 256 if bilinear)

        # Expanding Path (Decoder with Skip Connections)
        self.up1 = Up(f * 16, (f * 8) // factor, bilinear)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)

        # Final projection to class logits (no activation)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits of shape (B, out_channels, H, W)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise probabilities in range [0, 1] using sigmoid activation."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Compute hard binary mask {0.0, 1.0} at given decision threshold."""
        prob = self.predict_probability(x)
        return (prob > threshold).float()

    def count_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_output = model(dummy_input)
    print(f"U-Net initialized successfully.")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(dummy_output.shape)}")
    assert dummy_output.shape == (1, 1, 256, 256), f"Shape mismatch: {dummy_output.shape}"
    print("Dummy forward test PASSED!")
