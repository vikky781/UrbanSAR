"""
UrbanSAR - Dual-Branch CNN for Building Height Estimation

Architecture:
    SAR Branch (ResNet18) ──┐
                            ├── Concatenate → FC → Height (meters)
    Optical Branch (ResNet18)┘

SAR branch uses modified conv1 to handle 4-channel polarization input (HH, VV, HV, VH).
Optical branch uses standard 3-channel RGB input.
Both branches output 512-dim feature vectors, concatenated to 1024-dim,
then passed through FC layers for height regression.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SARBranch(nn.Module):
    """
    SAR feature extractor based on ResNet18.

    Modified to accept multi-polarization SAR input (default: 4 channels).
    Pretrained conv1 weights are adapted from 3→N channels by:
        1. Averaging 3-ch weights to 1 channel
        2. Repeating to match target channel count
    """

    def __init__(self, in_channels: int = 4, pretrained: bool = True):
        super().__init__()
        self.in_channels = in_channels

        # Load pretrained ResNet18
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Modify first conv layer for N-channel input
        original_conv1 = base.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Adapt pretrained 3-channel weights to N channels
            with torch.no_grad():
                # Average across 3 input channels → 1 channel
                avg_weight = original_conv1.weight.mean(dim=1, keepdim=True)
                # Repeat to fill N channels
                self.conv1.weight = nn.Parameter(
                    avg_weight.repeat(1, in_channels, 1, 1)
                )

        # Copy remaining layers (everything except conv1 and fc)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SAR tensor [B, in_channels, H, W]

        Returns:
            Feature vector [B, 512]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512]
        return x


class OpticalBranch(nn.Module):
    """
    Optical feature extractor based on ResNet18.

    Standard 3-channel RGB input, no modification needed.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Use all layers except the final FC
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Optical tensor [B, 3, H, W]

        Returns:
            Feature vector [B, 512]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 512]
        return x


class DualBranchCNN(nn.Module):
    """
    Dual-Branch CNN for building height estimation.

    Fuses SAR and optical features via concatenation, then regresses
    building height in meters.

    Architecture:
        SAR (512-d) + Optical (512-d) = 1024-d
        → FC(1024, 256) → ReLU → Dropout
        → FC(256, 64) → ReLU → Dropout
        → FC(64, 1) → Height prediction
    """

    def __init__(
        self,
        sar_channels: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.sar_branch = SARBranch(in_channels=sar_channels, pretrained=pretrained)
        self.optical_branch = OpticalBranch(pretrained=pretrained)

        # Fusion + regression head
        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sar: SAR tensor [B, C_sar, H, W]
            optical: Optical tensor [B, 3, H, W]

        Returns:
            Predicted heights [B, 1]
        """
        sar_features = self.sar_branch(sar)        # [B, 512]
        optical_features = self.optical_branch(optical)  # [B, 512]

        # Concatenate features
        fused = torch.cat([sar_features, optical_features], dim=1)  # [B, 1024]

        # Regress height
        height = self.fusion(fused)  # [B, 1]
        return height

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a brief model summary."""
        total = self.get_num_params()
        return (
            f"DualBranchCNN\n"
            f"  SAR Branch:     ResNet18 ({self.sar_branch.in_channels}-ch input)\n"
            f"  Optical Branch: ResNet18 (3-ch input)\n"
            f"  Fusion:         1024 → 256 → 64 → 1\n"
            f"  Total params:   {total:,}\n"
        )
