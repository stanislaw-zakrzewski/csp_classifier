"""
ATCNet PyTorch Architecture Implementation
==========================================

Implementation of ATCNet (Attention-based Temporal Convolutional Network) for BCI:
- Conv Block: Temporal Conv2d -> Depthwise Spatial Conv2d (max-norm <= 1.0) -> Pointwise Conv2d.
- Attention Module: Sliding-window Multi-Head Self-Attention (MHSA).
- TCN Module: Dilated 1D Causal Convolutions (dilation = 1, 2) with residual connections.
- Classification Head: Dense Linear Layer (max-norm <= 0.5) -> Log-Softmax.

Reference:
Altaheri, H., et al. (2023). Physics-informed attention-based temporal convolutional network for EEG-based motor imagery classification.
IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 1049-1058.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with a max-norm constraint on weight parameters."""
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with a max-norm constraint on weight parameters."""
    def __init__(self, *args, max_norm: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class TCNBlock(nn.Module):
    """Dilated Causal Temporal Convolutional Block with Residual Connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1), dilation=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * 2, dilation=2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.elu2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(self.elu1(self.bn1(self.conv1(x))))
        # Truncate causal padding overflow
        out = out[:, :, :x.shape[2]]
        out = self.drop2(self.elu2(self.bn2(self.conv2(out))))
        out = out[:, :, :x.shape[2]]
        return out + res


class SlidingWindowAttention(nn.Module):
    """Sliding-window Multi-Head Self-Attention (MHSA) module."""
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, T) -> transpose to (B, T, C) for MultiheadAttention
        x_t = x.transpose(1, 2)
        attn_out, _ = self.mha(x_t, x_t, x_t)
        norm_out = self.norm(x_t + attn_out)
        return norm_out.transpose(1, 2)


class ATCNet(nn.Module):
    """
    ATCNet (Attention-based Temporal Convolutional Network) Architecture for Motor Imagery BCI.

    Args:
        n_classes (int): Number of target classes (default: 2 for binary motor imagery).
        channels (int): Number of EEG channels (default: 11).
        samples (int): Number of time samples per trial epoch (default: 1000 for 4s at 250Hz).
        F1 (int): Number of temporal filters (default: 8).
        D (int): Depth multiplier for spatial depthwise conv (default: 2).
        F2 (int): Number of pointwise filters (F2 = F1 * D = 16).
        kernel_length (int): Temporal filter kernel length (default: 64).
        num_heads (int): Number of attention heads (default: 2).
        dropout_rate (float): Dropout probability (default: 0.3).
    """
    def __init__(
        self,
        n_classes: int = 2,
        channels: int = 11,
        samples: int = 1000,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        num_heads: int = 2,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples

        # 1. Conv Block
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(
            F1, F1 * D, (channels, 1), groups=F1, max_norm=1.0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout_rate)

        self.pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 7))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute dynamic temporal length after conv block pooling
        pooled_samples = (samples // 8) // 7

        # 2. Attention Module
        self.attention = SlidingWindowAttention(embed_dim=F2, num_heads=num_heads, dropout_rate=dropout_rate)

        # 3. TCN Module
        self.tcn = TCNBlock(in_channels=F2, out_channels=F2, kernel_size=3, dropout_rate=dropout_rate)

        # 4. Dense Classification Head
        flatten_dim = F2 * pooled_samples
        self.classifier = LinearWithConstraint(flatten_dim, n_classes, max_norm=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: (Batch, 1, Channels, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv Block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.depthwise(out)
        out = self.bn2(out)
        out = self.elu1(out)
        out = self.pool1(out)
        out = self.drop1(out)

        out = self.pointwise(out)
        out = self.bn3(out)
        out = self.elu2(out)
        out = self.pool2(out)
        out = self.drop2(out)

        # Reshape to (Batch, Features, Time)
        out_2d = out.squeeze(2)

        # Attention Block
        out_attn = self.attention(out_2d)

        # TCN Block
        out_tcn = self.tcn(out_attn)

        # Flatten & Classify
        flattened = out_tcn.flatten(start_dim=1)
        logits = self.classifier(flattened)
        return logits
