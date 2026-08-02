"""
EEGNet PyTorch Architecture Implementation
===========================================

Implementation of EEGNet-8,2 (Lawhern et al., 2018):
- Block 1: 2D Temporal Convolution -> Depthwise Spatial Convolution (kernel size C x 1, max-norm <= 1.0) -> BatchNorm -> ELU -> AvgPool -> Dropout.
- Block 2: Separable Convolution (Pointwise + Spatial) -> BatchNorm -> ELU -> AvgPool -> Dropout.
- Classification Head: Dense Linear Layer -> Softmax / Log-Softmax.

Reference:
Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013.
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
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    EEGNet-8,2 Architecture for Motor Imagery BCI Classification.
    
    Args:
        n_classes (int): Number of target classes (default: 2 for binary motor imagery).
        channels (int): Number of EEG channels (default: 11 for selected C-line channels).
        samples (int): Number of time samples per trial epoch (default: 750 for 3s epoch at 250Hz).
        F1 (int): Number of temporal filters (default: 8).
        D (int): Depth multiplier for spatial depthwise conv (default: 2).
        F2 (int): Number of pointwise filters (F2 = F1 * D = 16).
        kernel_length (int): Temporal filter kernel length (default: 64, ~half sampling rate).
        dropout_rate (float): Dropout probability (default: 0.25).
    """
    def __init__(
        self,
        n_classes: int = 2,
        channels: int = 11,
        samples: int = 750,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25
    ):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # --- Block 1: Temporal Conv + Depthwise Spatial Conv ---
        # Input shape: (Batch, 1, Channels, Samples)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv: Filters each channel independently
        self.depthwise = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # --- Block 2: Separable Conv (Depthwise + Pointwise) ---
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # --- Dense Classification Head ---
        # Calculate output feature dimensions after pooling
        self.feature_len = self.get_feature_size()
        self.classifier = LinearWithConstraint(self.feature_len, n_classes, max_norm=0.25)

    def get_feature_size(self) -> int:
        """Compute flat feature dimension after convolutional layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.channels, self.samples)
            x = self.conv1(dummy)
            x = self.batchnorm1(x)
            x = self.depthwise(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            
            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x shape: (Batch, 1, Channels, Samples) or (Batch, Channels, Samples)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dim: (Batch, 1, Channels, Samples)

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten & Classify
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Smoke Test
    model = EEGNet(n_classes=2, channels=11, samples=750)
    test_input = torch.randn(16, 1, 11, 750)
    output = model(test_input)
    print(f"EEGNet initialized successfully!")
    print(f"Input shape : {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
