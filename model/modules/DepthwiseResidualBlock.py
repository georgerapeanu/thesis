import torch.nn as nn
import torch
import math

class DepthwiseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, intermediary_channels: int):
        super(DepthwiseResidualBlock, self).__init__()
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding='same', bias=True, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=intermediary_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(intermediary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediary_channels, out_channels=intermediary_channels, kernel_size=3, padding='same', bias=True, groups=intermediary_channels),
            nn.Conv2d(in_channels=intermediary_channels, out_channels=in_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.final_step = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.final_step(self.residual_layer(X) + X)
