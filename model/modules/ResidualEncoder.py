import torch
import torch.nn as nn
import model.modules.ResidualBlock as ResidualBlock

class ResidualEncoder(nn.Module):
    def __init__(self, block_count: int, in_channels: int, intermediary_channels: int):
        super(ResidualEncoder, self).__init__()
        self.res_layers = torch.nn.Sequential(
            *[ResidualBlock(in_channels=in_channels, intermediary_channels=intermediary_channels) for _ in range(block_count)],
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=1, stride=1, padding='same', bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X_board: torch.Tensor) -> torch.Tensor:
        return self.res_layers(X_board)

