import torch.nn as nn
import math
import torch


# reference https://openjournals.uwaterloo.ca/index.php/vsl/article/view/3533/4579
class PositionalEncoding2D(nn.Module):

    def __init__(self, max_height: int, max_width: int, d_model: int):
        super().__init__()

        position_x = torch.arange(max_height).unsqueeze(1).broadcast_to((max_height, max_width)).unsqueeze(2)
        position_y = torch.arange(max_width).unsqueeze(0).broadcast_to((max_height, max_width)).unsqueeze(2)

        div_term = torch.exp(torch.arange(0, d_model, 4) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_height, max_width, d_model)
        pe[:, :, 0:2 * len(div_term):2] = torch.sin(position_x * div_term)
        pe[:, :, 1:2 * len(div_term):2] = torch.cos(position_x * div_term)
        pe[:, :, 2 * len(div_term)::2] = torch.sin(position_y * div_term)
        pe[:, :, 2 * len(div_term)::2] = torch.cos(position_y * div_term)
        self.register_buffer('pe', pe)
        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Dropout(x + self.pe[:x.size(1), :x.size(2)])
