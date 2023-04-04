import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvNet(nn.Module):

    def __init__(
        self,
        in_ch=3,
        chs: list[int] = [64, 128, 128],
        emb_size=64,
    ):
        super().__init__()

        dims = [in_ch] + chs
        conv_blocks = [
            conv_block(in_ch, out_ch)
            for in_ch, out_ch in zip(dims[:-1], dims[1:])
        ]

        self.convs = nn.Sequential(*conv_blocks)
        self.linear = nn.Linear(dims[-1], emb_size)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        x = F.avg_pool2d(x, x.shape[-2:]).view(x.shape[0], -1)
        x = self.linear(x)
        return x
