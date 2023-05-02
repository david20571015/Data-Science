import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class VGG(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT)

        self.features = nn.Sequential(*list(vgg.features.children())[:-1])

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.reg_layer(x)
        x = torch.abs(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(8, 3, 64, 64, device='cuda')
    expected_shape = (8, 1, 8, 8)

    model = VGG().to('cuda')

    assert model(inputs).shape == expected_shape
