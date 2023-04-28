import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision.ops import Conv2dNormActivation
import torchvision
import torchvision.transforms as T


class Frontend(nn.Sequential):

    def __init__(self) -> None:
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.DEFAULT)

        super().__init__(*list(vgg.features.children())[:23])

        # super().__init__(
        #     Conv2dNormActivation(input_ch, 64, kernel_size=3),
        #     Conv2dNormActivation(64, 64, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2),
        #     Conv2dNormActivation(64, 128, kernel_size=3),
        #     Conv2dNormActivation(128, 128, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2),
        #     Conv2dNormActivation(128, 256, kernel_size=3),
        #     Conv2dNormActivation(256, 256, kernel_size=3),
        #     Conv2dNormActivation(256, 256, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2),
        #     Conv2dNormActivation(256, 512, kernel_size=3),
        #     Conv2dNormActivation(512, 512, kernel_size=3),
        #     Conv2dNormActivation(512, 512, kernel_size=3),
        # )


class Backend(nn.Sequential):

    def __init__(self) -> None:
        super().__init__(
            Conv2dNormActivation(512, 512, kernel_size=3, dilation=2),
            Conv2dNormActivation(512, 512, kernel_size=3, dilation=2),
            Conv2dNormActivation(512, 512, kernel_size=3, dilation=2),
            Conv2dNormActivation(512, 256, kernel_size=3, dilation=2),
            Conv2dNormActivation(256, 128, kernel_size=3, dilation=2),
            Conv2dNormActivation(128, 64, kernel_size=3, dilation=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CSRNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.transforms = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        self.frontend = Frontend()
        self.backend = Backend()

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[-2:]
        x = self.transforms(x)
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.interpolate(x, size=output_shape, mode='bilinear')
        return x


if __name__ == '__main__':
    inputs = torch.randn(8, 3, 64, 64, device='cuda')

    model = CSRNet(3).to('cuda')

    assert model(inputs).shape[-2:] == inputs.shape[-2:]

    summary(model, inputs.shape, depth=2)

    # vgg = torchvision.models.vgg16(
    #     weights=torchvision.models.VGG16_Weights.DEFAULT).to('cuda')

    # print(nn.Sequential(*list(vgg.features.children())[:23]))
