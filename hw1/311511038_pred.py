import click
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models import efficientnet_b2
from torchvision.models import EfficientNet_B2_Weights
from tqdm import tqdm

BACKBONE = efficientnet_b2
WEIGHTS = EfficientNet_B2_Weights.DEFAULT


@click.command()
@click.argument('image-path-file', type=click.File('r'))
def predict(image_path_file):
    image_paths = [line.strip() for line in image_path_file]

    transform = T.Compose([
        WEIGHTS.transforms(),
    ])

    model = torch.nn.Sequential(
        BACKBONE(weights=WEIGHTS),
        torch.nn.AdaptiveAvgPool1d(1),
    )
    model.load_state_dict(
        torch.load('model_efficientnet_b2.pth',
                   map_location=torch.device('cpu')))

    results = []

    model.eval()
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            image = torchvision.io.read_image(image_path)
            image = transform(image)
            image = image.unsqueeze(0)
            output = model(image)
            result = torch.where(output > 0, 1, 0)
            results.append(result.item())

    with open('311511038.txt', 'w', encoding='utf-8') as file:
        print(*results, sep='', file=file)


if __name__ == '__main__':
    predict(None)
