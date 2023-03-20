import argparse

import pandas as pd
import torch
import torch.utils.data
from torchmetrics import Accuracy
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as T
from tqdm import tqdm


def predict(model: torch.nn.Module, testloader: torch.utils.data.DataLoader):
    acc_metric = Accuracy(task='multiclass',
                          num_classes=NUM_CLASSES).to(DEVICE)

    model.eval()
    with torch.no_grad():
        preds = []
        for data, target in tqdm(testloader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)

            pred = torch.argmax(output, dim=-1)
            preds.extend(pred.tolist())

            acc_metric.update(output, target)

    return acc_metric.compute(), preds


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # type: ignore

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--weight_path', '-w', type=str, required=True)
    args = arg_parser.parse_args()
    weight_path = args.weight_path

    model = resnet18(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

    transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, ))
    ])
    testset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1024,
                                             shuffle=False,
                                             num_workers=6,
                                             pin_memory=True)

    acc, preds = predict(model, testloader)

    print(f'Accuracy: {acc:.4f}')

    df = pd.DataFrame({'pred': preds})
    df.to_csv('pred.csv', index_label='id')
