import argparse
import os

import pandas as pd
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm

from src.datasets import InferDataset
from src.models import VGG


def main():
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = InferDataset(args.data, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

    model = VGG().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    result = torch.zeros(len(test_dataset), dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.inference_mode(True), tqdm(test_dataloader) as pbar:
        for file_id, image in pbar:
            image, file_id = image.to(DEVICE), file_id.to(DEVICE)

            try:
                pred = model(image)
                result[file_id - 1] = pred.sum()
            except:
                pass

    df = pd.DataFrame({'Count': result.tolist()})
    df.index += 1
    df.to_csv('result.csv', index_label='ID')


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('--data',
                        default=os.path.join('data', 'test'),
                        type=str,
                        help='path to dataset which contains images(*.jpg)')
    # model
    parser.add_argument('--weights',
                        default='best_model.pth',
                        type=str,
                        help='path to weights')

    args = parser.parse_args()

    main()
