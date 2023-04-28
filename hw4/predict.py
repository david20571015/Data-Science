import argparse
import os

import pandas as pd
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms as T
from tqdm import tqdm

from src.datasets import InferDataset, four_crop
from src.models import CSRNet


def main():
    test_dataset = InferDataset(args.test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  pin_memory=True)

    model = CSRNet().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    result = torch.zeros(len(test_dataset), dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad(), tqdm(test_dataloader) as pbar:
        for file_id, image in pbar:
            image, file_id = image.to(DEVICE), file_id.to(DEVICE)

            try:
                pred = model(image)
                result[file_id] = pred.relu().sum()
            except:
                image = four_crop(image)
                for i in range(4):
                    pred = model(image[i])
                    result[file_id] += pred.relu().sum()

    df = pd.DataFrame({'Count': result.cpu().tolist()})
    df.to_csv('result.csv', index_label='ID')


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', default='model.pth', type=str)
    parser.add_argument('--test-data',
                        default=os.path.join('data', 'test'),
                        type=str)
    args = parser.parse_args()

    main()
