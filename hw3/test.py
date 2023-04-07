import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import InferDataset
from src.model import ConvNet
from src.utils import euclidean_distance_sqaured


def predict():
    infer_dataset = InferDataset(args.test_data)
    infer_loader = DataLoader(infer_dataset,
                              batch_size=1,
                              num_workers=4,
                              pin_memory=True)

    model = ConvNet(chs=args.dims, emb_size=args.emb_dim).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    result = []

    model.eval()
    with torch.no_grad():
        for sup_images, sup_labels, qry_images in tqdm(infer_loader):

            sup_images = sup_images.squeeze(0).to(DEVICE)
            sup_labels = sup_labels.squeeze(0).to(DEVICE)
            qry_images = qry_images.squeeze(0).to(DEVICE)

            proto = model(sup_images)
            proto = torch.stack(
                [proto[sup_labels == i].mean(0) for i in range(5)])

            qry_emb = model(qry_images)
            logits = euclidean_distance_sqaured(qry_emb, proto)

            pred = logits.argmin(dim=1)

            result.extend(pred.tolist())

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--test-data',
                        default=os.path.join('data', 'test.pkl'),
                        type=str,
                        help='path to test pkl data. default: data/test.pkl')
    # Model
    parser.add_argument(
        '--dims',
        nargs='+',
        default=[64, 64, 64],
        type=int,
        help='channels of hidden conv layers. default: [64, 64, 64]')
    parser.add_argument(
        '--emb-dim',
        default=64,
        type=int,
        help='embedding dimension (output dim of the model). default: 64')
    parser.add_argument('--weights',
                        default='model.pth',
                        type=str,
                        help='path to model weights. default: model.pth')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = predict()

    df = pd.DataFrame({'category': results})
    df.to_csv('pred.csv', index_label='id')
