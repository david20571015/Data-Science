import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import InferDataset
from src.model import ConvNet
from src.utils import euclidean_distance_sqaured


def predict():
    infer_dataset = InferDataset('data/test.pkl')
    infer_loader = DataLoader(infer_dataset,
                              batch_size=1,
                              num_workers=4,
                              pin_memory=True)

    model = ConvNet(emb_size=args.emb_dim).to(DEVICE)
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
    parser.add_argument('--weights', type=str, default='model.pth')
    parser.add_argument('--emb-dim', type=int, default=128)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result = predict()

    df = pd.DataFrame({'category': result})
    df.to_csv('pred.csv', index_label='id')
