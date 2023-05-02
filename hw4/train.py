import argparse
import os

import torch
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import random_split
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanMetric
import torchvision.transforms as T
from tqdm import tqdm

from src.datasets import TrainDataset
from src.losses import compute_loss
from src.models import VGG


def main():
    transforms = T.Compose([
        T.RandomCrop((512, 512)),
        T.RandomHorizontalFlip(),
    ])

    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset, valid_dataset = random_split(
        TrainDataset(args.data_dir, transforms=transforms,
                     transform=transform), [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=8,
                                                   shuffle=True,
                                                   num_workers=3,
                                                   pin_memory=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=16,
                                                   shuffle=False,
                                                   num_workers=3,
                                                   pin_memory=True)

    model = VGG().to(DEVICE)
    # model.load_state_dict(torch.load('best_model.pth'))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid_mae = float('inf')

    for epoch in range(args.epoch):
        print(f'Epoch {epoch}')

        train_loss = MeanMetric().to(DEVICE)
        train_mae = MeanAbsoluteError().to(DEVICE)

        model.train()
        with tqdm(train_dataloader, dynamic_ncols=True) as pbar:
            for images, gts in pbar:
                images, gts = images.to(DEVICE), gts.to(DEVICE)

                preds = model(images)

                total_loss = torch.tensor(0.0, device=DEVICE)
                for pred, gt in zip(preds, gts):
                    loss = compute_loss(pred, gt, sigma=args.sigma)
                    total_loss += loss
                loss = total_loss / len(preds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss.update(loss)

                    pred_count = preds.view(preds.size(0), -1).sum(-1)
                    gt_count = gts.view(gts.size(0), -1).sum(-1)

                    train_mae.update(pred_count, gt_count)

                pbar.set_postfix_str(f'loss: {train_loss.compute():.5f}, '
                                     f'mae: {train_mae.compute():.5f}')

        valid_mae = MeanAbsoluteError().to(DEVICE)

        model.eval()
        with torch.inference_mode(True), tqdm(valid_dataloader,
                                              dynamic_ncols=True) as pbar:
            for images, gts in pbar:
                images, gts = images.to(DEVICE), gts.to(DEVICE)

                preds = model(images)

                pred_count = preds.view(preds.size(0), -1).sum(-1)
                gt_count = gts.view(gts.size(0), -1).sum(-1)

                valid_mae.update(pred_count, gt_count)

                pbar.set_postfix_str(f'mae: {valid_mae.compute():.5f}')

        if valid_mae.compute() < best_valid_mae:
            best_valid_mae = valid_mae.compute()
            torch.save(model.state_dict(),
                       f'model_{valid_mae.compute().item():.2f}.pth')

    torch.save(model.state_dict(), 'last_model.pth')


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument(
        '--data-dir',
        default=os.path.join('data_processed', 'train'),
        type=str,
        help=
        'path to training data image directory which contains images(*.jpg) and points(*.npy)'
    )

    # training
    parser.add_argument('--epoch', default=500, type=int, help='epoch')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight-decay',
                        default=1e-4,
                        type=float,
                        help='weight decay')
    parser.add_argument('--sigma',
                        default=8.0,
                        type=float,
                        help='sigma for gaussian kernel')

    args = parser.parse_args()

    main()
