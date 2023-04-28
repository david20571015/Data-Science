import argparse
import os

import torch
import torch.backends.cudnn
import torch.utils.data
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanMetric
import torchvision.transforms as T
from tqdm import tqdm

from src.datasets import TrainDataset
from src.models import CSRNet


def main():
    transforms = T.Compose([
        T.RandomCrop((512, 512)),
        T.RandomHorizontalFlip(),
    ])

    train_dataset = TrainDataset(args.image_dir,
                                 args.gt_dir,
                                 transforms=transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=8,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)

    model = CSRNet().to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_train_loss = float('inf')
    best_state_dict = None

    for epoch in range(args.epoch):
        print(f'Epoch {epoch}')

        train_loss = MeanMetric().to(DEVICE)
        train_mae = MeanAbsoluteError().to(DEVICE)

        model.train()
        with tqdm(train_dataloader, dynamic_ncols=True) as pbar:
            for image, density_map in pbar:
                image, density_map = image.to(DEVICE), density_map.to(DEVICE)

                pred = model(image)
                loss = loss_fn(pred, density_map)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss.update(loss)

                    pred_count = torch.flatten(pred, 1).relu().sum(-1)
                    gt_count = torch.flatten(density_map, 1).sum(-1)

                    train_mae.update(pred_count, gt_count)

                pbar.set_postfix_str(f'loss: {train_loss.compute():.5f}, '
                                     f'mae: {train_mae.compute():.5f}')

            scheduler.step()

        if train_loss.compute() < best_train_loss:
            best_train_loss = train_loss.compute()
            best_state_dict = model.state_dict()
            torch.save(best_state_dict,
                       f'model_best_{train_mae.compute().item():.2f}.pth')

    torch.save(model.state_dict(), 'model_last.pth')


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--image-dir',
                        default=os.path.join('data', 'train'),
                        type=str)
    parser.add_argument('--gt-dir',
                        default=os.path.join('data', 'train', 'dm10'),
                        type=str)
    args = parser.parse_args()

    main()
