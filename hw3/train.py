import argparse
import copy
from datetime import datetime
import os

import torch
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.dataset import TrainDataset
from src.model import ConvNet
from src.sampler import CategoriesSampler
from src.utils import euclidean_distance_sqaured


def main():
    train_dataset = TrainDataset(args.train_data)
    train_sampler = CategoriesSampler(train_dataset.labels,
                                      n_classes=args.train_n_class,
                                      n_batch=400,
                                      n_way=args.train_n_way,
                                      n_shot=args.n_shot,
                                      n_query=args.n_query)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True)

    valid_dataset = TrainDataset(args.valid_data)
    valid_sampler = CategoriesSampler(valid_dataset.labels,
                                      n_classes=args.valid_n_class,
                                      n_batch=400,
                                      n_way=args.valid_n_way,
                                      n_shot=args.n_shot,
                                      n_query=args.n_query)
    valid_loader = DataLoader(valid_dataset,
                              batch_sampler=valid_sampler,
                              num_workers=4,
                              pin_memory=True)

    model = ConvNet(chs=args.dims, emb_size=args.emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
                                                gamma=0.5)

    best_valid_acc = 0
    best_state_dict = None

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')

        train_loss = MeanMetric().to(DEVICE)
        train_acc = Accuracy(task='multiclass',
                             num_classes=args.train_n_way).to(DEVICE)
        model.train()
        with tqdm(train_loader, desc='train', dynamic_ncols=True) as pbar:
            for data, _ in pbar:
                # data.shape: (n_way, n_shot + n_query, channel, width, height)
                data = data.to(DEVICE)
                support, query = data[:, :args.n_shot], data[:, args.n_shot:]

                support = support.reshape(-1, *support.shape[-3:])
                proto = model(support)
                proto = proto.reshape(args.train_n_way, args.n_shot, -1)
                proto = proto.mean(dim=1)
                # proto.shape: (n_way, -1)

                query = query.reshape(-1, *query.shape[-3:])
                output = model(query)
                output = output.reshape(args.train_n_way * args.n_query, -1)

                # logits.shape: (n_way * n_query, n_way)
                logits = -euclidean_distance_sqaured(output, proto)

                labels = torch.arange(
                    args.train_n_way,
                    dtype=torch.long,
                    device=DEVICE,
                ).repeat(args.n_query, 1).t().reshape(-1)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.update(loss)
                train_acc.update(logits.argmax(dim=-1), labels)

                pbar.set_postfix_str(
                    f'loss: {train_loss.compute():.4f}, acc: {train_acc.compute():.4f}'
                )

        scheduler.step()

        valid_loss = MeanMetric().to(DEVICE)
        valid_acc = Accuracy(task='multiclass',
                             num_classes=args.valid_n_way).to(DEVICE)

        model.eval()
        with tqdm(valid_loader, desc='valid',
                  dynamic_ncols=True) as pbar, torch.no_grad():
            for data, _ in pbar:
                # data.shape: (n_way, n_shot + n_query, channel, width, height)
                data = data.to(DEVICE)
                support, query = data[:, :args.n_shot], data[:, args.n_shot:]

                support = support.reshape(-1, *support.shape[-3:])
                proto = model(support)
                proto = proto.reshape(args.valid_n_way, args.n_shot, -1)
                proto = proto.mean(dim=1)
                # proto.shape: (n_way, -1)

                query = query.reshape(-1, *query.shape[-3:])
                output = model(query)
                output = output.reshape(args.valid_n_way * args.n_query, -1)

                # logits.shape: (n_way * n_query, n_way)
                logits = -euclidean_distance_sqaured(output, proto)

                labels = torch.arange(
                    args.valid_n_way,
                    dtype=torch.long,
                    device=DEVICE,
                ).repeat(args.n_query, 1).t().reshape(-1)

                loss = F.cross_entropy(logits, labels)

                valid_loss.update(loss)
                valid_acc.update(logits.argmax(dim=-1), labels)

                pbar.set_postfix_str(
                    f'loss: {valid_loss.compute():.4f}, acc: {valid_acc.compute():.4f}'
                )

        if valid_acc.compute() > best_valid_acc:
            best_valid_acc = valid_acc.compute()
            best_state_dict = copy.deepcopy(model.state_dict())

    print(f'Best valid acc: {best_valid_acc:.4f}')
    torch.save(best_state_dict,
               f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--train-data',
                        default=os.path.join('data', 'train.pkl'),
                        type=str,
                        help='path to train pkl data')
    parser.add_argument('--valid-data',
                        default=os.path.join('data', 'validation.pkl'),
                        type=str,
                        help='path to validation pkl data')
    # Model
    parser.add_argument('--dims',
                        nargs='+',
                        default=[64, 64, 64],
                        type=int,
                        help='channels of hidden conv layers')
    parser.add_argument('--emb-dim',
                        default=128,
                        type=int,
                        help='embedding dimension (output dim of the model)')
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train-n-class',
                        default=64,
                        type=int,
                        help='number of total classes in the training dataset')
    parser.add_argument('--valid-n-class',
                        default=16,
                        type=int,
                        help='number of total classes in the valid dataset')
    parser.add_argument('--train-n-way',
                        default=20,
                        type=int,
                        help='number of classes in a training episode')
    parser.add_argument('--valid-n-way',
                        default=5,
                        type=int,
                        help='number of classes in a valid episode')
    parser.add_argument('--n-shot',
                        default=5,
                        type=int,
                        help='number of support examples per class')
    parser.add_argument('--n-query',
                        default=5,
                        type=int,
                        help='number of query examples per class')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True

    main()
